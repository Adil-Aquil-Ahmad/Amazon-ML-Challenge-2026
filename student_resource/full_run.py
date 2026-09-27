import os, re, time, gc
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import lightgbm as lgb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TOP_K = 50
TFIDF_BATCH = 5000
NEG_RATIO = 5
SUFFIX_MAP = {'\\bincorporated\\b': 'inc', '\\bcorporation\\b': 'corp', '\\blimited\\b': 'ltd', '\\bprivate\\b': 'pvt', '\\bcompany\\b': 'co', '\\benterprises\\b': 'ent', '\\bservices\\b': 'svcs', '\\binternational\\b': 'intl', '\\bmanagement\\b': 'mgmt', '\\bgroup\\b': 'grp', '\\bbrothers\\b': 'bros'}
ADDR_MAP = {'\\bstreet\\b': 'st', '\\broad\\b': 'rd', '\\bavenue\\b': 'ave', '\\bboulevard\\b': 'blvd', '\\bdrive\\b': 'dr', '\\blane\\b': 'ln', '\\bcourt\\b': 'ct', '\\bplace\\b': 'pl', '\\bsuite\\b': 'ste', '\\bapartment\\b': 'apt', '\\bfloor\\b': 'fl', '\\bnorth\\b': 'n', '\\bsouth\\b': 's', '\\beast\\b': 'e', '\\bwest\\b': 'w'}

def normalize(text, abbrev_map=None):
    if not isinstance(text, str):
        return ''
    t = text.lower().strip()
    t = re.sub('[^a-z0-9\\s]', ' ', t)
    t = re.sub('\\s+', ' ', t).strip()
    if abbrev_map:
        for pat, rep in abbrev_map.items():
            t = re.sub(pat, rep, t)
    return t

def preprocess(df):
    df = df.copy()
    df['norm_name'] = df['business_name'].apply(lambda x: normalize(x, SUFFIX_MAP))
    df['norm_address'] = df['business_address'].apply(lambda x: normalize(x, ADDR_MAP))
    df['combined'] = df['norm_name'] + ' ' + df['norm_address']
    df['country_lower'] = df['country'].str.lower().str.strip()
    return df

def load_tsv(path):
    return pd.read_csv(path, sep='\t', dtype=str).fillna('')
FEATURE_COLS = ['name_ratio', 'name_partial', 'name_token_sort', 'name_token_set', 'addr_ratio', 'addr_partial', 'addr_token_sort', 'addr_token_set', 'combined_ratio', 'combined_token_set', 'name_jaccard', 'addr_jaccard', 'same_country', 'src_flag', 'addr_first_match', 'name_len_diff']

def jaccard(s1, s2):
    t1, t2 = (set(s1.split()), set(s2.split()))
    if not t1 and (not t2):
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def pair_features(r1, r2):
    n1, n2 = (r1['norm_name'], r2['norm_name'])
    a1, a2 = (r1['norm_address'], r2['norm_address'])
    c1, c2 = (r1['combined'], r2['combined'])
    af1 = a1.split()[0] if a1.split() else ''
    af2 = a2.split()[0] if a2.split() else ''
    max_nlen = max(len(n1), len(n2), 1)
    return {'name_ratio': fuzz.ratio(n1, n2) / 100, 'name_partial': fuzz.partial_ratio(n1, n2) / 100, 'name_token_sort': fuzz.token_sort_ratio(n1, n2) / 100, 'name_token_set': fuzz.token_set_ratio(n1, n2) / 100, 'addr_ratio': fuzz.ratio(a1, a2) / 100, 'addr_partial': fuzz.partial_ratio(a1, a2) / 100, 'addr_token_sort': fuzz.token_sort_ratio(a1, a2) / 100, 'addr_token_set': fuzz.token_set_ratio(a1, a2) / 100, 'combined_ratio': fuzz.ratio(c1, c2) / 100, 'combined_token_set': fuzz.token_set_ratio(c1, c2) / 100, 'name_jaccard': jaccard(n1, n2), 'addr_jaccard': jaccard(a1, a2), 'same_country': int(r1['country_lower'] == r2['country_lower']), 'src_flag': 0 if r2['entity_id'].startswith('S2') else 1, 'addr_first_match': int(af1 == af2 and af1 != ''), 'name_len_diff': abs(len(n1) - len(n2)) / max_nlen}

def tfidf_blocking(df_s1, df_s23, top_k=TOP_K, batch_size=TFIDF_BATCH):
    countries = df_s1['country_lower'].unique()
    candidates = {}
    for sid in df_s1['entity_id']:
        candidates[sid] = set()
    for country in countries:
        s1_c = df_s1[df_s1['country_lower'] == country].reset_index(drop=True)
        s23_c = df_s23[df_s23['country_lower'] == country].reset_index(drop=True)
        if s1_c.empty or s23_c.empty:
            continue
        print(f'  [{country}] S1={len(s1_c):,}  S23={len(s23_c):,}')
        ids_s23 = s23_c['entity_id'].tolist()
        corpus_s23 = s23_c['combined'].tolist()
        corpus_s1 = s1_c['combined'].tolist()
        ids_s1 = s1_c['entity_id'].tolist()
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=200000, min_df=2, sublinear_tf=True)
        mat_s23 = vectorizer.fit_transform(corpus_s23)
        for start in tqdm(range(0, len(ids_s1), batch_size), desc=f'  Blocking {country}'):
            end = min(start + batch_size, len(ids_s1))
            b_ids = ids_s1[start:end]
            b_txt = corpus_s1[start:end]
            mat_b = vectorizer.transform(b_txt)
            sim = cosine_similarity(mat_b, mat_s23)
            for i, s1_id in enumerate(b_ids):
                row = sim[i]
                k = min(top_k, len(row))
                top_idx = np.argpartition(-row, k)[:k]
                top_idx = top_idx[np.argsort(-row[top_idx])]
                top_idx = [j for j in top_idx if row[j] > 0.01]
                candidates[s1_id] = set((ids_s23[j] for j in top_idx))
        del mat_s23, vectorizer, s1_c, s23_c
        gc.collect()
    return candidates

def main():
    t0 = time.time()
    print('\n[1/7] Loading data...')
    train_s1 = preprocess(load_tsv(os.path.join(TRAIN_DIR, 'train_source1.tsv')))
    train_s2 = preprocess(load_tsv(os.path.join(TRAIN_DIR, 'train_source2.tsv')))
    train_s3 = preprocess(load_tsv(os.path.join(TRAIN_DIR, 'train_source3.tsv')))
    train_gt = load_tsv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'))
    test_s1 = preprocess(load_tsv(os.path.join(TEST_DIR, 'test_source1.tsv')))
    test_s2 = preprocess(load_tsv(os.path.join(TEST_DIR, 'test_source2.tsv')))
    test_s3 = preprocess(load_tsv(os.path.join(TEST_DIR, 'test_source3.tsv')))
    train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)
    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
    print(f'  Train S1={len(train_s1):,}  S23={len(train_s23):,}')
    print(f'  Test  S1={len(test_s1):,}   S23={len(test_s23):,}')
    gt_map = {}
    for _, row in train_gt.iterrows():
        ids = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'].strip() else set()
        gt_map[row['source1_entity_id']] = ids
    print(f'\n[2/7] TF-IDF blocking on ALL {len(train_s1):,} training S1 entities...')
    tr_cands = tfidf_blocking(train_s1, train_s23, top_k=TOP_K, batch_size=TFIDF_BATCH)
    print(f'  Total train candidate pairs: {sum((len(v) for v in tr_cands.values())):,}')
    s1_samp = train_s1
    s23_samp = train_s23
    gc.collect()
    print(f'\n[3/7] Building training feature matrix from ALL {len(train_s1):,} S1 entities...')
    s1_lkp = s1_samp.set_index('entity_id').to_dict('index')
    s23_lkp = s23_samp.set_index('entity_id').to_dict('index')
    feat_rows, labels = ([], [])
    rng = np.random.default_rng(42)
    for s1_id, cands in tqdm(tr_cands.items(), desc='Pair features'):
        if s1_id not in s1_lkp or not cands:
            continue
        r1 = s1_lkp[s1_id]
        true_ids = gt_map.get(s1_id, set())
        pos = list(cands & true_ids)
        neg = list(cands - true_ids)
        for cid in pos:
            if cid in s23_lkp:
                feat_rows.append(pair_features(r1, s23_lkp[cid]))
                labels.append(1)
        n_neg = max(NEG_RATIO * len(pos), NEG_RATIO)
        for cid in rng.choice(neg, size=min(n_neg, len(neg)), replace=False):
            if cid in s23_lkp:
                feat_rows.append(pair_features(r1, s23_lkp[cid]))
                labels.append(0)
    X = pd.DataFrame(feat_rows)[FEATURE_COLS]
    y = np.array(labels)
    print(f'  Pairs: {len(X):,}  Positives: {y.sum():,} ({100 * y.mean():.1f}%)')
    del s1_lkp, s23_lkp, s1_samp, s23_samp, tr_cands
    gc.collect()
    print('\n[4/7] Training LightGBM...')
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scale_pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = lgb.train({'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': 63, 'scale_pos_weight': scale_pw, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'n_jobs': -1, 'seed': 42, 'verbose': -1}, lgb.Dataset(X_tr, label=y_tr), num_boost_round=500, valid_sets=[lgb.Dataset(X_val, label=y_val)], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
    val_proba = model.predict(X_val)
    best_thr, best_f05 = (0.5, 0.0)
    for thr in np.arange(0.1, 0.95, 0.05):
        yp = (val_proba >= thr).astype(int)
        p = precision_score(y_val, yp, zero_division=0)
        r = recall_score(y_val, yp, zero_division=0)
        f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
        if f > best_f05:
            best_f05, best_thr = (f, thr)
    print(f'  Best threshold: {best_thr:.2f}  |  Val F0.5: {best_f05:.3f}')
    model.save_model(os.path.join(OUTPUT_DIR, 'lgbm_model.txt'))
    del X, y, X_tr, X_val
    gc.collect()
    print(f'\n[5/7] TF-IDF blocking on FULL test set ({len(test_s1):,} S1 entities)...')
    test_cands = tfidf_blocking(test_s1, test_s23, top_k=TOP_K, batch_size=TFIDF_BATCH)
    print(f'  Total test candidate pairs: {sum((len(v) for v in test_cands.values())):,}')
    print('\n[6/7] Model inference on test candidates...')
    s1_lkp = test_s1.set_index('entity_id').to_dict('index')
    s23_lkp = test_s23.set_index('entity_id').to_dict('index')
    all_matches, all_cands = ({}, {})
    for s1_id in tqdm(test_s1['entity_id'], desc='Inference'):
        cids = list(test_cands.get(s1_id, set()))
        all_cands[s1_id] = set(cids)
        if not cids or s1_id not in s1_lkp:
            all_matches[s1_id] = set()
            continue
        r1 = s1_lkp[s1_id]
        feat_rows, valid = ([], [])
        for cid in cids:
            if cid in s23_lkp:
                feat_rows.append(pair_features(r1, s23_lkp[cid]))
                valid.append(cid)
        if not feat_rows:
            all_matches[s1_id] = set()
            continue
        proba = model.predict(pd.DataFrame(feat_rows)[FEATURE_COLS])
        all_matches[s1_id] = {cid for cid, p in zip(valid, proba) if p >= best_thr}
    n_with = sum((1 for v in all_matches.values() if v))
    print(f'  S1 with ≥1 match: {n_with:,} / {len(all_matches):,}')
    print('\n[7/7] Writing submission files...')
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    with open(match_path, 'w', encoding='utf-8') as fm, open(cand_path, 'w', encoding='utf-8') as fc:
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        fc.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in test_s1['entity_id']:
            m = all_matches.get(s1_id, set())
            c = all_cands.get(s1_id, set()) | m
            fm.write(f"{s1_id}\t{','.join(sorted(m))}\n")
            fc.write(f"{s1_id}\t{','.join(sorted(c))}\n")
    elapsed = (time.time() - t0) / 60
    print(f'\n✅ Done in {elapsed:.1f} min')
    print(f'   {match_path}')
    print(f'   {cand_path}')
    print('\nValidate with:')
    print('  python3 utils/validate_submission.py \\')
    print('      --matching output/matching_results.tsv \\')
    print('      --candidate output/candidate_pairs.tsv \\')
    print('      --test-dir dataset/test')
if __name__ == '__main__':
    main()
