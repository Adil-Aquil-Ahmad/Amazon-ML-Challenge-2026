import gc
import json
import os
import shutil
import time
import zlib
import numpy as np
import pandas as pd
import pyarrow as pa
import features
import text_norm
from postprocess import veto_mask
from blocking import TokenSpace, subset_csr, retrieve_union, pair_overlap, rerank_features, RERANK_COLS
from features import FEATURE_COLS, CTX_COLS, STAGE2_COLS, compute_features, record_meta, derive_cols, stage2_features
DEFAULTS = dict(n_threads=os.cpu_count() or 8, top_k1=250, top_k=30, key_cap=2000, train_frac=0.2, val_frac=0.05, rr_sample=40000, batch_rows=25000, feat_chunk=2000000, seed=42, novelty=False, clean_support=False, table_keep_count=5, decoy_veto=True)
_T0 = time.time()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - _T0) / 60:5.1f}m] {msg}", flush=True)

def load_tsv(path):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_values=[], quoting=3)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].fillna('').str.strip()
    return df

def load_source_file(path):
    df = load_tsv(path)
    for col in ('entity_id', 'business_name', 'business_address', 'country'):
        if col not in df.columns:
            df[col] = ''
    df = df[df['entity_id'] != '']
    n = len(df)
    df = df.drop_duplicates('entity_id', keep='first').reset_index(drop=True)
    if len(df) != n:
        log(f'  {os.path.basename(path)}: dropped {n - len(df)} duplicate entity_id rows')
    return df

class Source:

    def __init__(self, df, n_workers):
        self.ids = df['entity_id'].values
        self.country = df['country'].fillna('').str.strip().str.casefold().values
        self.cols, ptr, h = text_norm.normalize_frame(df, n_workers)
        self.meta = record_meta(self.cols, df['business_name'].values)
        self.meta['is_s3'] = (df['entity_id'].str[:2] == 'S3').values.astype(np.float32)
        self.name_hash = pd.util.hash_array(self.cols['name'].to_numpy(zero_copy_only=False)).astype(np.int64)
        self._ptr, self._h = (ptr, h)

    def __len__(self):
        return len(self.ids)

    def split_tokens(self, groups):
        out = {c: subset_csr(self._ptr, self._h, rows) for c, rows in groups.items()}
        self._ptr = self._h = None
        gc.collect()
        return out

    def view(self, rows):
        idx = pa.array(rows)
        d = {k: v.take(idx) for k, v in self.cols.items()}
        d.update(derive_cols(d))
        d.update({k: v[rows] for k, v in self.meta.items()})
        return d

def load_sources(data_dir, prefix, cfg):
    s1_df = load_source_file(os.path.join(data_dir, f'{prefix}_source1.tsv'))
    s23_df = pd.concat([load_source_file(os.path.join(data_dir, f'{prefix}_source2.tsv')), load_source_file(os.path.join(data_dir, f'{prefix}_source3.tsv'))], ignore_index=True)
    s23_df = s23_df.drop_duplicates('entity_id', keep='first').reset_index(drop=True)
    log(f'  S1={len(s1_df):,}  S2+S3={len(s23_df):,}  countries={sorted(pd.unique(s1_df.country))}')
    S1 = Source(s1_df, cfg['n_threads'])
    del s1_df
    S23 = Source(s23_df, cfg['n_threads'])
    del s23_df
    gc.collect()
    log('  normalised')
    return (S1, S23)

def country_groups(S1, S23):
    groups1, groups23 = ({}, {})
    for c in pd.unique(S1.country):
        groups1[c] = np.flatnonzero(S1.country == c)
        groups23[c] = np.flatnonzero(S23.country == c)
    return (groups1, groups23)

def _union_pairs(space, a_ptr, a_tok, rows, cfg):
    u = retrieve_union(space, a_ptr, a_tok, rows, cfg['top_k1'], cfg['n_threads'])
    r, k = np.nonzero(u[0] >= 0)
    pj = u[0][r, k].astype(np.int64)
    ov = pair_overlap(space, a_ptr, a_tok, rows[r], pj)
    _, _, X = rerank_features(u, ov)
    return (u, r, k, pj, X)

def rerank_sample(space, a_ptr, a_tok, rows, cfg):
    _, r, _, pj, X = _union_pairs(space, a_ptr, a_tok, rows, cfg)
    return (rows[r], pj, X)

def generate_candidates(space, a_ptr, a_tok, n_a, reranker, cfg):
    K2 = cfg['top_k']
    parts = []
    for s in range(0, n_a, cfg['batch_rows']):
        rows = np.arange(s, min(s + cfg['batch_rows'], n_a), dtype=np.int64)
        u, r, k, pj, X = _union_pairs(space, a_ptr, a_tok, rows, cfg)
        u_idx, u_c, u_cr = (u[0], u[1], u[2])
        p = reranker.predict(X, num_threads=cfg['n_threads']).astype(np.float32)
        del X
        P = np.full(u_idx.shape, -1.0, dtype=np.float32)
        P[r, k] = p
        top = np.argsort(-P, axis=1, kind='stable')[:, :K2]
        pv = np.take_along_axis(P, top, 1)
        rr, kk = np.nonzero(pv >= 0)
        ko = top[rr, kk]
        parts.append((rows[rr], u_idx[rr, ko].astype(np.int64), u_c[rr, ko], u_cr[rr, ko], pv[rr, kk], kk.astype(np.float32)))
        del u, P, top, pv
    if parts:
        pi, pj, blk, blk_rank, rr_p, rr_rank = (np.concatenate(x) for x in zip(*parts))
    else:
        pi = pj = np.zeros(0, np.int64)
        blk = blk_rank = rr_p = rr_rank = np.zeros(0, np.float32)
    ctx = {'blk_score': blk, 'blk_rank': blk_rank, 'rr_p': rr_p, 'rr_rank': rr_rank}
    row_max = np.zeros(n_a, np.float32)
    row_n = np.zeros(n_a, np.float32)
    np.maximum.at(row_max, pi, rr_p)
    np.add.at(row_n, pi, 1)
    ctx['rr_rel_row'] = rr_p / np.maximum(row_max[pi], 1e-06)
    ctx['row_n'] = row_n[pi]
    order = np.lexsort((-rr_p, pj))
    pj_s = pj[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pj_s)) + 1] if len(pj_s) else np.zeros(0, np.int64)
    grp_len = np.diff(np.r_[starts, len(pj_s)])
    grp_id = np.repeat(np.arange(len(starts)), grp_len)
    cid_rank = np.empty(len(pj), np.float32)
    cid_rank[order] = np.arange(len(pj_s)) - starts[grp_id]
    cid_max = rr_p[order][starts][grp_id]
    rel_cid = np.empty(len(pj), np.float32)
    rel_cid[order] = rr_p[order] / np.maximum(cid_max, 1e-06)
    cid_n = np.empty(len(pj), np.float32)
    cid_n[order] = grp_len[grp_id]
    ctx.update(rr_rel_cid=rel_cid, cid_rank=cid_rank, cid_n=cid_n)
    assert set(ctx) == set(CTX_COLS)
    return (pi, pj, ctx)

def features_to_disk(A, B, space, a_ptr, a_tok, pi, pj, ctx, cfg, path, table):
    F = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(pi), len(FEATURE_COLS)))
    for s in range(0, len(pi), cfg['feat_chunk']):
        e = min(s + cfg['feat_chunk'], len(pi))
        F[s:e] = compute_features(A, B, space, a_ptr, a_tok, pi[s:e], pj[s:e], {k: v[s:e] for k, v in ctx.items()}, table, cfg['n_threads'])
    F.flush()
    del F
    if cfg.get('country_norm'):
        country_normalise(path, cfg)
    return np.load(path, mmap_mode='r')

def country_normalise(path, cfg, n_ref=300000, seed=0):
    F = np.load(path, mmap_mode='r+')
    rng = np.random.default_rng(seed)
    ref = np.sort(np.asarray(F[np.sort(rng.choice(len(F), min(n_ref, len(F)), replace=False))], dtype=np.float32), axis=0)
    skip = [FEATURE_COLS.index(c) for c in cfg.get('norm_skip', [])]
    for s in range(0, len(F), cfg['feat_chunk']):
        X = np.asarray(F[s:s + cfg['feat_chunk']], dtype=np.float32)
        for j in range(X.shape[1]):
            if j in skip:
                continue
            lo = np.searchsorted(ref[:, j], X[:, j], 'left')
            hi = np.searchsorted(ref[:, j], X[:, j], 'right')
            X[:, j] = (lo + hi) / (2.0 * len(ref))
        F[s:s + cfg['feat_chunk']] = X
    F.flush()
    del F

class FeatureStore:

    def __init__(self, blocks):
        self.blocks = blocks
        self.offsets = np.cumsum([0] + [len(b) for b in blocks])

    def __len__(self):
        return int(self.offsets[-1])

    def rows(self, idx):
        out = np.empty((len(idx), len(FEATURE_COLS)), np.float32)
        bi = np.searchsorted(self.offsets, idx, side='right') - 1
        for b in np.unique(bi):
            m = bi == b
            out[m] = self.blocks[b][idx[m] - self.offsets[b]]
        return out

    def column(self, name):
        j = FEATURE_COLS.index(name)
        return np.concatenate([np.asarray(b[:, j], dtype=np.float32) for b in self.blocks])

    def chunks(self, size):
        for b, blk in enumerate(self.blocks):
            for s in range(0, len(blk), size):
                e = min(s + size, len(blk))
                yield (self.offsets[b] + s, self.offsets[b] + e, np.asarray(blk[s:e], dtype=np.float32))

def predict_stage1(store, models, fold, cfg):
    p = np.empty(len(store), np.float32)
    for s, e, X in store.chunks(cfg['feat_chunk']):
        if fold is None:
            p[s:e] = np.mean([m.predict(X, num_threads=cfg['n_threads']) for m in models], axis=0)
        else:
            f = fold[s:e]
            for k, m in enumerate(models):
                sel = f != k
                if sel.any():
                    p[s:e][sel] = m.predict(X[sel], num_threads=cfg['n_threads'])
    return p

def predict_stage2(store, S2, model, cfg):
    p = np.empty(len(store), np.float32)
    for s, e, X in store.chunks(cfg['feat_chunk']):
        p[s:e] = model.predict(np.hstack([X, S2[s:e]]), num_threads=cfg['n_threads'])
    return p

def one_to_one(s1_idx, s23_idx, p):
    order = np.lexsort((-p, s23_idx))
    first = np.r_[True, s23_idx[order][1:] != s23_idx[order][:-1]] if len(p) else np.zeros(0, bool)
    keep = np.zeros(len(p), bool)
    keep[order[first]] = True
    return keep

def macro_f05(n_s1, s1_idx, is_true, pred, n_true):
    tp = np.bincount(s1_idx[pred & is_true], minlength=n_s1)
    npred = np.bincount(s1_idx[pred], minlength=n_s1)
    f = np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_true + npred, 1e-09), 0.0)
    f = np.where((n_true == 0) & (npred == 0), 1.0, f)
    return float(f.mean())

def build_token_table(names_a, names_b, labels, cfg, min_count=5, keep_count=None, m=20.0):
    ext, mis = text_norm.token_stats(names_a, names_b, labels, cfg['n_threads'])
    keep_count = cfg.get('table_keep_count', 2000) if keep_count is None else keep_count

    def finish(d):
        n = sum((c for c, _ in d.values()))
        p = sum((q for _, q in d.values()))
        prior = p / max(n, 1)
        rare = [(c, q) for c, q in d.values() if c < min_count]
        prior_unseen = (sum((q for _, q in rare)) + m * prior) / (sum((c for c, _ in rare)) + m)
        tab = {t: round((q + m * prior) / (c + m), 5) for t, (c, q) in d.items() if c >= keep_count or (c >= min_count and zlib.crc32(t.encode()) % 2 == 0)}
        return (tab, float(prior_unseen))
    t_ext, p_ext = finish(ext)
    t_mis, p_mis = finish(mis)
    return {'extra': t_ext, 'missing': t_mis, 'prior_extra': p_ext, 'prior_missing': p_mis}

def stage2_inputs(store, GI, GJ, num1, num23, hash23):
    clean = store.column('nd_novel_bad') == 0 if 'nd_novel_bad' in FEATURE_COLS else None
    return dict(num_e=num1[GI], num_x=num23[GJ], hash_x=hash23[GJ], name_s=store.column('name_tset'), clean=clean)

def _lgb_params(cfg, lr):
    return dict(objective='binary', learning_rate=lr, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=cfg['n_threads'], seed=cfg['seed'], verbose=-1, metric=['binary_logloss'])

def _fit(X, y, XV, yv, names, cfg, lr, rounds):
    import lightgbm as lgb
    dtr = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(XV, label=yv, reference=dtr)
    return lgb.train(_lgb_params(cfg, lr), dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=['val'], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])

def train(train_dir, cfg, model_dir, cache_dir):
    import lightgbm as lgb
    rng = np.random.default_rng(cfg['seed'])
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log('Loading train data')
    S1, S23 = load_sources(train_dir, 'train', cfg)
    gt = load_tsv(os.path.join(train_dir, 'train_ground_truth.tsv'))
    g = gt[gt['matched_entity_ids'] != ''].copy()
    g['m'] = g['matched_entity_ids'].str.split(',')
    g = g.explode('m')
    gi = pd.Index(S1.ids).get_indexer(g['source1_entity_id'])
    gj = pd.Index(S23.ids).get_indexer(g['m'])
    ok = (gi >= 0) & (gj >= 0)
    N23 = np.int64(len(S23))
    gt_keys = np.unique(gi[ok].astype(np.int64) * N23 + gj[ok])
    n_true_all = np.bincount(gi[ok], minlength=len(S1))
    del g, gt, gi, gj
    gc.collect()
    u = rng.random(len(S1))
    split = np.where(u < cfg['train_frac'], 0, np.where(u < cfg['train_frac'] + cfg['val_frac'], 1, 2))
    if cfg.get('val_country'):
        vc = S1.country == cfg['val_country'].casefold()
        split = np.where(vc, np.where(u < 0.5, 1, 3), split)
        split = np.where(~vc & (split == 1), 2, split)
    fold = (rng.random(len(S1)) < 0.5).astype(np.int8)
    groups1, groups23 = country_groups(S1, S23)
    tok1, tok23 = (S1.split_tokens(groups1), S23.split_tokens(groups23))
    countries = [c for c in groups1 if len(groups23[c])]
    rr_X, rr_y, tk_a, tk_b, tk_y = ([], [], [], [], [])
    for c in countries:
        t = time.time()
        bptr, bh = tok23.pop(c)
        space = TokenSpace(bptr, bh, cfg['key_cap'])
        del bh
        aptr, ah = tok1.pop(c)
        a_ptr, a_tok = space.map_a(aptr, ah)
        del ah
        log(f'[{c}] index built in {time.time() - t:.0f}s (S1={len(groups1[c]):,}, S2+S3={len(groups23[c]):,})')
        pool_rows = np.flatnonzero(split[groups1[c]] == 2)
        if len(pool_rows) == 0 and (not cfg.get('val_country')):
            pool_rows = np.arange(len(groups1[c]))
        if len(pool_rows):
            samp = np.sort(rng.choice(pool_rows, min(cfg['rr_sample'], len(pool_rows)), replace=False))
            pi, pj, X = rerank_sample(space, a_ptr, a_tok, samp, cfg)
            yy = np.isin(groups1[c][pi].astype(np.int64) * N23 + groups23[c][pj], gt_keys)
            rr_X.append(X)
            rr_y.append(yy)
            ts = np.sort(rng.choice(len(pi), min(3000000, len(pi)), replace=False))
            tk_a.extend(S1.cols['name'].take(pa.array(groups1[c][pi[ts]])).to_pylist())
            tk_b.extend(S23.cols['name'].take(pa.array(groups23[c][pj[ts]])).to_pylist())
            tk_y.append(yy[ts])
            del pi, pj, X
        d = os.path.join(cache_dir, f'train_{c}')
        space.save(d)
        np.save(os.path.join(d, 'a_ptr.npy'), a_ptr)
        np.save(os.path.join(d, 'a_tok.npy'), a_tok)
        del space, a_ptr, a_tok
        gc.collect()
    log('Training re-ranker')
    rr_X = np.concatenate(rr_X)
    rr_y = np.concatenate(rr_y)
    reranker = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, num_threads=cfg['n_threads'], seed=cfg['seed'], verbose=-1), lgb.Dataset(rr_X, rr_y, feature_name=RERANK_COLS), 200)
    reranker.save_model(os.path.join(model_dir, 'reranker.txt'))
    log(f'  re-ranker trained on {len(rr_y):,} pairs ({rr_y.mean():.3%} positive)')
    table = build_token_table(tk_a, tk_b, np.concatenate(tk_y).tolist(), cfg)
    json.dump(table, open(os.path.join(model_dir, 'name_token_table.json'), 'w'))
    log(f"  word table: {len(table['extra']):,} extra-word rates, {len(table['missing']):,} missing-word rates, unseen extra-word rate {table['prior_extra']:.3f}")
    del rr_X, rr_y, tk_a, tk_b, tk_y
    gc.collect()
    blocks, GI, GJ, LAB = ([], [], [], [])
    recall = {}
    for c in countries:
        d = os.path.join(cache_dir, f'train_{c}')
        space = TokenSpace.load(d)
        a_ptr, a_tok = (np.load(os.path.join(d, 'a_ptr.npy')), np.load(os.path.join(d, 'a_tok.npy')))
        a_rows, b_rows = (groups1[c], groups23[c])
        pi, pj, ctx = generate_candidates(space, a_ptr, a_tok, len(a_rows), reranker, cfg)
        gpi, gpj = (a_rows[pi], b_rows[pj])
        lab = np.isin(gpi.astype(np.int64) * N23 + gpj, gt_keys)
        recall[c] = float(lab.sum() / max(n_true_all[a_rows].sum(), 1))
        log(f"[{c}] candidates={len(pi):,}  blocking recall@{cfg['top_k']}={recall[c]:.4f}")
        A, B = (S1.view(a_rows), S23.view(b_rows))
        blocks.append(features_to_disk(A, B, space, a_ptr, a_tok, pi, pj, ctx, cfg, os.path.join(cache_dir, f'F_train_{c}.npy'), table))
        GI.append(gpi.astype(np.int32))
        GJ.append(gpj.astype(np.int32))
        LAB.append(lab)
        log(f'[{c}] features written')
        del space, a_ptr, a_tok, pi, pj, ctx, A, B
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)
    num1, num23, hash23 = (S1.meta['first_num'], S23.meta['first_num'], S23.name_hash)
    del S1, S23
    gc.collect()
    store = FeatureStore(blocks)
    GI = np.concatenate(GI)
    GJ = np.concatenate(GJ)
    Y = np.concatenate(LAB)
    del LAB
    sp = split[GI]
    fo = fold[GI]
    s1_models = []
    for k in (0, 1):
        tr = np.flatnonzero((sp == 0) & (fo == k))
        va = np.flatnonzero((sp == 1) & (fo == k))
        log(f'Stage-1 model {k}: {len(tr):,} train pairs')
        m = _fit(store.rows(tr), Y[tr], store.rows(va), Y[va], FEATURE_COLS, cfg, 0.1, 1500)
        m.save_model(os.path.join(model_dir, f'stage1_fold{k}.txt'), num_iteration=m.best_iteration)
        s1_models.append(lgb.Booster(model_file=os.path.join(model_dir, f'stage1_fold{k}.txt')))
        log(f'  best iteration {m.best_iteration}')
        del m
        gc.collect()
    log('Stage-1 out-of-fold scoring of all pairs')
    p1 = predict_stage1(store, s1_models, fo, cfg)
    tr = np.flatnonzero(sp == 0)
    va = np.flatnonzero(sp == 1)
    S2 = stage2_features(GI, GJ, p1, **stage2_inputs(store, GI, GJ, num1, num23, hash23))
    S2_tr, S2_va = (S2[tr], S2[va])
    del S2
    gc.collect()
    log(f'Stage-2 model: {len(tr):,} train pairs ({Y[tr].mean():.3%} positive)')
    names2 = FEATURE_COLS + STAGE2_COLS
    XV2 = np.hstack([store.rows(va), S2_va])
    m2 = _fit(np.hstack([store.rows(tr), S2_tr]), Y[tr], XV2, Y[va], names2, cfg, 0.05, 3000)
    del S2_tr
    gc.collect()
    m2.save_model(os.path.join(model_dir, 'stage2.txt'), num_iteration=m2.best_iteration)
    log(f'  best iteration {m2.best_iteration}')
    val_s1 = np.flatnonzero(split == 1)
    pv = m2.predict(XV2, num_iteration=m2.best_iteration, num_threads=cfg['n_threads'])
    loc = np.searchsorted(val_s1, GI[va])
    yv = Y[va]
    n_true_val = n_true_all[val_s1]
    keep = one_to_one(loc, GJ[va], pv)
    sweep = {}
    for t in np.round(np.arange(0.2, 0.96, 0.025), 3):
        sweep[float(t)] = round(macro_f05(len(val_s1), loc, yv, keep & (pv >= t), n_true_val), 5)
    thr = max(sweep, key=sweep.get)
    keep1 = one_to_one(loc, GJ[va], p1[va])
    f_stage1 = max((macro_f05(len(val_s1), loc, yv, keep1 & (p1[va] >= t), n_true_val) for t in np.arange(0.3, 0.95, 0.025)))
    upper = macro_f05(len(val_s1), loc, yv, yv.astype(bool), n_true_val)
    if cfg.get('dump_val'):
        np.savez(os.path.join(model_dir, 'val_dump.npz'), gi=GI[va], gj=GJ[va], p2=pv, p1=p1[va], y=yv, val_s1=val_s1, n_true=n_true_val, X=store.rows(va))
    log(f'  validation macro F0.5: stage-1 {f_stage1:.4f}  stage-2 {sweep[thr]:.4f} at threshold {thr}  (blocking ceiling {upper:.4f})')
    imp = dict(zip(names2, m2.feature_importance('gain').round(1).tolist()))
    summary = dict(threshold=thr, val_macro_f05=sweep[thr], val_stage1_f05=f_stage1, val_ceiling=upper, blocking_recall=recall, sweep=sweep, config=cfg, feature_gain=imp)
    json.dump(summary, open(os.path.join(model_dir, 'train_summary.json'), 'w'), indent=1)
    del store, blocks
    gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)
    return load_models(model_dir)

def load_models(model_dir):
    import lightgbm as lgb
    B = lambda f: lgb.Booster(model_file=os.path.join(model_dir, f))
    thr = json.load(open(os.path.join(model_dir, 'train_summary.json')))['threshold']
    table = json.load(open(os.path.join(model_dir, 'name_token_table.json')))
    countries = list(json.load(open(os.path.join(model_dir, 'train_summary.json'))).get('blocking_recall', {}))
    return dict(countries=countries, reranker=B('reranker.txt'), stage1=[B('stage1_fold0.txt'), B('stage1_fold1.txt')], stage2=B('stage2.txt'), threshold=thr, table=table)

def predict_test(test_dir, models, cfg, out_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    log('Loading test data')
    S1, S23 = load_sources(test_dir, 'test', cfg)
    groups1, groups23 = country_groups(S1, S23)
    tok1, tok23 = (S1.split_tokens(groups1), S23.split_tokens(groups23))
    blocks, GI, GJ = ([], [], [])
    for c in groups1:
        a_rows, b_rows = (groups1[c], groups23[c])
        bptr, bh = tok23.pop(c)
        aptr, ah = tok1.pop(c)
        if len(b_rows) == 0:
            log(f'[{c}] no S2/S3 records -> all singletons')
            continue
        space = TokenSpace(bptr, bh, cfg['key_cap'])
        del bh
        a_ptr, a_tok = space.map_a(aptr, ah)
        del ah
        log(f'[{c}] index built (S1={len(a_rows):,}, S2+S3={len(b_rows):,})')
        pi, pj, ctx = generate_candidates(space, a_ptr, a_tok, len(a_rows), models['reranker'], cfg)
        log(f'[{c}] {len(pi):,} candidate pairs, computing features')
        A, B = (S1.view(a_rows), S23.view(b_rows))
        blocks.append(features_to_disk(A, B, space, a_ptr, a_tok, pi, pj, ctx, cfg, os.path.join(cache_dir, f'F_test_{c}.npy'), models['table']))
        GI.append(a_rows[pi].astype(np.int32))
        GJ.append(b_rows[pj].astype(np.int32))
        del space, a_ptr, a_tok, pi, pj, ctx, A, B
        gc.collect()
    store = FeatureStore(blocks)
    GI = np.concatenate(GI)
    GJ = np.concatenate(GJ)
    log('Stage-1 scoring')
    p1 = predict_stage1(store, models['stage1'], None, cfg)
    log('Stage-2 scoring')
    S2 = stage2_features(GI, GJ, p1, **stage2_inputs(store, GI, GJ, S1.meta['first_num'], S23.meta['first_num'], S23.name_hash))
    p2 = predict_stage2(store, S2, models['stage2'], cfg)
    del S2
    match = one_to_one(GI, GJ, p2) & (p2 >= models['threshold'])
    if cfg.get('decoy_veto', True):
        match &= ~decoy_veto(S1, S23, GI, GJ, match, models)
    log(f'Candidates {len(GI):,}  matches {match.sum():,}  S1 with >=1 match {len(np.unique(GI[match])):,}/{len(S1):,}')
    write_outputs(out_dir, S1.ids, S23.ids, GI, GJ, match)
    del store, blocks
    gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)

def decoy_veto(S1, S23, GI, GJ, match, models):
    seen = set(models.get('countries') or [])
    unseen = np.isin(S1.country[GI], [c for c in pd.unique(S1.country) if c not in seen])
    idx = np.flatnonzero(match & unseen)
    veto = np.zeros(len(GI), bool)
    if len(idx):
        na = S1.cols['name'].take(pa.array(GI[idx])).to_pylist()
        nb = S23.cols['name'].take(pa.array(GJ[idx])).to_pylist()
        veto[idx] = veto_mask(na, nb, S1.meta['first_num'][GI[idx]], S23.meta['first_num'][GJ[idx]], models['table'])
        log(f'  decoy veto (countries unseen in training: {sorted(set(S1.country[GI[idx]]))}): {veto.sum():,} of {len(idx):,} predicted pairs rejected')
    return veto

def write_outputs(out_dir, s1_ids, s23_ids, gi, gj, match):
    os.makedirs(out_dir, exist_ok=True)
    key = gi.astype(np.int64) * np.int64(len(s23_ids)) + gj
    uk, inv = np.unique(key, return_inverse=True)
    if len(uk) != len(key):
        m = np.zeros(len(uk), bool)
        np.logical_or.at(m, inv, match)
        first = np.unique(inv, return_index=True)[1]
        gi, gj, match = (gi[first], gj[first], m)
    order = np.argsort(gi, kind='stable')
    gi_s, gj_s, m_s = (gi[order], gj[order], match[order])
    bounds = np.searchsorted(gi_s, np.arange(len(s1_ids) + 1))
    specs = [('matching_results.tsv', 'matched_entity_ids', True), ('candidate_pairs.tsv', 'candidate_entity_ids', False)]
    for fname, header, only_match in specs:
        path = os.path.join(out_dir, fname)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'source1_entity_id\t{header}\n')
            for i in range(len(s1_ids)):
                a, b = (bounds[i], bounds[i + 1])
                js = gj_s[a:b]
                if only_match:
                    js = js[m_s[a:b]]
                f.write(f"{s1_ids[i]}\t{','.join(s23_ids[js])}\n")
        log(f'  wrote {path}')
    self_check(out_dir, s1_ids)

def self_check(out_dir, s1_ids):
    want = set(s1_ids)
    for fname in ('matching_results.tsv', 'candidate_pairs.tsv'):
        seen, bad = (set(), 0)
        with open(os.path.join(out_dir, fname), encoding='utf-8') as f:
            next(f)
            for line in f:
                sid, ids = line.rstrip('\n').split('\t')
                lst = [x for x in ids.split(',') if x]
                if sid in seen or len(set(lst)) != len(lst) or any((x[:3] not in ('S2-', 'S3-') for x in lst)):
                    bad += 1
                seen.add(sid)
        status = 'OK' if seen == want and bad == 0 else f'PROBLEM ({bad} bad rows, {len(want - seen)} missing)'
        log(f'  self-check {fname}: {len(seen):,} rows -> {status}')

def run_all(base_dir, out_dir, model_dir, stage='all', **overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    features.configure(novelty=cfg['novelty'], clean_support=cfg['clean_support'])
    log(f'config: {cfg}  stage={stage}')
    cache_dir = os.path.join(model_dir, '..', '.cache')
    if stage in ('all', 'train'):
        models = train(os.path.join(base_dir, 'dataset', 'train'), cfg, model_dir, cache_dir)
        gc.collect()
    else:
        models = load_models(model_dir)
    if stage in ('all', 'test'):
        predict_test(os.path.join(base_dir, 'dataset', 'test'), models, cfg, out_dir, cache_dir)
    text_norm.close_pool()
    log('finished')
