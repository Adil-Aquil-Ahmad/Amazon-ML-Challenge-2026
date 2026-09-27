import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
import text_norm
MAX_SHIFT = 30

def first_number(norm_addr_tokens):
    for t in norm_addr_tokens:
        if t.isdigit():
            return int(t)
    return -1

def veto_mask(names_a, names_b, num_a, num_b, table, max_shift=MAX_SHIFT):
    from rapidfuzz.fuzz import ratio
    t_ext, prior = (table['extra'], table['prior_extra'])
    num_a = np.asarray(num_a)
    num_b = np.asarray(num_b)
    shift = np.abs(num_a - num_b)
    small = (num_a >= 0) & (num_b >= 0) & (shift > 0) & (shift <= max_shift)
    out = np.zeros(len(names_a), bool)
    for i in np.flatnonzero(small):
        a, b = (names_a[i], names_b[i])
        extra, _ = text_norm.diff_tokens(a, b)
        ta = a.split()
        for t in extra:
            if len(t) >= 3 and (not t.isdigit()) and (t_ext.get(t, prior) < 0.2) and (max((ratio(t, u) for u in ta), default=0.0) < 50):
                out[i] = True
                break
    return out

def trained_countries(models_dir):
    s = json.load(open(os.path.join(models_dir, 'train_summary.json')))
    return {c.casefold() for c in s.get('blocking_recall', {})}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--matching', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--countries', default=None, help='comma-separated countries to filter (default: countries not seen in training)')
    a = ap.parse_args()
    rd = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, quoting=3)
    s1 = rd(os.path.join(a.test_dir, 'test_source1.tsv')).drop_duplicates('entity_id').set_index('entity_id')
    s23 = pd.concat([rd(os.path.join(a.test_dir, 'test_source2.tsv')), rd(os.path.join(a.test_dir, 'test_source3.tsv'))]).drop_duplicates('entity_id').set_index('entity_id')
    table = json.load(open(os.path.join(a.models, 'name_token_table.json')))
    ckey = s1['country'].str.strip().str.casefold()
    if a.countries:
        target = {c.strip().casefold() for c in a.countries.split(',')}
    else:
        target = set(ckey.unique()) - trained_countries(a.models)
    print('filtering countries:', sorted(target))
    m = rd(a.matching)
    rows = m[(m['matched_entity_ids'] != '') & ckey.reindex(m['source1_entity_id']).isin(target).values]
    pairs = rows.assign(x=rows['matched_entity_ids'].str.split(',')).explode('x')[['source1_entity_id', 'x']]
    e = pairs['source1_entity_id'].values
    x = pairs['x'].values
    na = [text_norm.normalize_name(n)[0] for n in s1['business_name'].reindex(e).values]
    nb = [text_norm.normalize_name(n)[0] for n in s23['business_name'].reindex(x).values]
    ua = [first_number(text_norm.normalize_address(v)) for v in s1['business_address'].reindex(e).values]
    ub = [first_number(text_norm.normalize_address(v)) for v in s23['business_address'].reindex(x).values]
    veto = veto_mask(na, nb, ua, ub, table)
    print(f'pairs checked {len(pairs):,}   vetoed {veto.sum():,} ({veto.mean():.2%})')
    keep = pairs[~veto].groupby('source1_entity_id')['x'].agg(','.join)
    new = m.copy()
    sel = new['source1_entity_id'].isin(rows['source1_entity_id'])
    new.loc[sel, 'matched_entity_ids'] = new.loc[sel, 'source1_entity_id'].map(keep).fillna('')
    backup = a.matching.replace('.tsv', '.before_veto.tsv')
    if not os.path.exists(backup):
        shutil.copy(a.matching, backup)
    new.to_csv(a.matching, sep='\t', index=False, lineterminator='\n')
    print(f'wrote {a.matching} (original kept as {backup})')
if __name__ == '__main__':
    main()
