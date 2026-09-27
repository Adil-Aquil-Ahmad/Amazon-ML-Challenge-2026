import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
import text_norm
from blocking import pair_overlap
FEATURE_COLS = ['name_ratio', 'name_tset', 'name_tsort', 'name_partial', 'core_ratio', 'core_tset', 'compact_ratio', 'compact_jw', 'compact_partial', 'skel_ratio', 'skel_tset', 'addr_ratio', 'addr_tset', 'addr_tsort', 'addr_partial', 'nums_tset', 'nums_equal', 'first_num_eq', 'num_absdiff_log', 'num_trunc', 'num_lev', 'nd_extra_n', 'nd_extra_min', 'nd_extra_mean', 'nd_miss_n', 'nd_miss_min', 'nd_extra_decoy', 'nd_novel_n', 'nd_novelty', 'nd_novel_bad', 'name_share_a', 'name_share_b', 'name_shared', 'addr_share_a', 'addr_share_b', 'addr_shared', 'blk_score', 'blk_rank', 'rr_p', 'rr_rank', 'rr_rel_row', 'row_n', 'rr_rel_cid', 'cid_rank', 'cid_n', 'src_s3', 'a_addr_empty', 'b_addr_empty', 'b_name_indic', 'b_name_domain', 'a_name_ntok', 'b_name_ntok', 'a_addr_ntok', 'b_addr_ntok']
_NOVELTY_COLS = ['nd_novel_n', 'nd_novelty', 'nd_novel_bad']
_ALL_FEATURE_COLS = list(FEATURE_COLS)
OPTIONS = {'novelty': False, 'clean_support': False}

def configure(novelty=False, clean_support=False):
    OPTIONS.update(novelty=novelty, clean_support=clean_support)
    FEATURE_COLS[:] = [c for c in _ALL_FEATURE_COLS if novelty or c not in _NOVELTY_COLS]
configure()
CTX_COLS = ['blk_score', 'blk_rank', 'rr_p', 'rr_rank', 'rr_rel_row', 'row_n', 'rr_rel_cid', 'cid_rank', 'cid_n']

def derive_cols(cols):
    nums = pc.replace_substring_regex(cols['addr'], '\\S*[a-z]\\S*', '')
    nums = pc.utf8_trim_whitespace(pc.replace_substring_regex(nums, '\\s+', ' '))
    return {'compact': pc.replace_substring(cols['core'], ' ', ''), 'nums': nums}

def record_meta(cols, raw_names):
    nums = derive_cols(cols)['nums'].to_pandas()
    first = pd.to_numeric(nums.str.split(' ', n=1).str[0], errors='coerce').fillna(-1).astype(np.int64).values
    del nums
    raw = pd.Series(raw_names)
    addr_len = pc.utf8_length(cols['addr']).to_numpy(zero_copy_only=False)
    return {'first_num': first, 'addr_empty': (addr_len == 0).astype(np.float32), 'name_indic': raw.str.contains('[ऀ-ൿ]', regex=True).values.astype(np.float32), 'name_domain': raw.str.contains('(?:\\.(?:com|net|org|in|co|biz|fr)\\s*$)|^\\s*@|^www\\.', regex=True, case=False).values.astype(np.float32), 'name_ntok': (pc.count_substring(cols['core'], ' ').to_numpy(zero_copy_only=False) + 1).astype(np.float32), 'addr_ntok': ((pc.count_substring(cols['addr'], ' ').to_numpy(zero_copy_only=False) + 1) * (addr_len > 0)).astype(np.float32)}

def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0

def _first_num_str(nums):
    return [x.split(' ', 1)[0] if x else '' for x in nums]

def compute_features(A, B, space, a_ptr, a_tok, pi, pj, ctx, table, n_workers):
    n = len(pi)
    F = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)
    c = 0
    ipi, ipj = (pa.array(pi), pa.array(pj))

    def g(k):
        return (A[k].take(ipi).to_pylist(), B[k].take(ipj).to_pylist())
    a, b = g('name')
    for sc in (fuzz.ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio):
        F[:, c] = _cp(a, b, sc)
        c += 1
    nd = text_norm.name_diff_features(a, b, table, n_workers)
    a, b = g('core')
    F[:, c] = _cp(a, b, fuzz.ratio)
    c += 1
    F[:, c] = _cp(a, b, fuzz.token_set_ratio)
    c += 1
    a, b = g('compact')
    F[:, c] = _cp(a, b, fuzz.ratio)
    c += 1
    F[:, c] = process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    c += 1
    F[:, c] = _cp(a, b, fuzz.partial_ratio)
    c += 1
    a, b = g('skel')
    F[:, c] = _cp(a, b, fuzz.ratio)
    c += 1
    F[:, c] = _cp(a, b, fuzz.token_set_ratio)
    c += 1
    a, b = g('addr')
    for sc in (fuzz.ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio):
        F[:, c] = _cp(a, b, sc)
        c += 1
    a, b = g('nums')
    F[:, c] = _cp(a, b, fuzz.token_set_ratio)
    c += 1
    F[:, c] = np.fromiter((x == y and x != '' for x, y in zip(a, b)), dtype=np.float32, count=n)
    c += 1
    fs_a, fs_b = (_first_num_str(a), _first_num_str(b))
    del a, b
    fa, fb = (A['first_num'][pi], B['first_num'][pj])
    missing = (fa < 0) | (fb < 0)
    F[:, c] = np.where(missing, 0.5, (fa == fb).astype(np.float32))
    c += 1
    F[:, c] = np.where(missing, -1.0, np.log1p(np.abs(fa - fb).astype(np.float64)))
    c += 1
    F[:, c] = np.fromiter((x != y and x != '' and (y != '') and (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x)) for x, y in zip(fs_a, fs_b)), dtype=np.float32, count=n)
    c += 1
    F[:, c] = np.where(missing, -1.0, process.cpdist(fs_a, fs_b, scorer=Levenshtein.distance, workers=-1, dtype=np.int32))
    c += 1
    del fs_a, fs_b
    k = 9 if OPTIONS['novelty'] else 6
    F[:, c:c + k] = nd[:, :k]
    c += k
    ov = pair_overlap(space, a_ptr, a_tok, pi, pj)
    eps = 1e-06
    F[:, c] = ov[:, 0] / (ov[:, 1] + eps)
    c += 1
    F[:, c] = ov[:, 0] / (ov[:, 2] + eps)
    c += 1
    F[:, c] = ov[:, 0]
    c += 1
    F[:, c] = ov[:, 3] / (ov[:, 4] + eps)
    c += 1
    F[:, c] = ov[:, 3] / (ov[:, 5] + eps)
    c += 1
    F[:, c] = ov[:, 3]
    c += 1
    for k in CTX_COLS:
        F[:, c] = ctx[k]
        c += 1
    F[:, c] = B['is_s3'][pj]
    c += 1
    F[:, c] = A['addr_empty'][pi]
    c += 1
    F[:, c] = B['addr_empty'][pj]
    c += 1
    F[:, c] = B['name_indic'][pj]
    c += 1
    F[:, c] = B['name_domain'][pj]
    c += 1
    F[:, c] = A['name_ntok'][pi]
    c += 1
    F[:, c] = B['name_ntok'][pj]
    c += 1
    F[:, c] = A['addr_ntok'][pi]
    c += 1
    F[:, c] = B['addr_ntok'][pj]
    c += 1
    assert c == len(FEATURE_COLS)
    return F
CLEAN_SUPPORT_OFF = False
STAGE2_COLS = ['p1', 'p1_row_max', 'p1_rel_row', 'p1_row_rank', 'p1_row_n50', 'p1_row_sum', 'p1_cid_other', 'p1_cid_margin', 'p1_cid_rank', 'p1_cid_n30', 'p1_cid_sum', 'sup_num_x', 'sup_num_e', 'sup_name_x', 'name_cid_other', 'name_cid_margin', 'name_cid_rank']

def _group_stats(key, p):
    order = np.lexsort((-p, key))
    ks, ps = (key[order], p[order])
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1] if len(ks) else np.zeros(0, np.int64)
    ln = np.diff(np.r_[starts, len(ks)])
    gid = np.repeat(np.arange(len(starts)), ln)
    rank = np.arange(len(ks)) - starts[gid]
    gmax = ps[starts][gid]
    second = np.where(ln > 1, ps[np.minimum(starts + 1, len(ps) - 1)], 0.0)[gid]
    other = np.where(rank == 0, second, gmax)
    csum = np.add.reduceat(ps, starts) if len(ps) else np.zeros(0)
    n50 = np.add.reduceat((ps > 0.5).astype(np.float32), starts) if len(ps) else np.zeros(0)
    n30 = np.add.reduceat((ps > 0.3).astype(np.float32), starts) if len(ps) else np.zeros(0)
    out = {}
    for name, arr in (('rank', rank), ('max', gmax), ('other', other), ('sum', csum[gid]), ('n50', n50[gid]), ('n30', n30[gid])):
        o = np.empty(len(p), np.float32)
        o[order] = arr
        out[name] = o
    return out

def _support(gi, key, p1, valid):
    k = np.where(valid, key, -1).astype(np.int64)
    order = np.lexsort((k, gi))
    gs, ks, ps = (gi[order], k[order], p1[order].astype(np.float64))
    brk = np.r_[True, (gs[1:] != gs[:-1]) | (ks[1:] != ks[:-1])] if len(gs) else np.zeros(0, bool)
    starts = np.flatnonzero(brk)
    tot = np.add.reduceat(ps, starts) if len(ps) else np.zeros(0)
    gid = np.cumsum(brk) - 1
    out = np.empty(len(p1), np.float32)
    out[order] = tot[gid] - ps
    out[~valid] = -1.0
    return out

def stage2_features(gi, gj, p1, num_e, num_x, hash_x, name_s, clean):
    r = _group_stats(gi, p1)
    c = _group_stats(gj, p1)
    nm = _group_stats(gj, name_s)
    if CLEAN_SUPPORT_OFF or not OPTIONS['clean_support'] or clean is None:
        clean = np.ones(len(p1), bool)
    pc = np.where(clean, p1, 0.0).astype(np.float32)
    sup_x = _support(gi, num_x, pc, num_x >= 0) + np.where(clean | (num_x < 0), 0.0, p1)
    same_e = (num_x == num_e) & (num_e >= 0)
    tot_e = np.zeros(int(gi.max()) + 1 if len(gi) else 1, np.float64)
    np.add.at(tot_e, gi, np.where(same_e, pc, 0.0))
    sup_e = np.where(num_e >= 0, tot_e[gi] - np.where(same_e, pc, 0.0), -1.0).astype(np.float32)
    sup_name = _support(gi, hash_x.astype(np.int64), p1, np.ones(len(p1), bool))
    S = np.column_stack([p1, r['max'], p1 / np.maximum(r['max'], 1e-06), r['rank'], r['n50'], r['sum'], c['other'], p1 - c['other'], c['rank'], c['n30'], c['sum'], sup_x, sup_e, sup_name, nm['other'], name_s - nm['other'], nm['rank']]).astype(np.float32)
    return S
