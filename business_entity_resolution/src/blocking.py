import numpy as np
import numba as nb
from numba import njit, prange

def subset_csr(indptr, data, rows):
    starts = indptr[rows]
    lens = indptr[rows + 1] - starts
    new_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lens, out=new_ptr[1:])
    total = int(new_ptr[-1])
    if total == 0:
        return (new_ptr, data[:0].copy())
    offs = np.repeat(starts - new_ptr[:-1], lens)
    offs += np.arange(total, dtype=np.int64)
    return (new_ptr, data[offs])

@njit(parallel=True, cache=True)
def _sort_within_rows(ptr, tok):
    for r in prange(len(ptr) - 1):
        s, e = (ptr[r], ptr[r + 1])
        if e - s > 1:
            tok[s:e] = np.sort(tok[s:e])

@njit(cache=True)
def _fill_postings(ptr, tok, df, cap, post_ptr, post_row):
    cur = post_ptr[:-1].copy()
    for r in range(len(ptr) - 1):
        for p in range(ptr[r], ptr[r + 1]):
            t = tok[p]
            if df[t] <= cap:
                post_row[cur[t]] = r
                cur[t] += 1

class TokenSpace:
    FIELDS = ('vocab', 'df', 'idf', 'grp', 'b_indptr', 'b_tok', 'post_ptr', 'post_row')

    def __init__(self, b_indptr=None, b_hash=None, cap=2000):
        self.cap = cap
        if b_indptr is None:
            return
        self.n_b = len(b_indptr) - 1
        order = np.argsort(b_hash)
        hs = b_hash[order]
        first = np.empty(len(hs), dtype=bool)
        if len(hs):
            first[0] = True
            np.not_equal(hs[1:], hs[:-1], out=first[1:])
        self.vocab = hs[first]
        del hs
        ids = np.cumsum(first, dtype=np.int64).astype(np.int32) - 1
        del first
        tok = np.empty(len(ids), dtype=np.int32)
        tok[order] = ids
        del order, ids
        self.df = np.bincount(tok, minlength=len(self.vocab)).astype(np.int32)
        self._finish(b_indptr, tok)

    def _finish(self, b_indptr, tok):
        self.n_b = len(b_indptr) - 1
        self.idf = np.log((self.n_b + 1.0) / (self.df + 0.5)).astype(np.float32)
        self.max_idf = np.float32(np.log(self.n_b + 1.0) + 0.7)
        self.grp = (self.vocab & 1).astype(np.uint8)
        _sort_within_rows(b_indptr, tok)
        self.b_indptr, self.b_tok = (b_indptr, tok)
        counts = np.where(self.df <= self.cap, self.df, 0).astype(np.int64)
        self.post_ptr = np.zeros(len(self.vocab) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.post_ptr[1:])
        self.post_row = np.empty(int(self.post_ptr[-1]), dtype=np.int32)
        _fill_postings(b_indptr, tok, self.df, self.cap, self.post_ptr, self.post_row)

    def save(self, d):
        import os
        os.makedirs(d, exist_ok=True)
        for f in self.FIELDS:
            np.save(os.path.join(d, f + '.npy'), getattr(self, f))
        np.save(os.path.join(d, 'meta.npy'), np.array([self.cap, self.n_b], dtype=np.int64))

    @classmethod
    def load(cls, d):
        import os
        cap, n_b = np.load(os.path.join(d, 'meta.npy'))
        sp = cls(cap=int(cap))
        for f in cls.FIELDS:
            setattr(sp, f, np.load(os.path.join(d, f + '.npy')))
        sp.n_b = int(n_b)
        sp.max_idf = np.float32(np.log(sp.n_b + 1.0) + 0.7)
        return sp

    def map_a(self, a_indptr, a_hash):
        order = np.argsort(a_hash)
        hs = a_hash[order]
        pos_s = np.searchsorted(self.vocab, hs)
        np.minimum(pos_s, max(len(self.vocab) - 1, 0), out=pos_s)
        tok_s = np.where(self.vocab[pos_s] == hs, pos_s, -1 - (hs & 1)).astype(np.int32)
        del pos_s, hs
        tok = np.empty(len(tok_s), dtype=np.int32)
        tok[order] = tok_s
        del order, tok_s
        _sort_within_rows(a_indptr, tok)
        return (a_indptr, tok)

    def weights(self, tok):
        w = np.full(len(tok), self.max_idf, dtype=np.float32)
        m = tok >= 0
        w[m] = self.idf[tok[m]]
        return w

@njit(parallel=True, cache=True)
def _topk_kernel(rows, a_ptr, a_tok, a_w, post_ptr, post_row, n_b, K, n_tasks, out_idx, out_sc):
    n_a = len(rows)
    for t in prange(n_tasks):
        acc = np.zeros(n_b, np.float32)
        touched = np.empty(n_b, np.int32)
        for rr in range(t, n_a, n_tasks):
            r = rows[rr]
            nt = 0
            for p in range(a_ptr[r], a_ptr[r + 1]):
                tk = a_tok[p]
                w = a_w[p]
                if tk < 0 or w <= 0.0:
                    continue
                for q in range(post_ptr[tk], post_ptr[tk + 1]):
                    j = post_row[q]
                    if acc[j] == 0.0:
                        touched[nt] = j
                        nt += 1
                    acc[j] += w
            if nt > 0:
                sc = np.empty(nt, np.float32)
                for i in range(nt):
                    sc[i] = acc[touched[i]]
                order = np.argsort(-sc)
                m = min(K, nt)
                for i in range(m):
                    out_idx[rr, i] = touched[order[i]]
                    out_sc[rr, i] = sc[order[i]]
                for i in range(nt):
                    acc[touched[i]] = 0.0

def retrieve(space, a_ptr, a_tok, a_w, rows, K, n_threads):
    out_idx = np.full((len(rows), K), -1, dtype=np.int32)
    out_sc = np.zeros((len(rows), K), dtype=np.float32)
    nb.set_num_threads(n_threads)
    _topk_kernel(rows.astype(np.int64), a_ptr, a_tok, a_w, space.post_ptr, space.post_row, space.n_b, K, n_threads * 8, out_idx, out_sc)
    return (out_idx, out_sc)

@njit(parallel=True, cache=True)
def _pair_overlap_kernel(pi, pj, a_ptr, a_tok, b_ptr, b_tok, idf, grp, max_idf, out):
    n = len(pi)
    for k in prange(n):
        i = pi[k]
        j = pj[k]
        p = a_ptr[i]
        pe = a_ptr[i + 1]
        q = b_ptr[j]
        qe = b_ptr[j + 1]
        sn = 0.0
        sa = 0.0
        tna = 0.0
        taa = 0.0
        tnb = 0.0
        tab = 0.0
        for x in range(p, pe):
            t = a_tok[x]
            if t < 0:
                if t == -2:
                    taa += max_idf
                else:
                    tna += max_idf
            elif grp[t] == 1:
                taa += idf[t]
            else:
                tna += idf[t]
        for x in range(q, qe):
            t = b_tok[x]
            if grp[t] == 1:
                tab += idf[t]
            else:
                tnb += idf[t]
        while p < pe and a_tok[p] < 0:
            p += 1
        while p < pe and q < qe:
            ta = a_tok[p]
            tb = b_tok[q]
            if ta == tb:
                if grp[ta] == 1:
                    sa += idf[ta]
                else:
                    sn += idf[ta]
                p += 1
                q += 1
            elif ta < tb:
                p += 1
            else:
                q += 1
        out[k, 0] = sn
        out[k, 1] = tna
        out[k, 2] = tnb
        out[k, 3] = sa
        out[k, 4] = taa
        out[k, 5] = tab

def pair_overlap(space, a_ptr, a_tok, pi, pj):
    out = np.zeros((len(pi), 6), dtype=np.float32)
    _pair_overlap_kernel(pi.astype(np.int64), pj.astype(np.int64), a_ptr, a_tok, space.b_indptr, space.b_tok, space.idf, space.grp, space.max_idf, out)
    return out

@njit(parallel=True, cache=True)
def _union_kernel(idx1, sc1, idx2, sc2, n_b, n_tasks, u_idx, u_c, u_cr, u_n, u_nr):
    n, K = idx1.shape
    for t in prange(n_tasks):
        pos = np.full(n_b, -1, np.int32)
        for r in range(t, n, n_tasks):
            m = 0
            for i in range(K):
                j = idx1[r, i]
                if j < 0:
                    break
                pos[j] = m
                u_idx[r, m] = j
                u_c[r, m] = sc1[r, i]
                u_cr[r, m] = i
                u_n[r, m] = 0.0
                u_nr[r, m] = K
                m += 1
            for i in range(K):
                j = idx2[r, i]
                if j < 0:
                    break
                q = pos[j]
                if q >= 0:
                    u_n[r, q] = sc2[r, i]
                    u_nr[r, q] = i
                else:
                    u_idx[r, m] = j
                    u_c[r, m] = 0.0
                    u_cr[r, m] = K
                    u_n[r, m] = sc2[r, i]
                    u_nr[r, m] = i
                    m += 1
            for i in range(K):
                j = idx1[r, i]
                if j < 0:
                    break
                pos[j] = -1

def retrieve_union(space, a_ptr, a_tok, rows, K, n_threads):
    a_w = space.weights(a_tok)
    idx1, sc1 = retrieve(space, a_ptr, a_tok, a_w, rows, K, n_threads)
    a_wn = a_w.copy()
    m = a_tok >= 0
    a_wn[m] *= space.grp[a_tok[m]] == 0
    a_wn[a_tok == -2] = 0
    idx2, sc2 = retrieve(space, a_ptr, a_tok, a_wn, rows, K, n_threads)
    n = len(rows)
    u_idx = np.full((n, 2 * K), -1, np.int32)
    u_c = np.zeros((n, 2 * K), np.float32)
    u_cr = np.full((n, 2 * K), K, np.float32)
    u_n = np.zeros((n, 2 * K), np.float32)
    u_nr = np.full((n, 2 * K), K, np.float32)
    _union_kernel(idx1, sc1, idx2, sc2, space.n_b, n_threads * 8, u_idx, u_c, u_cr, u_n, u_nr)
    return (u_idx, u_c, u_cr, u_n, u_nr, sc1[:, 0].copy(), sc2[:, 0].copy())
RERANK_COLS = ['comb_score', 'comb_rank', 'comb_rel', 'name_score', 'name_rank', 'name_rel', 'name_shared', 'addr_shared', 'name_share_a', 'name_share_b', 'addr_share_a', 'addr_share_b', 'name_tot_a', 'name_tot_b', 'addr_tot_a', 'addr_tot_b', 'cos']

def rerank_features(u, ov):
    u_idx, u_c, u_cr, u_n, u_nr, top_c, top_n = u
    r, k = np.nonzero(u_idx >= 0)
    eps = np.float32(1e-06)
    c, n = (u_c[r, k], u_n[r, k])
    tot_a = ov[:, 1] + ov[:, 4]
    tot_b = ov[:, 2] + ov[:, 5]
    X = np.column_stack([c, u_cr[r, k], c / np.maximum(top_c[r], eps), n, u_nr[r, k], n / np.maximum(top_n[r], eps), ov[:, 0], ov[:, 3], ov[:, 0] / (ov[:, 1] + eps), ov[:, 0] / (ov[:, 2] + eps), ov[:, 3] / (ov[:, 4] + eps), ov[:, 3] / (ov[:, 5] + eps), ov[:, 1], ov[:, 2], ov[:, 4], ov[:, 5], (ov[:, 0] + ov[:, 3]) / np.sqrt(tot_a * tot_b + eps)]).astype(np.float32)
    return (r, k, X)
