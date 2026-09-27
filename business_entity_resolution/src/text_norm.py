import os
import re
import unicodedata
_INDIC_OFF = {1: 'n', 2: 'n', 21: 'k', 22: 'k', 23: 'g', 24: 'g', 25: 'n', 26: 'c', 27: 'c', 28: 'j', 29: 'j', 30: 'n', 31: 't', 32: 't', 33: 'd', 34: 'd', 35: 'n', 36: 't', 37: 't', 38: 'd', 39: 'd', 40: 'n', 41: 'n', 42: 'p', 43: 'f', 44: 'b', 45: 'b', 46: 'm', 47: '', 48: 'r', 49: 'r', 50: 'l', 51: 'l', 52: 'l', 53: 'b', 54: 's', 55: 's', 56: 's', 57: '', 88: 'k', 89: 'k', 90: 'g', 91: 'j', 92: 'r', 93: 'r', 94: 'f', 95: ''}
_TRANS = {}
for _base in range(2304, 3456, 128):
    for _off in range(128):
        _ch = _base + _off
        if _off in _INDIC_OFF:
            _TRANS[_ch] = _INDIC_OFF[_off]
        elif 102 <= _off <= 111:
            _TRANS[_ch] = str(_off - 102)
        else:
            _TRANS[_ch] = ''
_TRANS[2970] = 's'
_TRANS[2947] = ''
_TRANS[8204] = ''
_TRANS[8205] = ''
_TRANS[ord('&')] = ' and '
_TRANS[ord('°')] = ' '
_TRANS[ord('º')] = ' '
_INDIC_RE = re.compile('[ऀ-ൿ]')
_DOMAIN_RE = re.compile('(^www\\.)|(\\.(com|net|org|in|co\\.in|co|biz|info|fr|us)$)|(^@)')
_NONALNUM_RE = re.compile('[^a-z0-9]+')
_MIXED_DIGIT = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b'})
GENERIC_NAME_WORDS = set('\ninc incorporated llc l l c ltd limited pvt private corp corporation co company cos lp llp pc plc\npllc lllp pa psc na the and of dr sri shri smt m s ms mr mrs messrs\nservices service svcs group grp center centre\nsarl sas sasu sa sci eurl ei snc scp scm selarl gie ets etablissements et de du des la le les\ngmbh ag bv nv\n'.split())
_ADDR_MAP = {'street': 'st', 'str': 'st', 'saint': 'st', 'road': 'rd', 'avenue': 'ave', 'av': 'ave', 'boulevard': 'blvd', 'bd': 'blvd', 'boul': 'blvd', 'drive': 'dr', 'lane': 'ln', 'court': 'ct', 'circle': 'cir', 'place': 'pl', 'suite': 'unit', 'apartment': 'apt', 'floor': 'fl', 'flr': 'fl', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'highway': 'hwy', 'parkway': 'pkwy', 'terrace': 'ter', 'trail': 'trl', 'square': 'sq', 'mount': 'mt', 'fort': 'ft', 'point': 'pt', 'rue': 'r', 'chemin': 'ch', 'che': 'ch', 'chem': 'ch', 'route': 'rte', 'impasse': 'imp', 'allee': 'all', 'quai': 'qu', 'cours': 'crs', 'faubourg': 'fbg', 'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th', 'nagar': 'ngr', 'marg': 'mg', 'colony': 'col', 'sector': 'sec', 'opposite': 'opp', 'opp': 'opp', 'near': 'nr', 'building': 'bldg', 'bldg': 'bldg', 'number': 'no', 'nos': 'no', 'num': 'no', 'door': 'no', 'house': 'h', 'hno': 'h', 'calcutta': 'kolkata', 'bombay': 'mumbai', 'madras': 'chennai', 'bengaluru': 'bangalore', 'gurugram': 'gurgaon'}
_ADDR_DROP = {'null', 'na', 'n', 'a', 'none', 'nan', 'unit', 'apt', 'fl', 'no', 'h', 'pmb', 'city', 'icty', 'township', 'of', 'the', 'de', 'du', 'des', 'la', 'le', 'les', 'd', 'l', 'bis', 'ter'}
_ADDR_GENERIC = {'st', 'rd', 'ave', 'blvd', 'dr', 'ln', 'ct', 'cir', 'pl', 'hwy', 'pkwy', 'ter', 'trl', 'r', 'ch', 'rte', 'imp', 'all', 'ngr', 'mg', 'col', 'sec', 'opp', 'nr', 'bldg', 'plot', 'main', 'cross', 'road', 'e', 'w', 's', 'n', 'ne', 'nw', 'se', 'sw', 'ground', 'stage', 'phase', 'block', 'old', 'new', 'layout', 'area'}

def _hash(s):
    return hash(s)

def _basic(text):
    if not text:
        return ''
    t = text.lower()
    if _INDIC_RE.search(t):
        t = t.translate(_TRANS)
    else:
        t = t.translate(_TRANS) if '&' in t or '°' in t or 'º' in t else t
    if not t.isascii():
        t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
    return t

def _fix_token(tok):
    if tok.isalpha() or tok.isdigit():
        return tok
    nd = sum((ch.isdigit() for ch in tok))
    if nd <= 2 and len(tok) - nd >= 3:
        return tok.translate(_MIXED_DIGIT)
    return tok
_SKEL_SUBS = [('ph', 'f'), ('ck', 'k'), ('ch', 'c'), ('sh', 's'), ('th', 't'), ('kh', 'k'), ('gh', 'g'), ('dh', 'd'), ('bh', 'b'), ('q', 'k'), ('x', 'ks'), ('z', 'j'), ('w', 'b'), ('v', 'b')]
_SKEL_DROP = str.maketrans('d', 't', 'aeiouyh')
_C_SOFT = re.compile('c(?=[eiy])')
_G_SOFT = re.compile('g(?=[eiy])')

def skeleton(tok):
    if tok.isdigit():
        return tok
    t = _C_SOFT.sub('s', tok)
    if 'g' in t:
        t = _G_SOFT.sub('j', t)
    for a, b in _SKEL_SUBS:
        if a in t:
            t = t.replace(a, b)
    t = t.replace('c', 'k') if 'c' in t else t
    t = t.translate(_SKEL_DROP)
    if len(t) > 1:
        out = [t[0]]
        for ch in t[1:]:
            if ch != out[-1]:
                out.append(ch)
        t = ''.join(out)
    if len(t) > 3:
        if t.endswith('ng'):
            t = t[:-1]
        if t[-1] in 'sj':
            t = t[:-1]
    return t
_GENERIC_SKEL = {skeleton(w) for w in GENERIC_NAME_WORDS if len(w) > 2} | {'prbt', 'lmtd', 'lmtt', 'lmt', 'prbtlmtd', 'lk', 'lp', 'ltd', 'krp', 'nk', 'sr', 'srbss', 'srbs'}

def normalize_name(raw):
    indic = bool(raw) and _INDIC_RE.search(raw) is not None
    t = _basic(raw).replace("'", '').replace('’', '')
    t = _DOMAIN_RE.sub(' ', t.strip())
    toks = [_fix_token(x) for x in _NONALNUM_RE.split(t) if x]
    if len(toks) == 1 and len(toks[0]) > 8 and toks[0].endswith('com'):
        toks = [toks[0][:-3]]
    name = ' '.join(toks)
    if indic:
        core_toks = [x for x in toks if skeleton(x) not in _GENERIC_SKEL]
    else:
        core_toks = [x for x in toks if x not in GENERIC_NAME_WORDS]
    if not core_toks:
        core_toks = toks
    skel_toks = []
    for x in core_toks:
        s = skeleton(x)
        if s and (not (indic and s in _GENERIC_SKEL)):
            skel_toks.append(s)
    return (name, core_toks, skel_toks)
_US_STATES = ['al', 'alabama', 'ak', 'alaska', 'az', 'arizona', 'ar', 'arkansas', 'ca', 'california', 'co', 'colorado', 'ct', 'connecticut', 'de', 'delaware', 'fl', 'florida', 'ga', 'georgia', 'hi', 'hawaii', 'id', 'idaho', 'il', 'illinois', 'in', 'indiana', 'ia', 'iowa', 'ks', 'kansas', 'ky', 'kentucky', 'la', 'louisiana', 'me', 'maine', 'md', 'maryland', 'ma', 'massachusetts', 'mi', 'michigan', 'mn', 'minnesota', 'ms', 'mississippi', 'mo', 'missouri', 'mt', 'montana', 'ne', 'nebraska', 'nv', 'nevada', 'nh', 'new hampshire', 'nj', 'new jersey', 'nm', 'new mexico', 'ny', 'new york', 'nc', 'north carolina', 'nd', 'north dakota', 'oh', 'ohio', 'ok', 'oklahoma', 'or', 'oregon', 'pa', 'pennsylvania', 'ri', 'rhode island', 'sc', 'south carolina', 'sd', 'south dakota', 'tn', 'tennessee', 'tx', 'texas', 'ut', 'utah', 'vt', 'vermont', 'va', 'virginia', 'wa', 'washington', 'wv', 'west virginia', 'wi', 'wisconsin', 'wy', 'wyoming', 'dc', 'district of columbia', 'pr', 'puerto rico']
_IN_STATES = ['andhra pradesh', 'ap', 'arunachal pradesh', 'assam', 'as', 'bihar', 'br', 'chhattisgarh', 'cg', 'ct', 'goa', 'gujarat', 'gj', 'haryana', 'hr', 'himachal pradesh', 'hp', 'jharkhand', 'jh', 'jammu and kashmir', 'jammu kashmir', 'jk', 'karnataka', 'ka', 'kerala', 'keralam', 'kl', 'madhya pradesh', 'mp', 'maharashtra', 'mh', 'manipur', 'meghalaya', 'ml', 'mizoram', 'mz', 'nagaland', 'nl', 'odisha', 'orissa', 'od', 'punjab', 'pb', 'rajasthan', 'rj', 'sikkim', 'sk', 'tamil nadu', 'telangana', 'tg', 'ts', 'tripura', 'tr', 'uttar pradesh', 'up', 'uttarakhand', 'uttaranchal', 'uk', 'west bengal', 'wb', 'delhi', 'dl', 'nct of delhi', 'chandigarh', 'ch', 'puducherry', 'pondicherry', 'py', 'ladakh', 'lakshadweep', 'ld', 'andaman and nicobar islands', 'an', 'dadra and nagar haveli and daman and diu', 'dn', 'dd']
_FR_REGIONS = ['hauts de france', 'nouvelle aquitaine', 'pays de la loire', 'ile de france', 'auvergne rhone alpes', 'bourgogne franche comte', 'bretagne', 'centre val de loire', 'corse', 'grand est', 'normandie', 'occitanie', 'provence alpes cote d azur', 'ain', 'aisne', 'allier', 'alpes de haute provence', 'hautes alpes', 'alpes maritimes', 'ardeche', 'ardennes', 'ariege', 'aube', 'aude', 'aveyron', 'bouches du rhone', 'calvados', 'cantal', 'charente', 'charente maritime', 'cher', 'correze', 'corse du sud', 'haute corse', 'cote d or', 'cotes d armor', 'creuse', 'dordogne', 'doubs', 'drome', 'eure', 'eure et loir', 'finistere', 'gard', 'haute garonne', 'gers', 'gironde', 'herault', 'ille et vilaine', 'indre', 'indre et loire', 'isere', 'jura', 'landes', 'loir et cher', 'loire', 'haute loire', 'loire atlantique', 'loiret', 'lot', 'lot et garonne', 'lozere', 'maine et loire', 'manche', 'marne', 'haute marne', 'mayenne', 'meurthe et moselle', 'meuse', 'morbihan', 'moselle', 'nievre', 'nord', 'oise', 'orne', 'pas de calais', 'puy de dome', 'pyrenees atlantiques', 'hautes pyrenees', 'pyrenees orientales', 'bas rhin', 'haut rhin', 'rhone', 'haute saone', 'saone et loire', 'sarthe', 'savoie', 'haute savoie', 'seine maritime', 'seine et marne', 'yvelines', 'deux sevres', 'somme', 'tarn', 'tarn et garonne', 'var', 'vaucluse', 'vendee', 'vienne', 'haute vienne', 'vosges', 'yonne', 'territoire de belfort', 'essonne', 'hauts de seine', 'seine saint denis', 'val de marne', 'val d oise']
_REGIONS = set(_US_STATES) | set(_IN_STATES) | set(_FR_REGIONS)

def _drop_regions(raw):
    if not raw or ',' not in raw:
        return raw
    keep = []
    for comp in raw.split(','):
        c = comp.strip()
        if not c:
            continue
        if _INDIC_RE.search(c) and (not re.search('[A-Za-z0-9]', c)):
            continue
        key = ' '.join(_NONALNUM_RE.split(_basic(c))).strip()
        if key in _REGIONS:
            continue
        keep.append(c)
    return ', '.join(keep) if keep else raw

def normalize_address(raw):
    t = _basic(_drop_regions(raw))
    toks = []
    for x in _NONALNUM_RE.split(t):
        if not x:
            continue
        x = _ADDR_MAP.get(x, x)
        if x.isdigit():
            x = x.lstrip('0') or '0'
        elif x[0].isdigit() and x[-1].isalpha() and x[:-2].isdigit() and (x[-2:] in ('st', 'nd', 'rd', 'th')):
            x = (x[:-2].lstrip('0') or '0') + x[-2:]
        toks.append(x)
    toks = [x for x in toks if x not in _ADDR_DROP]
    return toks
_ORDINAL = re.compile('^\\d+(st|nd|rd|th)$')

def _pairs(toks):
    out = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = (toks[i], toks[j])
            out.append(a + '|' + b if a < b else b + '|' + a)
    return out

def _salient(toks, n):
    u = sorted(set(toks), key=lambda x: (-len(x), x))
    return u[:n]

def record_features(name_raw, addr_raw):
    name, core_toks, skel_toks = normalize_name(name_raw)
    atoks = normalize_address(addr_raw)
    core = ' '.join(core_toks)
    compact = ''.join(core_toks)
    skel = ' '.join(skel_toks)
    addr = ' '.join(atoks)
    nums = [x for x in atoks if x.isdigit()]
    alphas = [x for x in atoks if not x.isdigit() and len(x) >= 3 and (x not in _ADDR_GENERIC) and (not x[0].isdigit() or _ORDINAL.match(x))]
    top_alpha = _salient(alphas, 5)
    top_sk = [skeleton(x) if x[0].isalpha() else x for x in top_alpha]
    keys = set()
    for x in core_toks:
        if len(x) >= 2:
            keys.add('n' + x)
    for sk in skel_toks:
        if len(sk) >= 2:
            keys.add('k' + sk)
    if len(compact) >= 4:
        keys.add('c' + compact[:7])
        if len(compact) >= 10:
            keys.add('e' + compact[-7:])
        if len(compact) >= 12:
            keys.add('C' + compact[:12])
    for pr in _pairs(core_toks[:4]):
        keys.add('p' + pr)
    for pr in _pairs(skel_toks[:4]):
        keys.add('q' + pr)
    if len(skel_toks) >= 2:
        keys.add('f' + ''.join(skel_toks))
    for x in alphas:
        keys.add('a' + x)
    for x in nums[:4]:
        keys.add('d' + x)
    for x in nums[:2]:
        for y in top_sk[:3]:
            keys.add('s' + x + '|' + y)
    for pr in _pairs(top_sk):
        keys.add('b' + pr)
    for x in _salient(skel_toks, 2):
        for y in top_sk[:3]:
            keys.add('x' + x + '|' + y)
    toks = [(_hash(k) & 4611686018427387903) << 1 | (1 if k[0] in 'adsbx' else 0) for k in keys]
    return (name, core, compact, skel, addr, ' '.join(nums), toks)
STORED_COLS = ('name', 'core', 'skel', 'addr')

def _worker(args):
    names, addrs = args
    import numpy as np
    out_cols = ([], [], [], [])
    counts = np.empty(len(names), dtype=np.int32)
    hashes = []
    for i, (n, a) in enumerate(zip(names, addrs)):
        r = record_features(n, a)
        out_cols[0].append(r[0])
        out_cols[1].append(r[1])
        out_cols[2].append(r[3])
        out_cols[3].append(r[4])
        counts[i] = len(r[6])
        hashes.extend(r[6])
    return (out_cols, counts, np.array(hashes, dtype=np.int64))

def diff_tokens(name_a, name_b):
    from rapidfuzz.fuzz import ratio
    ta = name_a.split()
    tb = name_b.split()
    sa = set(ta)
    sb = set(tb)
    ea = [t for t in sb if t not in sa]
    ma = [t for t in sa if t not in sb]
    extra = [t for t in ea if not any((ratio(t, u) >= 75 for u in ma))]
    miss = [t for t in ma if not any((ratio(t, u) >= 75 for u in ea))]
    return (extra, miss)

def _tokstat_worker(args):
    names_a, names_b, labels = args
    ext, mis = ({}, {})
    for a, b, y in zip(names_a, names_b, labels):
        sa = set(a.split())
        sb = set(b.split())
        if len(sa & sb) < max(1, len(sa) - 1):
            continue
        extra, miss = diff_tokens(a, b)
        for t in extra:
            c = ext.setdefault(t, [0, 0])
            c[0] += 1
            c[1] += int(y)
        for t in miss:
            c = mis.setdefault(t, [0, 0])
            c[0] += 1
            c[1] += int(y)
    return (ext, mis)

def _namediff_worker(args):
    import numpy as np
    from rapidfuzz.fuzz import ratio
    names_a, names_b, t_ext, t_mis, p_ext, p_mis = args
    n = len(names_a)
    out = np.empty((n, 9), dtype=np.float32)
    for i, (a, b) in enumerate(zip(names_a, names_b)):
        extra, miss = diff_tokens(a, b)
        re_ = [t_ext.get(t, p_ext) for t in extra]
        rm = [t_mis.get(t, p_mis) for t in miss]
        ta = a.split()
        best = [max((ratio(t, u) for u in ta), default=0.0) for t in extra]
        novel = [t for t, bs in zip(extra, best) if len(t) >= 3 and (not t.isdigit()) and (bs < 50)]
        out[i, 0] = len(extra)
        out[i, 1] = min(re_) if re_ else 1.0
        out[i, 2] = sum(re_) / len(re_) if re_ else 1.0
        out[i, 3] = len(miss)
        out[i, 4] = min(rm) if rm else 1.0
        out[i, 5] = sum((1 for r in re_ if r < 0.05))
        out[i, 6] = len(novel)
        out[i, 7] = (100.0 - min(best)) / 100.0 if best else 0.0
        out[i, 8] = sum((1 for t in novel if t_ext.get(t, p_ext) < 0.2))
    return out

def _chunks(n, size):
    return [(s, min(s + size, n)) for s in range(0, n, size)]

def token_stats(names_a, names_b, labels, n_workers, chunk=200000):
    pool = get_pool(n_workers)
    jobs = [(names_a[s:e], names_b[s:e], labels[s:e]) for s, e in _chunks(len(names_a), chunk)]
    ext, mis = ({}, {})
    for e_, m_ in pool.imap_unordered(_tokstat_worker, jobs):
        for src, dst in ((e_, ext), (m_, mis)):
            for t, (c, p) in src.items():
                d = dst.setdefault(t, [0, 0])
                d[0] += c
                d[1] += p
    return (ext, mis)

def name_diff_features(names_a, names_b, table, n_workers, chunk=100000):
    import numpy as np
    pool = get_pool(n_workers)
    t_ext, t_mis, p_ext, p_mis = (table['extra'], table['missing'], table['prior_extra'], table['prior_missing'])
    jobs = [(names_a[s:e], names_b[s:e], t_ext, t_mis, p_ext, p_mis) for s, e in _chunks(len(names_a), chunk)]
    outs = list(pool.imap(_namediff_worker, jobs))
    return np.concatenate(outs) if outs else np.zeros((0, 9), np.float32)
_POOL = None

def get_pool(n_workers):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        os.environ['PYTHONHASHSEED'] = '0'
        ctx = mp.get_context('spawn')
        _POOL = ctx.Pool(n_workers)
    return _POOL

def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None

def normalize_frame(df, n_workers, chunk=50000):
    import numpy as np
    import pyarrow as pa
    names = df['business_name'].tolist()
    addrs = df['business_address'].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    del names, addrs
    pool = get_pool(n_workers)
    cols = [[] for _ in STORED_COLS]
    counts, hashes = ([], [])
    for oc, cnt, hs in pool.imap(_worker, jobs):
        for c in range(len(STORED_COLS)):
            cols[c].append(pa.array(oc[c], type=pa.string()))
        counts.append(cnt)
        hashes.append(hs)
    del jobs
    out = {k: pa.chunked_array(cols[i], type=pa.string()).combine_chunks() for i, k in enumerate(STORED_COLS)}
    counts = np.concatenate(counts) if counts else np.zeros(0, np.int32)
    indptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    hashes = np.concatenate(hashes) if hashes else np.zeros(0, np.int64)
    return (out, indptr, hashes)
