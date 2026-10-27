"""Empirical Banzhaf statistics of the reader from the random-ablation measurements (Table 'reader prices its context').
For each question: marginal of element e = mean log y(S | e in S) - mean log y(S | e not in S), averaged over elements of a type;
interaction of adjacent pair (i, i+1) = E[f|11] - E[f|10] - E[f|01] + E[f|00].
python reader_price.py SRC"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grow.common import *

src = sys.argv[1]
f = f"{DATA}/teacherab_{src}.pkl"
T = pickle.load(open(f, "rb")) if os.path.exists(f) else pickle.load(open(f"{DATA}/teacherab_{src}.part.pkl", "rb"))[0]
qs = pickle.load(open(f"{DATA}/{src}_train.pkl", "rb"))
stat = {k: [] for k in ["gold", "context", "distractor", "pair_gold_ctx", "pair_other"]}
for k, ms in T.items():
    q = qs[k]; G = set(map(tuple, q["gold"])); ev = {j for j, _ in G}
    sets = [(set(S), np.log(max(y, 1e-6))) for S, y in ms if S]
    if len(sets) < 8: continue
    U = sorted(set().union(*[S for S, _ in sets]))
    def marg(e):
        a = [v for S, v in sets if e in S]; b = [v for S, v in sets if e not in S]
        return np.mean(a) - np.mean(b) if a and b else None
    per = {"gold": [], "context": [], "distractor": []}
    for e in U:
        m = marg(e)
        if m is None: continue
        per["gold" if e in G else "context" if e[0] in ev else "distractor"].append(m)
    for t, v in per.items():
        if v: stat[t].append(np.mean(v))
    pg, po = [], []
    for e in U:
        n = (e[0], e[1] + 1)
        if n not in U: continue
        c = {(x, z): [v for S, v in sets if ((e in S) == x) and ((n in S) == z)] for x in (0, 1) for z in (0, 1)}
        if any(len(v) == 0 for v in c.values()): continue
        I = np.mean(c[1, 1]) - np.mean(c[1, 0]) - np.mean(c[0, 1]) + np.mean(c[0, 0])
        (pg if (e in G) != (n in G) else po).append(I)
    if pg: stat["pair_gold_ctx"].append(np.mean(pg))
    if po: stat["pair_other"].append(np.mean(po))
print(src, "questions", len(T))
for t, v in stat.items():
    v = np.array(v); print(f"  {t:15s} n={len(v):5d} mean={v.mean():+.3f} P(<0)={np.mean(v < 0):.3f} P(>0)={np.mean(v > 0):.3f}")
