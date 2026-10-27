"""Sufficiency-priced elastic widening: decoding of GROW-Net.

maximise  log Suff(S|q) - lambda * tokens(S)   s.t. tokens(S) <= budget,  stop as soon as Suff(S|q) >= theta.
Moves: (a) open a new evidence span at sentence i together with its dependency closure,
       (b) widen an existing span by one sentence to the left / right.
Each step takes the move with the largest marginal log-sufficiency gain per token (cost-benefit greedy).
With rho fixed, sum_k w_k [1 - prod(1 - pi)] is monotone submodular, so cost-benefit greedy + best singleton enjoys the
classical (1 - 1/sqrt(e)) guarantee (Krause & Guestrin 2005; Lin & Bilmes 2010); dependency closures are treated as items.
Granularity promotion: if a span covers >= eta of a passage/document's tokens, the whole unit is emitted."""
import numpy as np

EPS = 1e-3


def log_suff_np(pi, w, dL, dR, S):
    """pi [K,D,L]; w [K]; dL,dR [D,L]; S [D,L] bool -> scalar log Suff (floored per aspect at EPS)."""
    Sl = np.zeros_like(S); Sl[:, 1:] = S[:, :-1]
    Sr = np.zeros_like(S); Sr[:, :-1] = S[:, 1:]
    rho = (1 - dL * (~Sl)) * (1 - dR * (~Sr))
    p = np.clip(pi * rho[None], 0, 1 - 1e-6)
    miss = (np.log1p(-p) * S[None]).sum((1, 2))
    cov = 1 - np.exp(miss)
    return float((w * np.log(EPS + cov)).sum()), float(np.prod((cov + 1e-12) ** w))


def closure(i, dL_row, dR_row, n, thr=0.5, max_ext=3):
    a = b = i
    while a > 0 and dL_row[a] > thr and i - a < max_ext: a -= 1
    while b < n - 1 and dR_row[b] > thr and b - i < max_ext: b += 1
    return a, b


def widen(pi, w, dL, dR, M, tok, title_tok, theta=0.6, lam=0.0, budget=600, n_seeds=40, eta=0.6, pas=None,
          min_gain=1e-3, use_closure=True, promote=True, min_cost=30):
    """pi [K,D,L], w [K], dL/dR [D,L], M [D,L] bool, tok [D,L] tokens/sentence, title_tok [D].
    pas: optional list per doc of passage spans (a,b) for passage-level promotion.
    returns S [D,L] bool, info dict."""
    K, D, L = pi.shape
    n = M.sum(1)
    S = np.zeros((D, L), bool)
    cur, suff = log_suff_np(pi, w, dL, dR, S)
    # relative sufficiency: stop once S reaches theta x the sufficiency attainable from the whole candidate pool
    target = theta * log_suff_np(pi, w, dL, dR, M.copy())[1]
    score = (w[:, None, None] * pi).max(0) * M
    seeds = np.dstack(np.unravel_index(np.argsort(-score, axis=None)[:n_seeds], (D, L)))[0]
    used = 0; steps = 0
    while suff < target:
        best = None
        cand = []
        for d, i in seeds:
            if not M[d, i] or S[d, i]: continue
            a, b = closure(i, dL[d], dR[d], n[d]) if use_closure else (i, i)
            cand.append((d, a, b))
        for d in np.where(S.any(1))[0]:
            idx = np.where(S[d])[0]
            for i in idx:
                if i > 0 and not S[d, i - 1]: cand.append((d, i - 1, i - 1))
                if i + 1 < n[d] and not S[d, i + 1]: cand.append((d, i + 1, i + 1))
        for d, a, b in cand:
            new = ~S[d, a:b + 1] & M[d, a:b + 1]
            c = tok[d, a:b + 1][new].sum() + (0 if S[d].any() else title_tok[d])
            if c <= 0 or used + c > budget: continue
            S2 = S.copy(); S2[d, a:b + 1] |= M[d, a:b + 1]
            v, s2 = log_suff_np(pi, w, dL, dR, S2)
            r = (v - cur) / max(c, min_cost)
            if best is None or r > best[0]: best = (r, d, a, b, c, v, s2)
        if best is None or best[0] * 100 < max(lam, 0) or (best[5] - cur) < min_gain: break
        r, d, a, b, c, v, s2 = best
        S[d, a:b + 1] |= M[d, a:b + 1]; used += c; cur, suff = v, s2; steps += 1
    # granularity promotion: fill passages / documents that are mostly covered already
    if promote:
        for d in np.where(S.any(1))[0]:
            dt = tok[d][M[d]].sum(); st = tok[d][S[d]].sum()
            if dt > 0 and st >= eta * dt:
                S[d] = M[d]; continue
            if pas is not None:
                for a, b in pas[d]:
                    seg = tok[d, a:b][M[d, a:b]].sum(); sc = tok[d, a:b][S[d, a:b]].sum()
                    if seg > 0 and sc > 0 and sc >= eta * seg: S[d, a:b] = M[d, a:b]
    return S, {"suff": suff, "steps": steps}
