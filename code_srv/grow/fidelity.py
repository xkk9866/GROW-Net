"""Field fidelity: does the network predict the reader's own answer likelihood on questions it was never trained on?
Hold-out = the last HOLD measured questions of each source (excluded from every training run by train_field.py).
For every held-out question and every measured random subset S we compare the predicted log-odds beta_q + Phi(S) with the
reader's measured log-odds logit y(S):
  rank  : within-question Spearman correlation between predicted and measured utility of the ~24 subsets
  pair  : within-question pairwise accuracy (subset pairs whose measured y differs by > 0.1)
  auc   : AUC of Phi(S) for "reader answers" (y(S) > 0.5) vs not, pooled over questions
  top   : measured y of the subset the model ranks first, vs the mean y of all subsets and the best measured subset
Reference surrogates (no network): gold_count = number of gold supporting sentences in S (uses gold labels), size = -|S|.
python fidelity.py CKPT1,CKPT2,...|gold_count|size  [--src hotpotqa,nq]"""
import sys, os, pickle, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from grow.common import *
from grow.field import field_value, expected_logit
from grow.infer_field import load_field

HOLD = {("hotpotqa", "ab"): range(11500, 12000), ("hotpotqa", "full"): range(17500, 18000), ("nq", "ab"): range(3800, 4000), ("nq", "full"): range(3800, 4000)}


@torch.no_grad()
def run(tag, src, design="full", bs=4):
    ref = tag in ("gold_count", "size")
    net = None if ref else load_field(tag)[0]
    ab = pickle.load(open(f"{DATA}/teacher{design}_{src}.pkl", "rb")); qs = pickle.load(open(f"{DATA}/{src}_train.pkl", "rb"))
    ks = [k for k in HOLD[(src, design)] if k in ab]
    rho, pair, P, Y, top, avg, best = [], [], [], [], [], [], []

    def add(phi, y):
        P.extend(phi); Y.extend(y)
        if y.std() > 1e-3 and phi.std() > 1e-6: rho.append(spearmanr(phi, y).correlation)
        c = t = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if abs(y[i] - y[j]) > 0.1: t += 1; c += (phi[i] - phi[j]) * (y[i] - y[j]) > 0
        if t: pair.append(c / t)
        top.append(y[int(np.flatnonzero(phi == phi.max())[0])]); avg.append(y.mean()); best.append(y.max())
    for s in range(0, len(ks), bs):
        kb = ks[s:s + bs]; qb = [qs[k] for k in kb]
        if ref:
            for k in kb:
                sets = [(S, y) for S, y in ab[k] if S]; G = set(map(tuple, qs[k]["gold"]))
                phi = np.array([len(G & set(S)) if tag == "gold_count" else -len(S) for S, _ in sets], float)
                y = np.array([v for _, v in sets]); add(phi, y)
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b, M = net.forward_batch([q["question"] for q in qb], [q["docs"] for q in qb], 24)
        if hasattr(net, "crf") or not net.use_pair: b = torch.zeros_like(b)
        B, D, L = M.shape
        for bb, k in enumerate(kb):
            sets = [(S, y) for S, y in ab[k] if S]
            X = torch.zeros(1, len(sets), D, L, device=M.device)
            for n, (S, _) in enumerate(sets):
                for j, i in S:
                    if j < D and i < L: X[0, n, j, i] = 1
            X = X * M[bb:bb + 1, None].float(); phi = field_value(a[bb:bb + 1], b[bb:bb + 1], X)[0].float()
            if getattr(net, "use_var", False):   # predicted expected log-odds (closed-book value measured, as in training)
                y0 = min(max([y for S, y in ab[k] if not S][0], 1e-4), 1 - 1e-4)
                phi = expected_logit(torch.tensor(np.log(y0 / (1 - y0)), device=phi.device), phi, (X * net._v[bb:bb + 1, None]).sum((-1, -2))[0])
            phi = phi.cpu().numpy()
            y = np.array([v for _, v in sets]); add(phi, y)
    Yb = np.array(Y) > 0.5
    r = {"rank": float(np.nanmean(rho)), "pair": float(np.mean(pair)), "auc": float(roc_auc_score(Yb, P)) if 0 < Yb.mean() < 1 else None,
         "top": float(np.mean(top)), "mean": float(np.mean(avg)), "best": float(np.mean(best)), "n_q": len(ks), "n_sets": len(Y)}
    print(f"{src:9s} {design:4s} {tag:12s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tags"); ap.add_argument("--src", default="hotpotqa,nq"); ap.add_argument("--design", default="full,ab")
    a = ap.parse_args()
    f = f"{RES}/fidelity.json"; out = json.load(open(f)) if os.path.exists(f) else {}
    for tag in a.tags.split(","):
        for src in a.src.split(","):
            for des in a.design.split(","):
                out[f"{tag}|{src}|{des}"] = run(tag, src, des)
            json.dump(out, open(f, "w"), indent=1)
