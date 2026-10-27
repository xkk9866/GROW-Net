"""Does a field order correctly the alternatives its decision compares?  Held-out decision-anchored probes
(DATA/probe_h0_hotpotqa.pkl; never trained on) contain, per question, the reader likelihood of the round-0 decision and of
its one-move neighbours. For every field we report
  pair   : pairwise accuracy over neighbour pairs whose measured y differs by > 0.05
  regret : best measured y  -  measured y of the neighbour the field ranks first      (lower is better)
  rho    : within-question Spearman correlation
and the same on held-out random coalitions (DATA/teacherfull_hotpotqa.pkl, questions 17500-17999).
python regret.py CKPT1,CKPT2,..."""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.stats import spearmanr
from grow.common import *
from grow.rof import field_value
from grow.infer_rof import load_rof


def score(phi, y):
    pr = [(phi[i] - phi[j]) * (y[i] - y[j]) > 0 for i in range(len(y)) for j in range(i + 1, len(y)) if abs(y[i] - y[j]) > 0.05]
    rho = spearmanr(phi, y).correlation if y.std() > 1e-3 and phi.std() > 1e-6 else np.nan
    return pr, y.max() - y[int(np.argmax(phi))], rho


@torch.no_grad()
def evaluate(net, recs):
    P, R, H = [], [], []
    for s in range(0, len(recs), 4):
        rb = recs[s:s + 4]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b, c, M = net.forward_batch([r["question"] for r in rb], [r["docs"] for r in rb], 24)
        if net.sup == "relevance": b = torch.zeros_like(b); c = torch.zeros_like(c)
        B, D, L = M.shape
        for k, r in enumerate(rb):
            X = torch.zeros(1, len(r["sets"]), D, L, device=M.device)
            for n, (S, _) in enumerate(r["sets"]):
                for j, i in S:
                    if j < D and i < L: X[0, n, j, i] = 1
            phi = field_value(a[k:k + 1], b[k:k + 1], c[k:k + 1], X * M[k:k + 1, None].float())[0].float().cpu().numpy()
            pr, rg, rho = score(phi, np.array([y for _, y in r["sets"]])); P += pr; R.append(rg); H.append(rho)
    return {"pair": float(np.mean(P)), "regret": float(np.mean(R)), "rho": float(np.nanmean(H)), "n_q": len(recs)}


if __name__ == "__main__":
    near = pickle.load(open(f"{DATA}/{os.environ.get('GROW_PROBE_DIR', '')}probe_h0_hotpotqa.pkl", "rb"))
    T = pickle.load(open(f"{DATA}/{os.environ.get('GROW_PROBE_DIR', '')}teacherfull_hotpotqa.pkl", "rb")); QS = pickle.load(open(f"{DATA}/hotpotqa_train.pkl", "rb"))
    rand = [{"question": QS[k]["question"], "docs": QS[k]["docs"], "sets": T[k]} for k in range(17500, 18000) if k in T]
    y_dec = [r["sets"][1][1] for r in near]; y_best = [max(y for _, y in r["sets"]) for r in near]
    out = {"_decision": {"y_decision": float(np.mean(y_dec)), "y_best_neighbour": float(np.mean(y_best)),
                         "decision_is_best": float(np.mean([a >= b - 1e-9 for a, b in zip(y_dec, y_best)]))}}
    print(out, flush=True)
    for ck in sys.argv[1].split(","):
        net = load_rof(ck)
        out[ck] = {"near": evaluate(net, near), "random": evaluate(net, rand)}
        print(ck, out[ck], flush=True)
        del net; torch.cuda.empty_cache()
    json.dump(out, open(f"{RES}/regret{os.environ.get('GROW_REGRET_TAG', '')}.json", "w"), indent=1)
