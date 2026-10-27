"""Equal-budget comparison. Every method with a cost knob traces a quality-cost curve (Rerank@k over k; GROW-Net over the
test-time offset delta). For each token budget B we report the F1 each curve reaches at B, by linear interpolation in
log-tokens between its two neighbouring operating points (no extrapolation), and the area under each curve over the
common budget range (normalised by the log-range).
python budget.py OURS_PREFIX [--reader TAG]      e.g. budget.py r4"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grow.common import *

ap = argparse.ArgumentParser(); ap.add_argument("ours"); ap.add_argument("--reader", default="Qwen__Qwen3-8B-AWQ"); a = ap.parse_args()
HV = json.load(open(f"{ROOT}/results/harvest.json"))


def curve(m, tags):
    pts = sorted((m[t]["Tokens"], m[t]["F1"]) for t in tags if t in m and m[t]["Tokens"] > 0)
    return np.log([p[0] for p in pts]), np.array([p[1] for p in pts])


def at(c, b):
    x, y = c; lb = np.log(b)
    return float(np.interp(lb, x, y)) if x[0] <= lb <= x[-1] else None


out = {}
for d in ("hotpotqa", "2wiki", "nq"):
    m = HV[a.reader][d]
    C = {"rerank": curve(m, ["rerank@1", "rerank@2", "rerank@3", "rerank@5"]),
         "ours": curve(m, [t for t in m if t.startswith(a.ours + "_d")])}
    lo, hi = max(c[0][0] for c in C.values()), min(c[0][-1] for c in C.values())
    grid = np.linspace(lo, hi, 200)
    auc = {k: float(np.trapz(np.interp(grid, *c), grid) / (hi - lo)) if hi > lo else None for k, c in C.items()}
    rows = {f"B={int(m[t]['Tokens'])} ({t})": {k: at(c, m[t]["Tokens"]) for k, c in C.items()} for t in ["rerank@1", "rerank@2", "rerank@3", "rerank@5"]}
    out[d] = {"auc": auc, "range": [float(np.exp(lo)), float(np.exp(hi))], "at": rows}
    print(d, json.dumps(out[d]))
json.dump(out, open(f"{RES}/budget_{a.ours}.json", "w"), indent=1)
