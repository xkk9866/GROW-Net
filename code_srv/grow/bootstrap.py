"""Paired bootstrap of F1 differences between GROW-Net and each baseline (10,000 resamples over questions).
python bootstrap.py DATASET OURS BASE1,BASE2,... [--reader TAG]"""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grow.common import *

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("ours"); ap.add_argument("bases")
ap.add_argument("--reader", default="Qwen__Qwen3-8B-AWQ"); a = ap.parse_args()
ld = lambda t: np.array(pickle.load(open(f"{RES}/pred_{a.dataset}_eval_{t}_{a.reader}.pkl", "rb"))["metrics"]["F1"])
o = ld(a.ours); rng = np.random.default_rng(0); idx = rng.integers(0, len(o), (10000, len(o)))
for b in a.bases.split(","):
    try: x = ld(b)
    except FileNotFoundError: continue
    d = o - x; bs = d[idx].mean(1)
    print(f"{a.dataset} {a.ours} vs {b}: dF1={100 * d.mean():+.2f}  95%CI=[{100 * np.percentile(bs, 2.5):+.2f},{100 * np.percentile(bs, 97.5):+.2f}]  p={(bs <= 0).mean():.4f}")
