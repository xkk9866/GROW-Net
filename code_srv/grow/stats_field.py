"""Post-hoc analysis (no part of the method):
  sig      : paired bootstrap of F1 / EM differences between a method and every baseline (10k resamples, per-question preds)
  types    : F1 / tokens / emergent granularity per question type
  field    : what the learned potentials encode -- unary a_i on gold vs non-gold sentences, adjacent interaction b_i for
             pairs whose second sentence opens with an anaphor vs other pairs (anaphor list used ONLY for this probe)
python stats_field.py sig DATASET METHOD BASE1,BASE2,..  [--reader Qwen3-14B-AWQ]
python stats_field.py types DATASET TAG1,TAG2,..
python stats_field.py field DATASET SCORES_TAG"""
import sys, os, pickle, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grow.common import *
from grow.eval_reader import granularity_of_spans


def preds(ds, tag, reader):
    return pickle.load(open(f"{RES}/pred_{ds}_eval_{tag}_{reader}.pkl", "rb"))["metrics"]


def sig(ds, meth, bases, reader="Qwen3-14B-AWQ", B=10000):
    rng = np.random.default_rng(0); m = preds(ds, meth, reader)
    for b in bases:
        if not os.path.exists(f"{RES}/pred_{ds}_eval_{b}_{reader}.pkl"): continue
        o = preds(ds, b, reader); row = [b]
        for k in ("F1", "EM"):
            d = np.array(m[k]) - np.array(o[k]); idx = rng.integers(0, len(d), (B, len(d)))
            bs = d[idx].mean(1); p = float((bs <= 0).mean())
            row += [f"{k} {100 * d.mean():+.2f} [{100 * np.percentile(bs, 2.5):+.2f},{100 * np.percentile(bs, 97.5):+.2f}] p={p:.4f}"]
        print(" | ".join(row))


def types(ds, tags, reader="Qwen3-14B-AWQ"):
    Bn = pickle.load(open(f"{DATA}/{ds}_eval.pkl", "rb")); qs, docs = Bn["queries"], Bn["docs"]
    ty = np.array([q.get("type", "all") for q in qs])
    for tag in tags:
        m = preds(ds, tag, reader); sel = pickle.load(open(f"{RES}/sel_{ds}_eval_{tag}.pkl", "rb"))
        g = [granularity_of_spans(sorted(set(map(tuple, s["sel"]))), docs) if s["sel"] else {"sent": 0, "pas": 0, "doc": 0} for s in sel]
        for t in sorted(set(ty)):
            ix = np.where(ty == t)[0]
            mixed = np.mean([(g[i]["sent"] > 0) and (g[i]["pas"] + g[i]["doc"] > 0) for i in ix])
            print(f"{ds:9s} {t:18s} {tag:14s} n={len(ix):4d} F1={100 * np.mean(np.array(m['F1'])[ix]):5.1f} "
                  f"Tok={np.mean(np.array(m['Tokens'])[ix]):6.1f} sent={np.mean([g[i]['sent'] for i in ix]):.2f} "
                  f"span={np.mean([g[i]['pas'] for i in ix]):.2f} doc={np.mean([g[i]['doc'] for i in ix]):.2f} mixed={mixed:.2f}")


ANA = re.compile(r"^(he|she|it|they|his|her|its|their|this|these|those|the (album|film|song|band|company|city|town|team|book|series|show|group|river|school))\b", re.I)


def field(ds, tag):
    Bn = pickle.load(open(f"{DATA}/{ds}_eval.pkl", "rb")); qs, docs = Bn["queries"], Bn["docs"]
    sc = pickle.load(open(f"{RES}/scores_{ds}_eval_{tag}.pkl", "rb"))
    ag, an, ba, bo = [], [], [], []
    for q, s in zip(qs, sc):
        G = set(map(tuple, q.get("gold", [])))
        for j, d in enumerate(s["cand"]):
            n = int(s["M"][j].sum())
            for i in range(n):
                (ag if (d, i) in G else an).append(float(s["a"][j, i]))
                if i + 1 < n: (ba if ANA.match(docs[d]["sents"][i + 1].strip()) else bo).append(float(s["b"][j, i]))
    f = lambda v: f"mean={np.mean(v):+.3f} P(>0)={np.mean(np.array(v) > 0):.3f} n={len(v)}"
    if ag: print(f"{ds} unary a  gold      {f(ag)}")
    print(f"{ds} unary a  non-gold  {f(an)}")
    print(f"{ds} pair  b  anaphoric {f(ba)}")
    print(f"{ds} pair  b  other     {f(bo)}")


if __name__ == "__main__":
    cmd = sys.argv[1]; rd = sys.argv[sys.argv.index("--reader") + 1] if "--reader" in sys.argv else "Qwen3-14B-AWQ"
    if cmd == "sig": sig(sys.argv[2], sys.argv[3], sys.argv[4].split(","), rd)
    if cmd == "types": types(sys.argv[2], sys.argv[3].split(","), rd)
    if cmd == "field": field(sys.argv[2], sys.argv[3])
