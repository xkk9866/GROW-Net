"""Find real examples for Fig. 1 / case study: GROW-Net correct with compact mixed-granularity evidence, Rerank@3 wrong.
python case.py DATASET OURS [BASE]"""
import sys, pickle
sys.path.insert(0, "code")
from grow.common import *
name, ours = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else "rerank@3"
B = pickle.load(open(f"{DATA}/{name}_eval.pkl", "rb")); docs, qs = B["docs"], B["queries"]
S = pickle.load(open(f"{RES}/sel_{name}_eval_{ours}.pkl", "rb"))
P = pickle.load(open(f"{RES}/pred_{name}_eval_{ours}_Qwen__Qwen3-8B-AWQ.pkl", "rb"))
PB = pickle.load(open(f"{RES}/pred_{name}_eval_{base}_Qwen__Qwen3-8B-AWQ.pkl", "rb"))
n = 0
for k, q in enumerate(qs):
    sel = sorted(map(tuple, S[k]["sel"]))
    if not (2 <= len(sel) <= 7) or P["metrics"]["F1"][k] < 1 or PB["metrics"]["F1"][k] > 0.3: continue
    by = {}
    for d, i in sel: by.setdefault(d, []).append(i)
    runs = []
    for d, idx in by.items():
        s = idx[0]
        for a, b in zip(idx, idx[1:] + [None]):
            if b != a + 1: runs.append((d, s, a)); s = b
    if not (any(a == b for _, a, b in runs) and any(b > a for _, a, b in runs)) or len(by) > 3: continue
    n += 1
    print("=" * 100); print(k, q["qid"], q["question"], "| gold:", q["answers"], "| ours:", P["preds"][k], "| base:", PB["preds"][k])
    print("gold SP:", q.get("gold"))
    for d, a, b in runs:
        print(f"  [{docs[d]['title']}] sents {a}-{b} of {len(docs[d]['sents'])}:", " ".join(docs[d]["sents"][a:b + 1])[:400])
    if n >= 12: break
