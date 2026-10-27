import sys, pickle
sys.path.insert(0, "code")
from grow.common import *
B = pickle.load(open(f"{DATA}/hotpotqa_eval.pkl", "rb")); docs, qs = B["docs"], B["queries"]
k = int(sys.argv[1]) if len(sys.argv) > 1 else 291
for tag in ("rerank@3", "r3_d0"):
    S = pickle.load(open(f"{RES}/sel_hotpotqa_eval_{tag}.pkl", "rb"))[k]
    P = pickle.load(open(f"{RES}/pred_hotpotqa_eval_{tag}_Qwen__Qwen3-8B-AWQ.pkl", "rb"))
    print("==", tag, qs[k]["question"], "| pred:", P["preds"][k], "| F1", P["metrics"]["F1"][k], "| tok", P["metrics"]["Tokens"][k])
    by = {}
    for d, i in S["sel"]: by.setdefault(d, []).append(i)
    for d, idx in by.items():
        print("  ", docs[d]["title"], sorted(idx), "of", len(docs[d]["sents"]), "|", " ".join(docs[d]["sents"][i] for i in sorted(idx))[:300])

