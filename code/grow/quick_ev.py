"""Reader-free evidence diagnostics for selection files: SP-R/P, All-SP, Ans-Recall, tokens, granularity mix."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *
from grow.eval_reader import granularity_of_spans


def evidence(name, split, sels):
    B = pickle.load(open(f"{DATA}/{name}_{split}.pkl", "rb")); docs, qs = B["docs"], B["queries"]
    m = {k: [] for k in ("SP_R", "SP_P", "AllSP", "AnsR", "Tok", "gs", "gp", "gd")}
    for q, s in zip(qs, sels):
        sel = set(map(tuple, s["sel"])); ctx = format_context(s["sel"], docs) if s["sel"] else ""
        if q.get("gold"):
            G = set(map(tuple, q["gold"])); m["SP_R"].append(len(G & sel) / len(G)); m["SP_P"].append(len(G & sel) / max(1, len(sel))); m["AllSP"].append(float(G <= sel))
        m["AnsR"].append(has_answer(ctx, q["answers"]) if q.get("answers") else 0); m["Tok"].append(ntok(ctx) if ctx else 0)
        g = granularity_of_spans(sorted(sel), docs) if sel else {"sent": 0, "pas": 0, "doc": 0}
        m["gs"].append(g["sent"]); m["gp"].append(g["pas"]); m["gd"].append(g["doc"])
    return {k: float(np.mean(v)) * (100 if k in ("SP_R", "SP_P", "AllSP", "AnsR") else 1) for k, v in m.items() if v}


if __name__ == "__main__":
    name, split = sys.argv[1], sys.argv[2]
    for tag in sys.argv[3:]:
        r = evidence(name, split, pickle.load(open(f"{RES}/sel_{name}_{split}_{tag}.pkl", "rb")))
        print(f"{tag:45s} " + " ".join(f"{k}={v:.2f}" for k, v in r.items()), flush=True)
