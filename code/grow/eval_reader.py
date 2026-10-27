"""Reader evaluation with an int4 (AWQ) LLM served by vLLM.
python eval_reader.py DATASET TAG1,TAG2,... [--split eval] [--reader PATH]
For each selection file RES/sel_{dataset}_{split}_{tag}.pkl compute
  answer quality : EM, F1, Acc (answer contained in prediction)
  evidence       : SP-Recall / SP-Precision / All-SP (multi-hop, sentence level), Doc-Recall, Ans-Recall (answer string in context),
                   Gold-passage hit (NQ)
  cost           : context tokens (reader tokenizer), #units, granularity mix (sentence / passage / document spans)."""
import sys, os, argparse, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grow.common import *


def granularity_of_spans(sel, docs):
    """classify every contiguous selected span as sentence / passage-or-larger / whole document."""
    by = {}
    for d, i in sel: by.setdefault(d, []).append(i)
    cnt = {"sent": 0, "pas": 0, "doc": 0}
    for d, idx in by.items():
        idx = sorted(idx); n = len(docs[d]["sents"])
        if len(idx) == n and n > 1: cnt["doc"] += 1; continue
        runs, s = [], idx[0]
        for a, b in zip(idx, idx[1:] + [None]):
            if b != a + 1: runs.append((s, a)); s = b
        for a, b in runs: cnt["sent" if a == b else "pas"] += 1
    return cnt


def evaluate(name, tags, split="eval", reader=READER, closed_book=True):
    B = pickle.load(open(f"{DATA}/{name}_{split}.pkl", "rb")); docs, qs = B["docs"], B["queries"]
    from vllm import LLM, SamplingParams
    llm = LLM(model=reader, quantization="awq_marlin" if "AWQ" in reader else None, gpu_memory_utilization=0.85,
              max_model_len=8192, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    rtag = os.path.basename(reader.rstrip("/"))
    results = {}
    runs = ([("closed_book", None)] if closed_book else []) + [(t, pickle.load(open(f"{RES}/sel_{name}_{split}_{t}.pkl", "rb"))) for t in tags if os.path.exists(f"{RES}/sel_{name}_{split}_{t}.pkl")]
    for tag, sels in runs:
        prompts, ctxs = [], []
        for k, q in enumerate(qs):
            sel = sels[k]["sel"] if sels else []
            assert sels is None or sels[k]["qid"] == q["qid"]
            ctx = format_context(sel, docs) if sel else ""
            ctxs.append(ctx)
            msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(q["question"], ctx)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        preds = [o.outputs[0].text.strip().split("\n")[0] for o in outs]
        m = {"EM": [], "F1": [], "Acc": [], "Tokens": [], "AnsRecall": [], "SP_R": [], "SP_P": [], "AllSP": [], "DocR": [], "PasHit": [],
             "g_sent": [], "g_pas": [], "g_doc": []}
        ntoks = [len(x) for x in tok(ctxs, add_special_tokens=False)["input_ids"]]
        for k, q in enumerate(qs):
            p, golds = preds[k], q["answers"]
            m["EM"].append(em(p, golds)); m["F1"].append(f1(p, golds)); m["Acc"].append(acc(p, golds))
            m["Tokens"].append(ntoks[k]); m["AnsRecall"].append(has_answer(ctxs[k], golds))
            sel = set(map(tuple, sels[k]["sel"])) if sels else set()
            if q.get("gold"):
                G = set(map(tuple, q["gold"]))
                m["SP_R"].append(len(G & sel) / len(G)); m["SP_P"].append(len(G & sel) / max(1, len(sel)))
                m["AllSP"].append(float(G <= sel))
                gd = {d for d, _ in G}; m["DocR"].append(len(gd & {d for d, _ in sel}) / len(gd))
            if q.get("gold_pas"):
                hit = any(any((d, i) in sel for i in range(*doc_passages(docs[d])[p])) for d, p in q["gold_pas"])
                m["PasHit"].append(float(hit))
            if sels:
                g = granularity_of_spans(sorted(sel), docs)
                for kk in ("sent", "pas", "doc"): m["g_" + kk].append(g[kk])
        summ = {k: float(np.mean(v)) * (100 if k in ("EM", "F1", "Acc", "AnsRecall", "SP_R", "SP_P", "AllSP", "DocR", "PasHit") else 1)
                for k, v in m.items() if v}
        summ["F1_per_1k_tok"] = summ["F1"] / max(1e-9, summ["Tokens"] / 1000) if summ["Tokens"] > 0 else None
        results[tag] = summ
        dump({"preds": preds, "metrics": m}, f"{RES}/pred_{name}_{split}_{tag}_{rtag}.pkl")
        print(f"{name:10s} {tag:40s} " + " ".join(f"{k}={v:.2f}" for k, v in summ.items() if isinstance(v, float) and not k.startswith("g_")) +
              f" | gran s/p/d={summ.get('g_sent', 0):.2f}/{summ.get('g_pas', 0):.2f}/{summ.get('g_doc', 0):.2f} ({time.time() - t:.0f}s)", flush=True)
    f = f"{RES}/metrics_{name}_{split}_{rtag}.json"
    old = json.load(open(f)) if os.path.exists(f) else {}
    old.update(results); json.dump(old, open(f, "w"), indent=1)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("tags"); ap.add_argument("--split", default="eval")
    ap.add_argument("--reader", default=READER); ap.add_argument("--no_cb", action="store_true")
    a = ap.parse_args()
    evaluate(a.dataset, [t for t in a.tags.split(",") if t], a.split, a.reader, closed_book=not a.no_cb)
