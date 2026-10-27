"""Act-level reader measurement: for every probed selection S, let the int4 reader answer greedily (the exact decoding used at
test time) and record the token-F1 of its answer,  y_act(S) = F1(greedy_R(q, S), a).  This replaces the teacher-forced
likelihood of the gold answer, which keeps rising with any related text even when the reader's answer no longer changes.
Inputs are the existing probe designs (same selections, same prompts); outputs keep their structure:
  DATA/gen/teacher{ab,full}_{src}.pkl : {k: [(S, y_act)]}        DATA/gen/probe_{p1,h0}_{src}.pkl : [record with sets=[(S, y_act)]]
python probe_gen.py PART NPARTS      (prompts of all designs are pooled and split into NPARTS shards; then `merge`)"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *

OUT = f"{DATA}/gen"
SRCS = [("teacherab", "hotpotqa"), ("teacherab", "nq"), ("teacherfull", "hotpotqa"), ("teacherfull", "nq"),
        ("probe_p1", "hotpotqa"), ("probe_p1", "nq"), ("probe_h0", "hotpotqa")]


def items():
    """(file id, key, question, answers, docs, S) for every measured selection, in a fixed order."""
    QS = {s: pickle.load(open(f"{DATA}/{s}_train.pkl", "rb")) for s in ("hotpotqa", "nq")}
    out = []
    for des, src in SRCS:
        f = f"{DATA}/{des}_{src}.pkl"
        if not os.path.exists(f): continue
        T = pickle.load(open(f, "rb"))
        if isinstance(T, dict):
            for k in sorted(T):
                q = QS[src][k]
                for S, _ in T[k]: out.append(((des, src), k, q["question"], q["answers"], q["docs"], S))
        else:
            for r_i, r in enumerate(T):
                for S, _ in r["sets"]: out.append(((des, src), r_i, r["question"], r["answers"], r["docs"], S))
    return out


def run(part, nparts):
    from vllm import LLM, SamplingParams
    allit = items(); it = allit[part::nparts]
    llm = LLM(model=READER, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("GROW_GPU_UTIL", 0.85)),
              max_model_len=8192, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer(); prompts, keep = [], []
    for j, (_, _, qn, ans, docs, S) in enumerate(it):
        ctx = format_context(list(S), docs) if S else ""
        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(qn, ctx)}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tok(p, add_special_tokens=False)["input_ids"]) < 8100: prompts.append(p); keep.append(j)
    print(f"part {part}/{nparts}: {len(it)} selections, {len(prompts)} prompts", flush=True)
    sp = SamplingParams(max_tokens=32, temperature=0.0); res = {}; t = time.time()
    for s in range(0, len(prompts), 20000):
        outs = llm.generate(prompts[s:s + 20000], sp, use_tqdm=False)
        for o, j in zip(outs, keep[s:s + 20000]):
            res[j] = f1(o.outputs[0].text.strip().split("\n")[0], it[j][3])
        print(f"  {s + len(outs)}/{len(prompts)} {time.time() - t:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True); dump(res, f"{OUT}/part{part}of{nparts}.pkl")


def merge(nparts):
    allit = items(); y = {}
    for p in range(nparts):
        for j, v in pickle.load(open(f"{OUT}/part{p}of{nparts}.pkl", "rb")).items(): y[j * nparts + p] = v
    for des, src in SRCS:
        f = f"{DATA}/{des}_{src}.pkl"
        if not os.path.exists(f): continue
        T = pickle.load(open(f, "rb")); idx = [n for n, x in enumerate(allit) if x[0] == (des, src)]; c = 0
        if isinstance(T, dict):
            G = {}
            for k in sorted(T):
                G[k] = [(S, y[idx[c + i]]) for i, (S, _) in enumerate(T[k]) if idx[c + i] in y]; c += len(T[k])
        else:
            G = []
            for r in T:
                r2 = dict(r); r2["sets"] = [(S, y[idx[c + i]]) for i, (S, _) in enumerate(r["sets"]) if idx[c + i] in y]; c += len(r["sets"]); G.append(r2)
        dump(G, f"{OUT}/{des}_{src}.pkl"); print("wrote", des, src, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge": merge(int(sys.argv[2]))
    else: run(int(sys.argv[1]), int(sys.argv[2]))
