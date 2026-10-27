"""Decision-anchored probing: measure the reader where the field takes its decisions.
An argmax only needs the field to order correctly the selections it compares, so the reader is probed on the decoder's own
decision x* for a training question (retrieved with the test-time pipeline, including re-reading) and on its one-move
neighbours in the space of run-structured selections:
   empty | x* | x* minus one run | one run extended by one sentence to the left / right | one run shrunk to its best
   sentence | one opened document read in full | x* plus the best run of one of the 3 highest-gain closed documents
(+ 2 random run-structured selections for exploration). No gold label is used.
  python probe_rof.py plan CKPT TAG SRC START N      (GPU, field)  -> DATA/plan_TAG_SRC.pkl
  python probe_rof.py measure TAG SRC                (GPU, reader) -> DATA/probe_TAG_SRC.pkl
SRC: hotpotqa (training-question pool, data/hotpotqatr_train.pkl) | nq (NQ-open train questions on the BEIR-NQ corpus)."""
import sys, os, random, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grow.common import *

LMAX = 24


def build_hotpot_pool():
    """open-retrieval training setting built like the evaluation corpus: union (dedup by title) of the paragraphs of the
    training contexts; gold evidence is remapped to the pooled documents (used only for analysis)."""
    f = f"{DATA}/hotpotqatr_train.pkl"
    if os.path.exists(f): return pickle.load(open(f, "rb"))
    tr = pickle.load(open(f"{DATA}/hotpotqa_train.pkl", "rb"))
    docs, t2id, qs = [], {}, []
    for k, q in enumerate(tr):
        loc = {}
        for j, d in enumerate(q["docs"]):
            if d["title"] not in t2id: t2id[d["title"]] = len(docs); docs.append({"doc_id": len(docs), "title": d["title"], "sents": d["sents"]})
            loc[j] = t2id[d["title"]]
        qs.append({"qid": q["qid"], "k": k, "question": q["question"], "answers": q["answers"], "type": q["type"],
                   "gold": [(loc[j], i) for j, i in q["gold"]]})
    B = {"docs": docs, "queries": qs}; dump(B, f)
    print("hotpot training pool: docs", len(docs), "sents", sum(len(d["sents"]) for d in docs), flush=True)
    return B


def load_src(src):
    if src == "hotpotqa": B = build_hotpot_pool(); return B["docs"], B["queries"], "hotpotqatr"
    docs = pickle.load(open(f"{DATA}/nq_eval.pkl", "rb"))["docs"]
    tr = pickle.load(open(f"{DATA}/nq_train_full.pkl", "rb"))
    qs = [{"qid": q["qid"], "k": k, "question": q["question"], "answers": q["answers"], "type": "single", "gold": []} for k, q in enumerate(tr)]
    return docs, qs, "nq"


def runs_of(row):
    idx = np.flatnonzero(row); out = []
    if len(idx) == 0: return out
    s = idx[0]
    for a, b in zip(idx, list(idx[1:]) + [None]):
        if b != a + 1: out.append((int(s), int(a))); s = b
    return out


def neighbours(x, a, c, M, rng, gain, best_chain):
    """x [D,L] bool decision; returns list of selections (tuples of (local doc, sent))."""
    D, L = x.shape; n = M.sum(1)
    S0 = {(j, i) for j in range(D) for i in np.flatnonzero(x[j])}
    runs = [(j, s, e) for j in range(D) for s, e in runs_of(x[j])]
    out = [frozenset(), frozenset(S0)]
    for j, s, e in runs:
        R = {(j, i) for i in range(s, e + 1)}
        out.append(frozenset(S0 - R))
        if s > 0: out.append(frozenset(S0 | {(j, s - 1)}))
        if e + 1 < n[j]: out.append(frozenset(S0 | {(j, e + 1)}))
        if e > s:
            i = s + int(np.argmax(a[j, s:e + 1])); out.append(frozenset((S0 - R) | {(j, i)}))
    for j in sorted({j for j, _, _ in runs}):
        out.append(frozenset(S0 | {(j, i) for i in range(n[j])}))
    closed = [j for j in np.argsort(-gain) if not x[j].any() and n[j] > 0][:3]
    for j in closed:
        out.append(frozenset(S0 | {(j, i) for i in np.flatnonzero(best_chain[j])}))
    for _ in range(2):   # exploration: random run-structured selection
        S = set()
        for j in rng.sample([j for j in range(D) if n[j] > 0], k=min(rng.randint(1, 3), int((n > 0).sum()))):
            g = rng.randint(0, 2); ln = int(n[j])
            if g == 0: i = rng.randrange(ln); S.add((j, i))
            elif g == 1: s = rng.randrange(ln); e = min(ln, s + rng.randint(2, 4)); S |= {(j, i) for i in range(s, e)}
            else: S |= {(j, i) for i in range(ln)}
        out.append(frozenset(S))
    seen, res = set(), []
    for S in out:
        if S not in seen: seen.add(S); res.append(tuple(sorted(S)))
    if len(res) > 16: res = res[:2] + rng.sample(res[2:], 14)   # at most 16 measurements per question
    return res


def plan(ckpt, tag, src, start, N):
    import torch
    from grow.rof import chain_best
    from grow.infer_rof import load_rof, run
    from grow.grow_infer import sentence_index
    docs, qs, iname = load_src(src)
    qs = [q for q in qs if start <= q["k"] < start + N]
    Z, U = sentence_index(iname, docs)
    net = load_rof(ckpt)
    out, cands, pots = run(net, qs, docs, Z, U, keep=True)
    rng = random.Random(start); recs = []
    for q, sel, cc, p in zip(qs, out[0.0], cands, pots):
        a, b, c, M = p["a"], p["b"], p["c"], p["M"]
        v, bc = chain_best(a, b, M)
        gain = (c + v).numpy(); pos = {d: j for j, d in enumerate(cc)}
        x = np.zeros(M.shape, bool)
        for d, i in sel["sel"]: x[pos[d], i] = True
        sets = neighbours(x, a.numpy(), c.numpy(), M.numpy(), rng, gain, bc.numpy())
        recs.append({"k": q["k"], "qid": q["qid"], "question": q["question"], "answers": q["answers"],
                     "gold": [(pos[d], i) for d, i in q["gold"] if d in pos],
                     "docs": [{"title": docs[d]["title"], "sents": docs[d]["sents"][:LMAX]} for d in cc], "sets": sets})
    dump(recs, f"{DATA}/plan_{tag}_{src}.pkl")
    print("planned", len(recs), "questions, sets/q %.1f" % np.mean([len(r["sets"]) for r in recs]), flush=True)


def fixed_kv_budget():
    """GPU 3 is shared with another job whose memory vLLM's profiler counts as our own; when GROW_KV_GB is set the
    engine runs in-process and receives a fixed KV-cache budget instead of the profiled one."""
    gb = os.environ.get("GROW_KV_GB")
    if not gb: return
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm.v1.worker import gpu_worker
    gpu_worker.Worker.determine_available_memory = lambda self: int(float(gb) * 2 ** 30)


def measure(tag, src):
    fixed_kv_budget()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    recs = pickle.load(open(f"{DATA}/plan_{tag}_{src}.pkl", "rb"))[:int(os.environ.get("GROW_NPROBE", 10 ** 9))]
    llm = LLM(model=READER, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("GROW_GPU_UTIL", 0.85)),
              max_model_len=int(os.environ.get("GROW_MAXLEN", 8192)), max_num_batched_tokens=int(os.environ.get("GROW_MAXBT", 2048)), enable_prefix_caching=True, seed=0, enforce_eager=True)
    tok = llm.get_tokenizer(); jobs, meta = [], []
    for r_i, r in enumerate(recs):
        a_ids = tok(r["answers"][0], add_special_tokens=False)["input_ids"]
        for S in r["sets"]:
            ctx = format_context(list(S), r["docs"]) if S else ""
            msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(r["question"], ctx)}]
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(p_ids) + len(a_ids) > int(os.environ.get("GROW_MAXLEN", 8192)): continue   # longer than the engine limit
            jobs.append(TokensPrompt(prompt_token_ids=list(p_ids) + a_ids)); meta.append((r_i, S, len(a_ids)))
    print(f"{src}: {len(recs)} questions, {len(jobs)} prompts", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0); t = time.time()
    for r in recs: r["sets_m"] = []
    for s in range(0, len(jobs), 20000):
        outs = llm.generate(jobs[s:s + 20000], sp, use_tqdm=False)
        for o, (r_i, S, na) in zip(outs, meta[s:s + 20000]):
            ids, lp = o.prompt_token_ids, o.prompt_logprobs
            recs[r_i]["sets_m"].append((S, float(np.exp(np.mean([lp[t][ids[t]].logprob for t in range(len(ids) - na, len(ids))])))))
        print(f"  {s + len(outs)}/{len(jobs)} {time.time() - t:.0f}s", flush=True)
    for r in recs: r["sets"] = r.pop("sets_m")
    dump(recs, f"{DATA}/probe_{tag}_{src}.pkl")


if __name__ == "__main__":
    if sys.argv[1] == "plan": plan(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    elif sys.argv[1] == "measure": measure(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "pool":
        from grow.grow_infer import sentence_index
        B = build_hotpot_pool(); sentence_index("hotpotqatr", B["docs"])
