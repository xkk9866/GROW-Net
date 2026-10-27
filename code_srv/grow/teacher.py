"""Reader-grounded sufficiency supervision.
For every training question we build a family of counterfactual evidence sets (gold, gold+neighbours, whole gold docs,
partial evidence missing one hop, single sentences with/without their left neighbour, distractors, ...) and ask the
int4 (AWQ) reader LLM how likely it is to produce the gold answer from each set:
    y(S) = exp( mean_t log P_LLM(a_t | q, S, a_<t) ),
    target(S) = max(0, y(S) - y(empty)) / (1 - y(empty))      (evidence-attributable sufficiency)
Usage: python teacher.py SHARD NSHARD   (one process per GPU)"""
import sys, os, random, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *


def build_sets(q, rng):
    docs, G = q["docs"], sorted(set(map(tuple, q["gold"])))
    gd = sorted({j for j, _ in G})
    sets = {}
    def add(name, S):
        S = tuple(sorted(set((j, i) for j, i in S if 0 <= i < len(docs[j]["sents"]))))
        if S and S not in sets.values(): sets[f"{name}"] = S
    add("gold", G)
    add("gold_nbr", G + [(j, i + o) for j, i in G for o in (-1, 1)])
    add("gold_docs", [(j, i) for j in gd for i in range(len(docs[j]["sents"]))])
    if len(gd) >= 2:
        for d in gd:
            add(f"partial_{d}", [x for x in G if x[0] != d])
            add(f"partialdoc_{d}", [(j, i) for j in gd if j != d for i in range(len(docs[j]["sents"]))])
    for k, (j, i) in enumerate(G):
        add(f"single_{k}", [(j, i)])
        if i > 0: add(f"singleL_{k}", [(j, i - 1), (j, i)])
        if i + 1 < len(docs[j]["sents"]): add(f"singleR_{k}", [(j, i), (j, i + 1)])
    neg_pool = [(j, i) for j in range(len(docs)) if j not in gd for i in range(len(docs[j]["sents"]))]
    if neg_pool:
        N = rng.sample(neg_pool, min(3, len(neg_pool)))
        add("neg", N); add("gold_plus_neg", G + N)
        add("neg_doc", [(N[0][0], i) for i in range(len(docs[N[0][0]]["sents"]))])
    add("lead", [(j, 0) for j in gd])
    return sets


def main(shard, nshard, n_train=None, src="hotpotqa"):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    train = pickle.load(open(f"{DATA}/{src}_train.pkl", "rb"))[:n_train]
    rng = random.Random(1234)
    allsets = [build_sets(q, rng) for q in train]
    mine = [k for k in range(len(train)) if k % nshard == shard]
    llm = LLM(model=READER, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("GROW_GPU_UTIL", 0.70)), max_model_len=8192, max_num_batched_tokens=int(os.environ.get("GROW_MAXBT", 4096)), enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    jobs, meta = [], []
    for k in mine:
        q = train[k]; a = q["answers"][0]
        a_ids = tok(a, add_special_tokens=False)["input_ids"]
        variants = [("empty", ())] + list(allsets[k].items())
        for name, S in variants:
            ctx = format_context(list(S), q["docs"]) if S else ""
            msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(q["question"], ctx)}]
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            jobs.append(TokensPrompt(prompt_token_ids=list(p_ids) + a_ids)); meta.append((k, name, S, len(a_ids)))
    print(f"shard {shard}: {len(mine)} questions, {len(jobs)} prompts", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    t = time.time(); res = {}
    for s in range(0, len(jobs), 20000):
        outs = llm.generate(jobs[s:s + 20000], sp, use_tqdm=False)
        for o, (k, name, S, na) in zip(outs, meta[s:s + 20000]):
            ids = o.prompt_token_ids; lp = o.prompt_logprobs
            vals = [lp[t][ids[t]].logprob for t in range(len(ids) - na, len(ids))]
            res.setdefault(k, {})[name] = (S, float(np.exp(np.mean(vals))))
        print(f"  {s + len(outs)}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    dump(res, f"{DATA}/teacher_{shard}of{nshard}.pkl" if src == "hotpotqa" else f"{DATA}/teacher_{src}_{shard}of{nshard}.pkl")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "all" else None,
         sys.argv[4] if len(sys.argv) > 4 else "hotpotqa")
