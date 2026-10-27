"""Random-ablation reader teacher (the measurement that defines the reader-utility field).
For every training question we draw R random evidence subsets of a small universe U(q) -- the sentences of the evidence
documents (HotpotQA: gold documents; NQ: up to two answer-bearing documents, a window of 3 sentences around the anchors)
plus the leading sentences of one random distractor document -- including every element independently with
probability 1/2 (the Banzhaf design), and ask the int4 reader how likely it is to produce the gold answer:
    y(S) = exp( mean_t log P_LLM(a_t | q, S, a_<t) ).
No subset is hand-designed and no label is derived: the network later fits log-odds(y(S)) directly.
python teacher_ab.py SRC N R     (SRC in hotpotqa|nq)"""
import sys, os, random, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *


def universe(q, rng, win=3, nd=6):
    docs = q["docs"]; G = sorted(set(map(tuple, q["gold"])))
    ev = list(dict.fromkeys(j for j, _ in G))[:2]
    U = []
    for j in ev:
        n = len(docs[j]["sents"]); anc = [i for jj, i in G if jj == j]
        keep = sorted({k for i in anc for k in range(max(0, i - win), min(n, i + win + 1))})
        U += [(j, i) for i in keep]
    other = [j for j in range(len(docs)) if j not in ev]
    if other:
        j = rng.choice(other); U += [(j, i) for i in range(min(nd, len(docs[j]["sents"])))]
    return U


def subsets(U, R, rng):
    out, seen = [], set()
    for _ in range(4 * R):
        S = tuple(x for x in U if rng.random() < 0.5)
        if S and S not in seen: seen.add(S); out.append(S)
        if len(out) == R: break
    return out


def main(src, N, R):
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm import SamplingParams
    train = pickle.load(open(f"{DATA}/{src}_train.pkl", "rb"))[:N]
    rng = random.Random(2027)
    llm = LLM(model=READER, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("GROW_GPU_UTIL", 0.85)),
              max_model_len=8192, max_num_batched_tokens=int(os.environ.get("GROW_MAXBT", 4096)), enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    jobs, meta = [], []
    for k, q in enumerate(train):
        a_ids = tok(q["answers"][0], add_special_tokens=False)["input_ids"]
        U = universe(q, rng)
        for S in [()] + subsets(U, R, rng):
            ctx = format_context(list(S), q["docs"]) if S else ""
            msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(q["question"], ctx)}]
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            jobs.append(TokensPrompt(prompt_token_ids=list(p_ids) + a_ids)); meta.append((k, S, len(a_ids)))
    print(f"{src}: {len(train)} questions, {len(jobs)} prompts", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    t = time.time(); part = f"{DATA}/teacherab_{src}.part.pkl"
    res, done = pickle.load(open(part, "rb")) if os.path.exists(part) else ({}, 0)   # resumable after a crash
    for s in range(done, len(jobs), 20000):
        outs = llm.generate(jobs[s:s + 20000], sp, use_tqdm=False)
        for o, (k, S, na) in zip(outs, meta[s:s + 20000]):
            ids = o.prompt_token_ids; lp = o.prompt_logprobs
            res.setdefault(k, []).append((S, float(np.exp(np.mean([lp[t][ids[t]].logprob for t in range(len(ids) - na, len(ids))])))))
        print(f"  {s + len(outs)}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
        dump((res, s + len(outs)), part)
    dump(res, f"{DATA}/teacherab_{src}.pkl")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
