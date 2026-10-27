"""Full-candidate reader probe (the measurement that identifies the reader field on the domain it is decided on).
The field is only identified on sentences that the reader has actually been shown. Decisions at test time are made over
ALL sentences of the candidate documents, so here the probed universe U(q) is every sentence of every candidate document
of a training question (HotpotQA: its ten paragraphs; NQ: the top-6 retrieved documents plus any answer-bearing one,
<= 24 sentences each). Each of R random coalitions first draws an inclusion rate p ~ U(0,1) and then keeps every
sentence of U(q) independently with probability p (the Shapley design: all coalition sizes are equally likely), so the
reader is observed both on a few sentences and on nearly complete contexts. The empty context is always scored.
    y(S) = exp( mean_t log P_LLM(a_t | q, S, a_<t) )
python teacher_full.py SRC START N R     (SRC in hotpotqa|nq; questions START..START+N-1 of DATA/SRC_train.pkl)"""
import sys, os, random, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *

LMAX = 24


def universe(q, src):
    G = {j for j, _ in q["gold"]}
    docs = range(len(q["docs"])) if src == "hotpotqa" else sorted(set(range(min(6, len(q["docs"])))) | G)
    return [(j, i) for j in docs for i in range(min(LMAX, len(q["docs"][j]["sents"])))]


def coalitions(U, R, rng):
    out, seen = [], set()
    for _ in range(8 * R):
        p = rng.random(); S = tuple(x for x in U if rng.random() < p)
        if S and S not in seen: seen.add(S); out.append(S)
        if len(out) == R: break
    return out


def main(src, start, N, R):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    train = pickle.load(open(f"{DATA}/{src}_train.pkl", "rb"))
    rng = random.Random(4242 + start)
    llm = LLM(model=READER, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("GROW_GPU_UTIL", 0.85)),
              max_model_len=8192, max_num_batched_tokens=int(os.environ.get("GROW_MAXBT", 8192)), enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    jobs, meta = [], []
    for k in range(start, min(start + N, len(train))):
        q = train[k]; a_ids = tok(q["answers"][0], add_special_tokens=False)["input_ids"]
        for S in [()] + coalitions(universe(q, src), R, rng):
            ctx = format_context(list(S), q["docs"]) if S else ""
            msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt(q["question"], ctx)}]
            p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            jobs.append(TokensPrompt(prompt_token_ids=list(p_ids) + a_ids)); meta.append((k, S, len(a_ids)))
    print(f"{src}: questions {start}..{start + N - 1}, {len(jobs)} prompts, "
          f"mean prompt {np.mean([len(j['prompt_token_ids']) for j in jobs]):.0f} tokens", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    t = time.time(); part = f"{DATA}/teacherfull_{src}.part.pkl"
    res, done = pickle.load(open(part, "rb")) if os.path.exists(part) else ({}, 0)   # resumable after a crash
    for s in range(done, len(jobs), 10000):
        outs = llm.generate(jobs[s:s + 10000], sp, use_tqdm=False)
        for o, (k, S, na) in zip(outs, meta[s:s + 10000]):
            ids = o.prompt_token_ids; lp = o.prompt_logprobs
            res.setdefault(k, []).append((S, float(np.exp(np.mean([lp[t][ids[t]].logprob for t in range(len(ids) - na, len(ids))])))))
        print(f"  {s + len(outs)}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
        dump((res, s + len(outs)), part)
    dump(res, f"{DATA}/teacherfull_{src}.pkl")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
