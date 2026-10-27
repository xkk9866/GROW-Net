"""Open-corpus multi-hop QA benchmarks (HotpotQA / 2WikiMultihopQA) with sentence-level provenance.
Corpus = union (dedup by title) of all paragraphs appearing in the dev contexts (gold + TF-IDF hard distractors).
Eval/tune queries are disjoint random dev subsets; training questions (HotpotQA train) keep their 10 local paragraphs."""
import pandas as pd, glob, random, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *


def load(name, split):
    pat = f"{HF}/hotpot_qa/distractor/{split}-*.parquet" if name == "hotpotqa" else f"{HF}/framolfese__2WikiMultihopQA/data/{split}-*.parquet"
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pat))], ignore_index=True)


def clean(s): return " ".join(str(s).split())


def ctx_docs(ctx):
    out = []
    for t, ss in zip(ctx["title"], ctx["sentences"]):
        ss = [clean(x) for x in ss if clean(x)]
        if ss: out.append((clean(t), ss))
    return out


def mkq(r, t2id, docs):
    gold = []
    for t, sid in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
        j = t2id.get(clean(t))
        if j is not None and int(sid) < len(docs[j]["sents"]): gold.append((j, int(sid)))
    return {"qid": r["id"], "question": clean(r["question"]), "answers": [clean(r["answer"])], "type": r["type"], "gold": sorted(set(gold))}


def build(name, n_eval=1000, n_tune=300, n_train=0, seed=0):
    dev = load(name, "validation")
    docs, t2id = [], {}
    for ctx in dev["context"]:
        for t, ss in ctx_docs(ctx):
            if t not in t2id: t2id[t] = len(docs); docs.append({"doc_id": len(docs), "title": t, "sents": ss})
    idx = list(range(len(dev))); random.Random(seed).shuffle(idx)
    qs = [mkq(dev.iloc[i], t2id, docs) for i in idx[:n_eval + n_tune]]
    ev, tu = [q for q in qs[:n_eval] if q["gold"]], [q for q in qs[n_eval:] if q["gold"]]
    dump({"docs": docs, "queries": ev}, f"{DATA}/{name}_eval.pkl")
    dump({"docs": docs, "queries": tu}, f"{DATA}/{name}_tune.pkl")
    print(name, "docs", len(docs), "sents", sum(len(d["sents"]) for d in docs), "eval", len(ev), "tune", len(tu), flush=True)
    if n_train:
        tr = load(name, "train"); tix = list(range(len(tr))); random.Random(seed + 1).shuffle(tix)
        train = []
        for i in tix[:n_train]:
            r = tr.iloc[i]; ld, lt = [], {}
            for t, ss in ctx_docs(r["context"]):
                if t not in lt: lt[t] = len(ld); ld.append({"doc_id": len(ld), "title": t, "sents": ss})
            q = mkq(r, lt, ld)
            if q["gold"]: q["docs"] = ld; train.append(q)
        dump(train, f"{DATA}/{name}_train.pkl"); print(name, "train", len(train))


if __name__ == "__main__":
    build("hotpotqa", n_train=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
    build("2wiki")
