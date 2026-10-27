"""Natural Questions on the BEIR-NQ Wikipedia corpus (2.68M passages).
Documents = consecutive BEIR passages sharing a title (Wikipedia article); passages = native BEIR passages;
sentences = regex split. Answers come from NQ-open (dev) matched to BEIR-NQ test queries by normalised question text."""
import json, gzip, os, sys, random, glob, collections
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *

def opener(p): return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")

def norm_q(q): return " ".join(normalize_answer(q).split())

base = f"{HF}/BeIR__nq"
def rows(kind):
    fs = glob.glob(f"{base}/**/{kind}*.jsonl*", recursive=True)
    if fs:
        for line in opener(fs[0]): yield json.loads(line)
    else:
        for f in sorted(glob.glob(f"{base}/{kind}/*.parquet")):
            df = pd.read_parquet(f)
            for r in df.to_dict("records"): yield r
# articles are cut into document units of <= SEG consecutive native passages (~300 words, LongRAG-style long units)
SEG = 3
docs, cur_title, cur = [], None, None
pid2loc = {}
if True:
    for r in rows("corpus"):
        t = str(r["title"]).strip(); txt = " ".join(r["text"].split())
        if t != cur_title or len(cur["pas"]) >= SEG:
            art = (cur["article"] + (1 if t != cur_title else 0)) if cur else 0
            cur = {"doc_id": len(docs), "title": t, "sents": [], "pas": [], "pids": [], "article": art}; docs.append(cur); cur_title = t
        ss = split_sents(txt); a = len(cur["sents"]); cur["sents"] += ss
        cur["pas"].append((a, len(cur["sents"]))); cur["pids"].append(str(r["_id"]))
        pid2loc[str(r["_id"])] = (cur["doc_id"], len(cur["pas"]) - 1)
print("docs", len(docs), "passages", len(pid2loc), "sents", sum(len(d["sents"]) for d in docs), flush=True)
queries = {}
for r in rows("queries"): queries[str(r["_id"])] = r["text"]
qrels = collections.defaultdict(list)
qr = glob.glob(f"{HF}/BeIR__nq-qrels/**/test.tsv", recursive=True) + glob.glob(f"{HF}/BeIR__nq-qrels/test.tsv")
for line in open(qr[0], encoding="utf-8").read().splitlines()[1:]:
    q, p, s = line.split("\t")
    if int(s) > 0 and p in pid2loc: qrels[q].append(pid2loc[p])
nqo = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{HF}/google-research-datasets__nq_open/**/validation*.parquet", recursive=True)])
ans = {norm_q(q): list(a) for q, a in zip(nqo["question"], nqo["answer"])}
allq = []
for qid, text in queries.items():
    a = ans.get(norm_q(text))
    if a and qrels.get(qid):
        allq.append({"qid": qid, "question": text.strip() + ("?" if not text.strip().endswith("?") else ""), "answers": [str(x) for x in a],
                     "type": "single", "gold_pas": sorted(set(qrels[qid])), "gold": []})
print("matched queries", len(allq), "of", len(queries))
random.Random(0).shuffle(allq)
dump({"docs": docs, "queries": allq[:1000]}, f"{DATA}/nq_eval.pkl")
dump({"docs": docs, "queries": allq[1000:1300]}, f"{DATA}/nq_tune.pkl")
