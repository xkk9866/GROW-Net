"""BEIR zero-shot retrieval sets: documents (title + text) split into sentences; passages = 3-sentence windows."""
import json, gzip, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grow.common import *
def opener(p): return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
for b in sys.argv[1:] or ["scifact", "nfcorpus", "fiqa", "scidocs", "arguana", "trec-covid"]:
    base = f"{HF}/BeIR__{b}"
    def rows(kind):
        fs = glob.glob(f"{base}/**/{kind}*.jsonl*", recursive=True)
        if fs:
            for line in opener(fs[0]): yield json.loads(line)
        else:
            import pandas as pd
            for f in sorted(glob.glob(f"{base}/{kind}/*.parquet")): yield from pd.read_parquet(f).to_dict("records")
    docs, did = [], {}
    for r in rows("corpus"):
        r["_id"] = str(r["_id"]); txt = " ".join(r["text"].split())
        if not txt: continue
        did[r["_id"]] = len(docs); docs.append({"doc_id": len(docs), "title": " ".join(r.get("title", "").split()) or "Document", "sents": split_sents(txt), "orig": r["_id"]})
    qtext = {str(r["_id"]): r["text"] for r in rows("queries")}
    qrels = collections.defaultdict(dict)
    split = "test"
    qr = glob.glob(f"{HF}/BeIR__{b}-qrels/**/{split}.tsv", recursive=True) + glob.glob(f"{HF}/BeIR__{b}-qrels/{split}.tsv")
    for line in open(qr[0], encoding="utf-8").read().splitlines()[1:]:
        q, p, s = line.split("\t")
        if p in did: qrels[q][did[p]] = int(s)
    qs = [{"qid": q, "question": qtext[q], "rels": r} for q, r in qrels.items() if q in qtext]
    dump({"docs": docs, "queries": qs}, f"{DATA}/beir_{b}.pkl")
    print(b, "docs", len(docs), "queries", len(qs), "sents", sum(len(d["sents"]) for d in docs), flush=True)
