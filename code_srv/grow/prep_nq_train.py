"""Reader-grounded distant supervision for single-hop QA without evidence annotation.
NQ-open *train* questions are retrieved against the BEIR-NQ corpus with the same single sentence index used at test time;
the 10 pooled candidate documents become the reading context, and sentences containing a gold answer string are candidate
anchors. Whether an anchor is real evidence (and how far its span extends) is decided later by the int4 reader
(teacher.py / train.py), using likelihood comparisons only.
python prep_nq_train.py N"""
import sys, os, glob, random, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from grow.common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
LMAX, K = 24, 10
B = pickle.load(open(f"{DATA}/nq_eval.pkl", "rb")); docs = B["docs"]
held = {normalize_answer(q["question"]) for q in B["queries"]}
held |= {normalize_answer(q["question"]) for q in pickle.load(open(f"{DATA}/nq_tune.pkl", "rb"))["queries"]}
df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{HF}/google-research-datasets__nq_open/**/train*.parquet", recursive=True)])
rows = [(q, list(a)) for q, a in zip(df["question"], df["answer"]) if normalize_answer(q) not in held]
random.Random(7).shuffle(rows); rows = rows[:N]
U = Units(docs)
Z = np.load(f"{EMB}/bge_nq_sent.npy", mmap_mode="r")
enc = Encoder()
qtext = [q.strip() + ("?" if not q.strip().endswith("?") else "") for q, _ in rows]
qv = enc.encode(qtext, prefix=QPREFIX)
sv, si = dense_search(qv, Z, 200)
out = []
for k, ((_, ans), qt) in enumerate(zip(rows, qtext)):
    dsc = {}
    for v, g in zip(sv[k], si[k]):
        d = int(U.sent_doc[g]); dsc[d] = dsc.get(d, 0.0) + np.exp((v - sv[k][0]) / 0.05)
    cand = sorted(dsc, key=lambda d: -dsc[d])[:K]
    ld = [{"doc_id": j, "title": docs[d]["title"], "sents": docs[d]["sents"][:LMAX]} for j, d in enumerate(cand)]
    anchors = [(j, i) for j, d in enumerate(ld) for i, s in enumerate(d["sents"]) if has_answer(s, ans)]
    if anchors:
        out.append({"qid": f"nqtr{k}", "question": qt, "answers": [str(a) for a in ans], "type": "single", "src": "nq",
                    "gold": anchors, "docs": ld})
dump(out, f"{DATA}/nq_train.pkl")
print("nq train questions with answer-bearing candidates:", len(out), "of", len(rows),
      "anchors/q %.2f" % np.mean([len(q["gold"]) for q in out]), flush=True)
