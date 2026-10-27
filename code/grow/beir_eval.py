"""Zero-shot BEIR document ranking (nDCG@10) + evidence compression.
Rows: bge-base FirstP (document vectors), MaxP over passages / sentences, single-index soft-max pooling (GROW first stage),
bge-reranker-v2-m3 (rerank top-100), GROW-Net (rerank top-100 by document sufficiency Suff(d|q) of the same network),
and the fraction of each top document that GROW's elastic widening keeps as evidence.
python beir_eval.py CKPT scifact,nfcorpus,..."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from grow.common import *
from grow.grow_infer import load_net, LMAX, CE
from grow.model import log_suff
from grow.select_ew import widen


def ndcg10(ranked, rels):
    dcg = sum((2 ** rels.get(d, 0) - 1) / np.log2(r + 2) for r, d in enumerate(ranked[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def pool_docs(sv, si, owner, tau=0.05, mode="lse"):
    out = []
    for v, ids in zip(sv, si):
        sc = {}
        for x, g in zip(v, ids):
            d = int(owner[g])
            if mode == "max": sc[d] = max(sc.get(d, -1e9), float(x))
            else: sc[d] = sc.get(d, 0.0) + float(np.exp((x - v[0]) / tau))
        out.append(sorted(sc, key=lambda d: -sc[d]))
    return out


@torch.no_grad()
def grow_rerank(net, docs, U, Z, q, ranked, topn=100, chunk=10):
    cand = ranked[:topn]; scores = {}; ev = None
    for c0 in range(0, len(cand), chunk):
        cc = cand[c0:c0 + chunk]; D = len(cc)
        dl = [{"title": docs[d]["title"], "sents": docs[d]["sents"][:LMAX]} for d in cc]
        cache = [torch.from_numpy(np.asarray(Z[U.doc_start[d]:U.doc_start[d] + min(LMAX, len(docs[d]["sents"]))], dtype=np.float32)) for d in cc]
        with torch.autocast("cuda", dtype=torch.float16):
            o, M = net.forward_batch([q], [dl], QPREFIX, LMAX, bi_cache=cache)
        # document evidence score: CRF log-odds that the best sentence of d opens or continues an evidence span
        hh = net._h[0].float(); e = net.crf.emit(hh)
        lo = (torch.logsumexp(e[..., 1:], -1) - e[..., 0]).masked_fill(~M[0], -1e4)
        ls = lo.max(-1).values.cpu().numpy()
        for j, d in enumerate(cc): scores[d] = float(ls[j])
        if c0 == 0:   # evidence the CRF keeps from the top-ranked candidate document
            j = int(np.argmax(ls)); n = int(M[0, j].sum())
            path = net.crf.viterbi(hh[j:j + 1], M[0, j:j + 1])[0].cpu().numpy()
            tok = np.array(ntok_batch(docs[cc[j]]["sents"][:n]), dtype=float)
            ev = (tok[path[:n] > 0].sum(), tok.sum())
    return sorted(cand, key=lambda d: -scores[d]) + ranked[topn:], ev


def ce_rerank(tok, m, docs, q, ranked, topn=100):
    cand = ranked[:topn]
    texts = [span_text(docs[d], 0, len(docs[d]["sents"])) for d in cand]
    s = []
    with torch.no_grad():
        for b in range(0, len(texts), 32):
            enc = tok([[q, t] for t in texts[b:b + 32]], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            s += m(**enc).logits.squeeze(-1).float().cpu().tolist()
    return [cand[i] for i in np.argsort(-np.array(s))] + ranked[topn:]


if __name__ == "__main__":
    ckpt, names = sys.argv[1], sys.argv[2].split(",")
    enc = Encoder(); net, _ = load_net(ckpt)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    rtok = AutoTokenizer.from_pretrained(CE); rm = AutoModelForSequenceClassification.from_pretrained(CE, torch_dtype=torch.float16).cuda().eval()
    f = f"{RES}/beir_{ckpt}.json"; allres = json.load(open(f)) if os.path.exists(f) else {}
    for name in names:
        if name in allres: continue
        B = pickle.load(open(f"{DATA}/beir_{name}.pkl", "rb")); docs, qs = B["docs"], B["queries"]; U = Units(docs)
        E = {lv: cached(f"{EMB}/bge_beir_{name}_{lv}.npy", lambda lv=lv: enc.encode(U.texts(lv), max_len=512 if lv == "doc" else 256).astype(np.float16)) for lv in ("sent", "pas", "doc")}
        Q = enc.encode([q["question"] for q in qs], prefix=QPREFIX)
        R = {}
        v, i = dense_search(Q, E["doc"], 100); R["bge-FirstP(doc)"] = [list(map(int, x)) for x in i]
        v, i = dense_search(Q, E["pas"], 300); R["bge-MaxP(pas)"] = pool_docs(v, i, U.pas_doc, mode="max")
        v, i = dense_search(Q, E["sent"], 500); R["bge-MaxP(sent)"] = pool_docs(v, i, U.sent_doc, mode="max")
        R["single-index(lse)"] = pool_docs(v, i, U.sent_doc)
        R["bge-reranker-v2-m3"] = [ce_rerank(rtok, rm, docs, q["question"], r) for q, r in zip(qs, R["bge-FirstP(doc)"])]
        gr, ev = [], []
        for q, r in zip(qs, R["bge-FirstP(doc)"]):   # same candidate pool as the cross-encoder reranker
            x, e = grow_rerank(net, docs, U, E["sent"], q["question"], r); gr.append(x); ev.append(e)
        R["GROW-Net"] = gr
        res = {m: 100 * float(np.mean([ndcg10(r, q["rels"]) for r, q in zip(rs, qs)])) for m, rs in R.items()}
        ev = np.array([e for e in ev if e]); res["GROW_evidence_token_ratio"] = float(ev[:, 0].sum() / max(1, ev[:, 1].sum()))
        allres[name] = res; print(name, json.dumps(res), flush=True)
        json.dump(allres, open(f, "w"), indent=1)
