"""Baselines. Every method returns, per query, a provenance-preserving selection = list of (doc, sentence) so that
evidence recall, token cost and reader prompts are computed identically for all methods.
  fixed granularity : sent@k, pas@k, doc@k                        (vanilla bge-base-en-v1.5 dense retrieval)
  hierarchical      : sentence-window (small-to-big), auto-merging, coarse-to-fine (doc -> sentence)
  multi-granularity : Mix-of-Granularity router (COLING'25), re-implemented with dense retrieval
  rerank / prune    : bge-reranker-v2-m3 passage reranking; Provence (ICLR'25) context pruning
python baselines.py DATASET[,..] [--split eval] [--only m1,m2]"""
import sys, os, argparse, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from grow.common import *


class Ctx:
    def __init__(self, name, split, enc):
        B = pickle.load(open(f"{DATA}/{name}_{split}.pkl", "rb"))
        self.name, self.docs, self.qs, self.enc = name, B["docs"], B["queries"], enc
        self.U = Units(self.docs)
        mk = lambda lv: cached(f"{EMB}/bge_{name}_{lv}.npy", lambda: enc.encode(self.U.texts(lv), max_len=512 if lv == "doc" else 256).astype(np.float16))
        self.E = {lv: mk(lv) for lv in ("sent", "pas", "doc")}
        self.q = enc.encode([q["question"] for q in self.qs], prefix=QPREFIX)
        self.hits = {lv: dense_search(self.q, self.E[lv], 100) for lv in ("sent", "pas", "doc")}

    def sent(self, g): return (int(self.U.sent_doc[g]), int(self.U.sent_idx[g]))
    def pas(self, p):
        d = int(self.U.pas_doc[p]); a, b = self.U.pas_span[p]; return [(d, i) for i in range(a, b)]
    def doc(self, d): return [(int(d), i) for i in range(len(self.docs[d]["sents"]))]


def order_sel(sel):
    """dedupe, keep first-seen document order, sentence order inside a document."""
    seen, docs = set(), []
    for d, i in sel:
        if (d, i) in seen: continue
        seen.add((d, i))
        if d not in docs: docs.append(d)
    return [(d, i) for d in docs for i in sorted(i for dd, i in seen if dd == d)]


def fixed(c, lv, k):
    f = {"sent": lambda x: [c.sent(x)], "pas": c.pas, "doc": c.doc}[lv]
    return [order_sel(sum((f(x) for x in c.hits[lv][1][qi][:k]), [])) for qi in range(len(c.qs))]


def sent_window(c, k=5, w=1):
    out = []
    for qi in range(len(c.qs)):
        sel = []
        for g in c.hits["sent"][1][qi][:k]:
            d, i = c.sent(g); n = len(c.docs[d]["sents"])
            sel += [(d, j) for j in range(max(0, i - w), min(n, i + w + 1))]
        out.append(order_sel(sel))
    return out


def auto_merge(c, k=8, ratio=0.5):
    out = []
    for qi in range(len(c.qs)):
        sel = [c.sent(g) for g in c.hits["sent"][1][qi][:k]]
        S = set(sel)
        for p in set(int(c.U.sent_pas[c.U.gid(d, i)]) for d, i in sel):           # merge to parent passage
            units = c.pas(p)
            if sum(u in S for u in units) >= ratio * len(units): S |= set(units)
        for d in set(d for d, _ in S):                                              # merge to parent document
            units = c.doc(d)
            if sum(u in S for u in units) >= ratio * len(units): S |= set(units)
        out.append(order_sel([x for x in sel] + sorted(S)))
    return out


def coarse_to_fine(c, kd=5, ks=6):
    out = []
    for qi in range(len(c.qs)):
        top = c.hits["doc"][1][qi][:kd]
        gids = np.concatenate([np.arange(c.U.doc_start[d], c.U.doc_start[d + 1]) for d in top])
        sc = c.E["sent"][gids].astype(np.float32) @ c.q[qi]
        out.append(order_sel([c.sent(g) for g in gids[np.argsort(-sc)[:ks]]]))
    return out


# ------------------------------------------------------------------ Mix-of-Granularity (router over granularity levels)
class MoGRouter(torch.nn.Module):
    def __init__(self, h=768, L=3):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(h, 256), torch.nn.ReLU(), torch.nn.Linear(256, L))
    def forward(self, x): return torch.sigmoid(self.net(x))


def train_mog(enc, n=8000):
    f = f"{ROOT}/ckpt/mog_router.pt"
    r = MoGRouter()
    if os.path.exists(f): r.load_state_dict(torch.load(f)); return r
    tr = pickle.load(open(f"{DATA}/hotpotqa_train.pkl", "rb"))[:n]
    Q = enc.encode([q["question"] for q in tr], prefix=QPREFIX)
    labels = []
    for k, q in enumerate(tr):
        U = Units(q["docs"]); gold_txt = " ".join(sent_text(q["docs"][j], i) for j, i in q["gold"])
        g = enc.encode([gold_txt])[0]
        sims = []
        for lv in ("sent", "pas", "doc"):
            T = U.texts(lv); E = enc.encode(T); top = int(np.argmax(E @ Q[k]))       # top-1 unit at this level
            sims.append(float(E[top] @ g))                                             # similarity to reference snippet
        o = np.argsort(sims)[::-1]; y = np.zeros(3); y[o[0]] = 0.8; y[o[1]] = 0.2; labels.append(y)
    X = torch.tensor(Q); Y = torch.tensor(np.array(labels), dtype=torch.float32)
    opt = torch.optim.Adam(r.parameters(), 1e-3)
    for ep in range(30):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 256):
            b = perm[s:s + 256]; loss = torch.nn.functional.binary_cross_entropy(r(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(r.state_dict(), f); print("MoG router trained, label dist", Y.argmax(1).bincount(minlength=3).tolist(), flush=True)
    return r


def mog(c, router, kr=3, m=3):
    W = router(torch.tensor(c.q)).detach().numpy()
    out = []
    for qi in range(len(c.qs)):
        w = W[qi]; lv_scores = []
        for lv in ("sent", "pas", "doc"):
            v, ids = c.hits[lv][0][qi][:kr], c.hits[lv][1][qi][:kr]; lv_scores.append(dict(zip(ids.tolist(), v.tolist())))
        cands = set(c.hits["sent"][1][qi][:kr].tolist())
        for p in lv_scores[1]: cands |= set(c.U.gid(d, i) for d, i in c.pas(p))
        for d in lv_scores[2]: cands |= set(range(c.U.doc_start[d], c.U.doc_start[d + 1]))
        def trs(g):
            return np.array([lv_scores[0].get(g, 0.0), lv_scores[1].get(int(c.U.sent_pas[g]), 0.0), lv_scores[2].get(int(c.U.sent_doc[g]), 0.0)])
        ranked = sorted(cands, key=lambda g: -float(trs(g) @ w))
        gr = int(np.argmax(w)); sel, used = [], set()
        for g in ranked:
            unit = [c.sent(g)] if gr == 0 else (c.pas(int(c.U.sent_pas[g])) if gr == 1 else c.doc(int(c.U.sent_doc[g])))
            key = tuple(unit)
            if key in used: continue
            used.add(key); sel += unit
            if len(used) >= m: break
        out.append(order_sel(sel))
    return out


# ------------------------------------------------------------------ cross-encoder reranking and Provence pruning
_CE = {}


def ce_scores(question_texts):
    """bge-reranker-v2-m3 relevance logits for (question, passage text) pairs."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not _CE:
        p = f"{HF}/BAAI__bge-reranker-v2-m3"
        _CE["tok"] = AutoTokenizer.from_pretrained(p)
        _CE["m"] = AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.float16).cuda().eval()
    out = []
    for s in range(0, len(question_texts), 64):
        with torch.no_grad():
            enc = _CE["tok"](question_texts[s:s + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            out.append(_CE["m"](**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def pas_text(c, p):
    return f"{c.docs[int(c.U.pas_doc[p])]['title']}: " + " ".join(c.docs[d]["sents"][i] for d, i in c.pas(p))


def rerank_order(c, k_in=20):
    """per query: top-k_in dense passages sorted by cross-encoder score (cached, shared by all rerank@k)."""
    if not hasattr(c, "_rr"):
        c._rr = []
        for qi, q in enumerate(c.qs):
            ps = list(c.hits["pas"][1][qi][:k_in]); s = ce_scores([[q["question"], pas_text(c, p)] for p in ps])
            c._rr.append(([ps[j] for j in np.argsort(-s)], np.sort(s)[::-1]))
    return c._rr


def rerank_passages(c, k_in=20, k_out=3):
    return [order_sel(sum((c.pas(p) for p in ps[:k_out]), [])) for ps, _ in rerank_order(c, k_in)]


def rerank_2hop(c, k_in=20, k_out=3):
    """iterative reranking (same two retrieval rounds as GROW-Net's re-reading): the top reranked passage is appended to
    the question, the passage index is searched again, and the union of both candidate pools is reranked; a passage's
    score is its best score under the original or the expanded question."""
    first = rerank_order(c, k_in)
    q2 = [q["question"] + " " + pas_text(c, ps[0]) for q, (ps, _) in zip(c.qs, first)]
    z2 = c.enc.encode(q2, prefix=QPREFIX, max_len=256)
    _, h2 = dense_search(z2, c.E["pas"], k_in)
    out = []
    for qi, q in enumerate(c.qs):
        ps1, s1 = first[qi]; sc = dict(zip(ps1, s1))
        new = [p for p in h2[qi] if p not in sc]
        s_new = ce_scores([[q["question"], pas_text(c, p)] for p in new]) if new else []
        for p, v in zip(new, s_new): sc[p] = float(v)
        cand = list(sc); s_exp = ce_scores([[q2[qi], pas_text(c, p)] for p in cand])
        best = {p: max(sc[p], float(v)) for p, v in zip(cand, s_exp)}
        top = sorted(best, key=lambda p: -best[p])[:k_out]
        out.append(order_sel(sum((c.pas(p) for p in top), [])))
    return out


def provence(c, k=5, threshold=0.1):
    import nltk, nltk.tokenize  # Provence sentence-splits with nltk; use our splitter (no punkt download needed)
    nltk.sent_tokenize = nltk.tokenize.sent_tokenize = lambda t, language="english": split_sents(t, min_chars=1)
    from transformers import AutoModel
    m = AutoModel.from_pretrained(f"{HF}/naver__provence-reranker-debertav3-v1", trust_remote_code=True).cuda().eval()
    out = []
    for qi, q in enumerate(c.qs):
        ps = c.hits["pas"][1][qi][:k]; sel, scored = [], []
        for p in ps:
            units = c.pas(p); d = units[0][0]
            ctx = " ".join(c.docs[d]["sents"][i] for _, i in units)
            try:
                r = m.process(q["question"], ctx, title=c.docs[d]["title"], threshold=threshold, always_select_title=True,
                              enable_warnings=False, reorder=False)
            except Exception:   # degenerate passages (e.g. empty after tokenisation): keep unpruned
                scored.append((-1e9, units)); continue
            pr = r["pruned_context"]; pr = pr[0] if isinstance(pr, list) else pr
            pr = normalize_answer(pr if isinstance(pr, str) else str(pr))
            kept = [(d, i) for _, i in units if normalize_answer(c.docs[d]["sents"][i])[:60] in pr]
            sc = r.get("reranking_score", 0); sc = sc[0] if isinstance(sc, list) else sc
            scored.append((float(sc), kept))
        for _, kept in sorted(scored, key=lambda x: -x[0]): sel += kept
        out.append(order_sel(sel))
    return out


METHODS = {
    "sent@5": lambda c: fixed(c, "sent", 5), "sent@10": lambda c: fixed(c, "sent", 10),
    "pas@3": lambda c: fixed(c, "pas", 3), "pas@5": lambda c: fixed(c, "pas", 5),
    "doc@3": lambda c: fixed(c, "doc", 3), "doc@5": lambda c: fixed(c, "doc", 5),
    "sentwin": sent_window, "automerge": auto_merge, "c2f": coarse_to_fine,
    "rerank@3": rerank_passages, "provence": provence,
    "rerank@1": lambda c: rerank_passages(c, k_out=1), "rerank@2": lambda c: rerank_passages(c, k_out=2),
    "rerank@5": lambda c: rerank_passages(c, k_out=5), "rerank2hop@3": rerank_2hop,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("datasets"); ap.add_argument("--split", default="eval"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    enc = Encoder()
    for name in a.datasets.split(","):
        c = Ctx(name, a.split, enc)
        ms = a.only.split(",") if a.only else list(METHODS) + ["mog"]
        for mname in ms:
            f = f"{RES}/sel_{name}_{a.split}_{mname}.pkl"
            if os.path.exists(f): continue
            t = time.time()
            sel = mog(c, train_mog(enc)) if mname == "mog" else METHODS[mname](c)
            dump([{"qid": q["qid"], "sel": s} for q, s in zip(c.qs, sel)], f)
            print(name, mname, f"{time.time() - t:.0f}s", flush=True)
