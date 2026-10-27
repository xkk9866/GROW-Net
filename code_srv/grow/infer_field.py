"""GROW-Net (field form) inference.
single sentence index -> K candidate documents -> reader log-odds field -> exact argmax (= evidence + granularity)
-> (re-reading) re-query the same index with [q; decoded evidence], read the merged candidate set once more.
python infer_field.py CKPT DATASETS [--split eval --K_docs 20 --hop2 --deltas 0 --unary --tag T --save_scores]
Writes RES/sel_{dataset}_{split}_{tag}_d{delta}.pkl"""
import sys, os, argparse, time, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from grow.common import *
from grow.field import GROWField, exact_argmax, bayes_argmax
from grow.grow_infer import sentence_index

LMAX = 24
CE = f"{HF}/BAAI__bge-reranker-v2-m3"


def load_field(ckpt):
    cfg = json.load(open(f"{ROOT}/ckpt/{ckpt}.json"))
    net = GROWField(BGE, CE, train_layers=0, use_set=not cfg["no_set"], use_pair=not cfg["no_pair"] and cfg["sup"] == "reader",
                    use_crf=cfg["sup"] == "crf", use_var=cfg.get("use_var", False))
    miss = net.load_state_dict(torch.load(f"{ROOT}/ckpt/{ckpt}.pt", map_location="cpu"), strict=False)
    assert all(k.startswith("bi.") for k in miss.missing_keys), miss.missing_keys[:5]
    return net.cuda().eval(), cfg


def pool_docs(sv, si, U, K):
    dsc = {}
    for v, g in zip(sv, si):
        d = int(U.sent_doc[g]); dsc[d] = dsc.get(d, 0.0) + np.exp((v - sv[0]) / 0.05)
    return sorted(dsc, key=lambda d: -dsc[d])[:K]


@torch.no_grad()
def decode(net, qs, cands, docs, deltas, unary=False, bs=8, keep_scores=False, decision="bayes"):
    out = {d: [] for d in deltas}; scores = []
    for s in range(0, len(qs), bs):
        qb, cb = qs[s:s + bs], cands[s:s + bs]
        dl = [[{"title": docs[d]["title"], "sents": docs[d]["sents"][:LMAX]} for d in c] for c in cb]
        with torch.autocast("cuda", dtype=torch.float16):
            a, b, M = net.forward_batch([q["question"] for q in qb], dl, LMAX)
        if unary: b = torch.zeros_like(b)
        B_, D_, L_ = M.shape
        if hasattr(net, "crf"):   # ablation: Viterbi (most-likely labelling) instead of the utility argmax
            p = net.crf.viterbi(net._h.reshape(B_ * D_, L_, -1).float(), M.reshape(-1, L_)).reshape(B_, D_, L_).cpu().numpy()
            for dt in deltas:
                for k, (q, c) in enumerate(zip(qb, cb)):
                    out[dt].append({"qid": q["qid"], "sel": [(c[j], int(i)) for j in range(len(c)) for i in np.where(p[k, j] > 0)[0]]})
            continue
        bayes = decision == "bayes" and net.use_var
        if bayes:
            with torch.autocast("cuda", dtype=torch.float16):
                beta = net.closed_book([q["question"] for q in qb])
        for dt in deltas:
            if bayes:   # expected-utility decision under the field's own uncertainty
                x = bayes_argmax(a - dt, b, net._v, M, beta.float()).cpu().numpy()
            else:       # argmax of the mean field
                x = exact_argmax(a.reshape(-1, L_), b.reshape(-1, L_), M.reshape(-1, L_), dt).reshape(B_, D_, L_).cpu().numpy()
            for k, (q, c) in enumerate(zip(qb, cb)):
                out[dt].append({"qid": q["qid"], "sel": [(c[j], int(i)) for j in range(len(c)) for i in np.where(x[k, j])[0]]})
        if keep_scores:
            for k, (q, c) in enumerate(zip(qb, cb)):
                n = len(c); scores.append({"qid": q["qid"], "cand": c, "a": a[k, :n].cpu().numpy(), "b": b[k, :n].cpu().numpy(), "M": M[k, :n].cpu().numpy()})
    return out, scores


def run(net, name, split="eval", K_docs=20, K1=10, n_sent=200, hop2=True, deltas=(0.0,), unary=False, tag=None, keep_scores=False, decision="bayes"):
    B = pickle.load(open(f"{DATA}/{name}_{split}.pkl", "rb")); docs, qs = B["docs"], B["queries"]
    Z, U = sentence_index(name, docs)
    qz = torch.cat([net.encode_query([q["question"] for q in qs[s:s + 256]], QPREFIX) for s in range(0, len(qs), 256)])
    sv, si = dense_search(qz.cpu().numpy(), Z, n_sent)
    cands = [pool_docs(sv[k], si[k], U, K_docs) for k in range(len(qs))]
    if hop2:   # re-reading: the decoded evidence (self-contained text) is the query refinement
        first, _ = decode(net, qs, [c[:K1] for c in cands], docs, (0.0,), unary, decision=decision)
        first = first[0.0]
        q2 = [q["question"] + " " + " ".join(docs[d]["sents"][i] for d, i in r["sel"]) for q, r in zip(qs, first)]
        qz2 = torch.cat([net.encode_query(q2[s:s + 256], QPREFIX, max_len=256) for s in range(0, len(q2), 256)])
        sv2, si2 = dense_search(qz2.cpu().numpy(), Z, n_sent)
        for k in range(len(qs)):
            keep = list(dict.fromkeys(d for d, _ in first[k]["sel"]))
            new = [d for d in pool_docs(sv2[k], si2[k], U, 4 * K_docs) if d not in keep]
            old = [d for d in cands[k] if d not in keep]
            cands[k] = list(dict.fromkeys(keep + [x for p in zip(new, old) for x in p]))[:K_docs]
    else:
        cands = [c[:K1] if K1 else c for c in cands]
    out, scores = decode(net, qs, cands, docs, deltas, unary, keep_scores=keep_scores, decision=decision)
    if tag:
        for dt in deltas: dump(out[dt], f"{RES}/sel_{name}_{split}_{tag}_d{dt:g}.pkl")
        if keep_scores: dump(scores, f"{RES}/scores_{name}_{split}_{tag}.pkl")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("datasets"); ap.add_argument("--split", default="eval")
    ap.add_argument("--K_docs", type=int, default=20); ap.add_argument("--K1", type=int, default=10)
    ap.add_argument("--no_hop2", action="store_true"); ap.add_argument("--deltas", default="0")
    ap.add_argument("--unary", action="store_true"); ap.add_argument("--tag", default=None); ap.add_argument("--save_scores", action="store_true")
    ap.add_argument("--decision", default="bayes", choices=["bayes", "mean"])
    a = ap.parse_args()
    net, _ = load_field(a.ckpt)
    for name in a.datasets.split(","):
        t = time.time()
        run(net, name, a.split, a.K_docs, a.K1, hop2=not a.no_hop2, deltas=[float(x) for x in a.deltas.split(",")],
            unary=a.unary, tag=a.tag or a.ckpt, keep_scores=a.save_scores, decision=a.decision)
        print(name, "done", f"{time.time() - t:.0f}s", flush=True)
