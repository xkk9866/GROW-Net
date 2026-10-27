"""GROW-Net inference: single sentence index -> candidate documents -> nested reader-odds field -> exact argmax
(= evidence and its granularity) -> re-reading: re-query the index with [q; decoded evidence] and decide once more on the
merged candidate set with the same field.
python infer_rof.py CKPT DATASETS [--split eval --deltas 0 --no_hop2 --tag T]   ->  RES/sel_{ds}_{split}_{tag}_d{delta}.pkl"""
import sys, os, argparse, time, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from grow.common import *
from grow.rof import ReaderOddsField, nested_argmax
from grow.grow_infer import sentence_index

LMAX = 24
CE = f"{HF}/BAAI__bge-reranker-v2-m3"


def load_rof(ckpt):
    cfg = json.load(open(f"{ROOT}/ckpt/{ckpt}.json"))
    net = ReaderOddsField(BGE, CE, train_layers=0, use_set=not cfg["no_set"], use_pair=not cfg["no_pair"], use_doc=not cfg["no_doc"])
    miss = net.load_state_dict(torch.load(f"{ROOT}/ckpt/{ckpt}.pt", map_location="cpu"), strict=False)
    assert all(k.startswith("bi.") for k in miss.missing_keys), miss.missing_keys[:5]
    net.sup = cfg["sup"]
    return net.cuda().eval()


def pool_docs(sv, si, U, K):
    dsc = {}
    for v, g in zip(sv, si):
        d = int(U.sent_doc[g]); dsc[d] = dsc.get(d, 0.0) + np.exp((v - sv[0]) / 0.05)
    return sorted(dsc, key=lambda d: -dsc[d])[:K]


@torch.no_grad()
def decide(net, qs, cands, docs, deltas=(0.0,), bs=8, keep=False):
    out = {d: [] for d in deltas}; pots = []
    for s in range(0, len(qs), bs):
        qb, cb = qs[s:s + bs], cands[s:s + bs]
        dl = [[{"title": docs[d]["title"], "sents": docs[d]["sents"][:LMAX]} for d in c] for c in cb]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b, c, M = net.forward_batch([q["question"] for q in qb], dl, LMAX)
        if getattr(net, "sup", "reader") == "relevance":   # ablation: relevance classifier, decision = its own logit > 0
            b = torch.zeros_like(b); c = torch.zeros_like(c)
        for dt in deltas:
            x = nested_argmax(a, b, c, M, dt)[0].cpu().numpy()
            for k, (q, cc) in enumerate(zip(qb, cb)):
                out[dt].append({"qid": q["qid"], "sel": [(cc[j], int(i)) for j in range(len(cc)) for i in np.where(x[k, j])[0]]})
        if keep:
            for k, cc in enumerate(cb):
                n = len(cc); pots.append({"a": a[k, :n].cpu(), "b": b[k, :n].cpu(), "c": c[k, :n].cpu(), "M": M[k, :n].cpu()})
    return out, pots


def run(net, qs, docs, Z, U, K=20, K1=10, n_sent=200, hop2=True, deltas=(0.0,), keep=False):
    qz = torch.cat([net.encode_query([q["question"] for q in qs[s:s + 256]], QPREFIX) for s in range(0, len(qs), 256)])
    sv, si = dense_search(qz.cpu().numpy(), Z, n_sent)
    cands = [pool_docs(sv[k], si[k], U, K) for k in range(len(qs))]
    if hop2:
        first = decide(net, qs, [c[:K1] for c in cands], docs)[0][0.0]
        q2 = [q["question"] + " " + " ".join(docs[d]["sents"][i] for d, i in r["sel"]) for q, r in zip(qs, first)]
        qz2 = torch.cat([net.encode_query(q2[s:s + 256], QPREFIX, max_len=256) for s in range(0, len(q2), 256)])
        sv2, si2 = dense_search(qz2.cpu().numpy(), Z, n_sent)
        for k in range(len(qs)):
            kept = list(dict.fromkeys(d for d, _ in first[k]["sel"]))
            new = [d for d in pool_docs(sv2[k], si2[k], U, 4 * K) if d not in kept]
            old = [d for d in cands[k] if d not in kept]
            cands[k] = list(dict.fromkeys(kept + [x for p in zip(new, old) for x in p]))[:K]
    else:
        cands = [c[:K1] for c in cands]
    out, pots = decide(net, qs, cands, docs, deltas, keep=keep)
    return out, cands, pots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("datasets"); ap.add_argument("--split", default="eval")
    ap.add_argument("--no_hop2", action="store_true"); ap.add_argument("--deltas", default="0"); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    net = load_rof(a.ckpt); deltas = [float(x) for x in a.deltas.split(",")]
    for name in a.datasets.split(","):
        t = time.time()
        B = pickle.load(open(f"{DATA}/{name}_{a.split}.pkl", "rb")); docs, qs = B["docs"], B["queries"]
        Z, U = sentence_index(name, docs)
        out, _, _ = run(net, qs, docs, Z, U, hop2=not a.no_hop2, deltas=deltas)
        for dt in deltas: dump(out[dt], f"{RES}/sel_{name}_{a.split}_{a.tag or a.ckpt}_d{dt:g}.pkl")
        print(name, "done", f"{time.time() - t:.0f}s", flush=True)
