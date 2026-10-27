"""GROW-Net inference: single sentence index -> query-conditioned reading of K_d candidate documents -> structure /
sufficiency -> sufficiency-priced elastic widening.
python grow_infer.py CKPT DATASET[,DATASET...] [--split eval --theta 0.8 --budget 600 --K_docs 10 --tag T]
Writes RES/sel_{dataset}_{split}_{tag}.pkl (per-query selections as (doc, sent) lists, with sufficiency and slot gates)."""
import sys, os, argparse, time, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from grow.common import *
from grow.model import GROWNet
from grow.select_ew import widen

LMAX = 24
CE = f"{HF}/BAAI__bge-reranker-v2-m3"


def load_net(ckpt):
    cfg = json.load(open(f"{ROOT}/ckpt/{ckpt}.json"))
    net = GROWNet(BGE, CE, K=cfg["K"], train_layers=0, use_cnn=not cfg["no_cnn"], use_set=not cfg["no_set"], use_xq=not cfg["no_xq"],
                  use_reader=not cfg["no_reader"], use_dep=not cfg["no_dep"])
    sd = torch.load(f"{ROOT}/ckpt/{ckpt}.pt", map_location="cpu"); missing = net.load_state_dict(sd, strict=False)
    assert all(k.startswith("bi.") for k in missing.missing_keys), missing.missing_keys[:5]
    net.is_crf = cfg.get("crf", False)
    return net.cuda().eval(), cfg


def sentence_index(name, docs, enc=None):
    """the single index: bi-encoder vectors of every sentence (shared with the baselines' sentence level)."""
    U = Units(docs)
    return cached(f"{EMB}/bge_{name}_sent.npy", lambda: (enc or Encoder()).encode(U.texts("sent")).astype(np.float16)), U


@torch.no_grad()
def run(net, name, split="eval", K_docs=10, n_sent=200, thetas=(0.8,), lam=0.0, budget=600, eta=0.6, tag=None,
        use_closure=True, promote=True, return_scores=False, bs=8, hop2=False):
    cfg_crf = getattr(net, "is_crf", False)
    B = pickle.load(open(f"{DATA}/{name}_{split}.pkl", "rb")); docs, qs = B["docs"], B["queries"]
    Z, U = sentence_index(name, docs)
    qz = torch.cat([net.encode_query([q["question"] for q in qs[s:s + 256]], QPREFIX)[0] for s in range(0, len(qs), 256)])
    sv, si = dense_search(qz.cpu().numpy(), Z, n_sent)
    cands = []
    for qi in range(len(qs)):   # soft-max pooled document scores from the single sentence index
        dsc = {}
        for v, g in zip(sv[qi], si[qi]):
            d = int(U.sent_doc[g]); dsc[d] = dsc.get(d, 0.0) + np.exp((v - sv[qi][0]) / 0.05)
        cands.append(sorted(dsc, key=lambda d: -dsc[d])[:K_docs])
    if hop2 and cfg_crf:   # evidence-conditioned second hop: re-query with the first-pass evidence, read the union once more
        first = run(net, name, split, min(10, K_docs), n_sent, thetas=(1.0,), budget=budget, bs=bs)[1.0]
        q2 = [q["question"] + " " + " ".join(docs[d]["sents"][i] for d, i in r["sel"]) for q, r in zip(qs, first)]
        qz2 = torch.cat([net.encode_query(q2[s:s + 256], QPREFIX, max_len=256)[0] for s in range(0, len(q2), 256)])
        sv2, si2 = dense_search(qz2.cpu().numpy(), Z, n_sent)
        for qi in range(len(qs)):
            keep = list(dict.fromkeys([d for d, _ in first[qi]["sel"]]))
            dsc = {}
            for v, g in zip(sv2[qi], si2[qi]):
                d = int(U.sent_doc[g]); dsc[d] = dsc.get(d, 0.0) + np.exp((v - sv2[qi][0]) / 0.05)
            new = [d for d in sorted(dsc, key=lambda d: -dsc[d]) if d not in keep]
            old = [d for d in cands[qi] if d not in keep]
            merged = keep + [x for pair in zip(new, old) for x in pair]
            cands[qi] = list(dict.fromkeys(merged))[:K_docs]
    out = {th: [] for th in thetas}
    for s in range(0, len(qs), bs):
        qb = qs[s:s + bs]; cb = cands[s:s + bs]
        dl = [[{"title": docs[d]["title"], "sents": docs[d]["sents"][:LMAX]} for d in c] for c in cb]
        cache = [torch.from_numpy(np.asarray(Z[U.doc_start[d]:U.doc_start[d] + min(LMAX, len(docs[d]["sents"]))], dtype=np.float32)) for c in cb for d in c]
        with torch.autocast("cuda", dtype=torch.float16):
            o, M = net.forward_batch([q["question"] for q in qb], dl, QPREFIX, LMAX, bi_cache=cache)
        if cfg_crf:
            Bq, D_, L_ = M.shape
            hh = net._h.reshape(Bq * D_, L_, -1).float(); mm = M.reshape(Bq * D_, L_)
            if getattr(net, "indep", False):   # ablation: independent per-sentence decisions from the same emissions (no CRF structure)
                path = (net.crf.emit(hh).argmax(-1) * mm.long()).reshape(Bq, D_, L_).cpu().numpy()
            else:
                path = net.crf.viterbi(hh, mm).reshape(Bq, D_, L_).cpu().numpy()
            for b, (q, cand) in enumerate(zip(qb, cb)):
                sel = [(cand[j], int(i)) for j in range(len(cand)) for i in np.where(path[b, j] > 0)[0]]
                for th in thetas: out[th].append({"qid": q["qid"], "sel": sel})
            continue
        for b, (q, cand) in enumerate(zip(qb, cb)):
            D = len(cand); Mb = M[b, :D].cpu().numpy()
            pi, w = o["pi"][b][:, :D].cpu().numpy(), o["w"][b].cpu().numpy()
            dL, dR = o["dL"][b][:D].cpu().numpy(), o["dR"][b][:D].cpu().numpy()
            tok = np.zeros((D, LMAX)); ttok = np.zeros(D)
            for j, d in enumerate(cand):
                n = int(Mb[j].sum()); tok[j, :n] = ntok_batch(docs[d]["sents"][:n]); ttok[j] = ntok(docs[d]["title"]) + 3
            pas = [[(a, min(bb, LMAX)) for a, bb in doc_passages(docs[d]) if a < LMAX] for d in cand]
            for th in thetas:
                S, info = widen(pi, w, dL, dR, Mb, tok, ttok, theta=th, lam=lam, budget=budget, eta=eta, pas=pas,
                                use_closure=use_closure, promote=promote)
                sel = [(cand[j], int(i)) for j in range(D) for i in np.where(S[j])[0]]   # retrieval order, discourse order
                rec = {"qid": q["qid"], "sel": sel, "suff": info["suff"], "w": w.tolist()}
                if return_scores: rec.update({"cand": cand, "pi": pi, "w": w, "dL": dL, "dR": dR, "M": Mb, "tok": tok, "ttok": ttok})
                out[th].append(rec)
    if tag:
        for th in thetas: dump(out[th], f"{RES}/sel_{name}_{split}_{tag}_t{th}.pkl")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("datasets"); ap.add_argument("--split", default="eval")
    ap.add_argument("--theta", default="0.8"); ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--budget", type=int, default=600); ap.add_argument("--K_docs", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.6); ap.add_argument("--tag", default=None)
    ap.add_argument("--no_closure", action="store_true"); ap.add_argument("--no_promote", action="store_true")
    ap.add_argument("--save_scores", action="store_true"); ap.add_argument("--hop2", action="store_true"); ap.add_argument("--indep", action="store_true")
    a = ap.parse_args()
    net, _ = load_net(a.ckpt); net.indep = a.indep
    for name in a.datasets.split(","):
        t = time.time()
        run(net, name, a.split, a.K_docs, thetas=[float(x) for x in a.theta.split(",")], lam=a.lam, budget=a.budget, eta=a.eta,
            tag=a.tag or a.ckpt, use_closure=not a.no_closure, promote=not a.no_promote, return_scores=a.save_scores, hop2=a.hop2)
        print(name, "done", f"{time.time() - t:.0f}s", flush=True)
