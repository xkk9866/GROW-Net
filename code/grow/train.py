"""End-to-end training of GROW-Net on HotpotQA-train (every other dataset is zero-shot).
One network, one backward pass:
   L = L_suff  (set-level sufficiency, BCE to reader-grounded targets mixed with gold-evidence closure;
                trains slots, coverage AND the label-free dependency head, and the reader backbone)
     + L_list  (listwise ranking of supporting sentences among all candidate sentences)
     + a * L_cov (max_k pi_{k,i} vs supporting facts)
Usage: python train.py OUT [--no_cnn --no_set --no_xq --no_reader --no_dep --K 4 --no_suff --no_teacher --teacher_only
                            --no_list --train_layers 8]"""
import sys, os, glob, random, pickle, time, argparse, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from grow.common import *
from grow.model import GROWNet, log_suff

CE = f"{HF}/BAAI__bge-reranker-v2-m3"
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--K", type=int, default=4)
for f in ["crf", "no_cnn", "no_set", "no_xq", "no_reader", "no_dep", "no_suff", "no_teacher", "teacher_only", "no_list"]:
    ap.add_argument("--" + f, action="store_true")
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr_enc", type=float, default=1e-5); ap.add_argument("--lr_head", type=float, default=3e-4)
ap.add_argument("--n_train", type=int, default=None); ap.add_argument("--Lmax", type=int, default=24)
ap.add_argument("--a_cov", type=float, default=0.5); ap.add_argument("--mix", type=float, default=0.5)
ap.add_argument("--train_layers", type=int, default=8)
ap.add_argument("--nq", type=int, default=0)   # number of NQ-open training questions with reader-grounded distant supervision
args = ap.parse_args()
torch.manual_seed(0); random.seed(0)

train = pickle.load(open(f"{DATA}/hotpotqa_train.pkl", "rb"))[:args.n_train]
teacher = {}
for f in glob.glob(f"{DATA}/teacher_[0-9]*of*.pkl"): teacher.update(pickle.load(open(f, "rb")))
if args.nq:
    # reader-grounded distant supervision: an answer-bearing sentence is evidence iff the reader answers better with it
    # than with no context (y(single) > y(empty)); questions with no such sentence are dropped.
    nqtr = pickle.load(open(f"{DATA}/nq_train.pkl", "rb"))[:args.nq]; tn = {}
    for f in glob.glob(f"{DATA}/teacher_nq_*of*.pkl"): tn.update(pickle.load(open(f, "rb")))
    added = 0
    for k, q in enumerate(nqtr):
        t = tn.get(k)
        if t is None: continue
        G = sorted(set(map(tuple, q["gold"]))); y0 = t["empty"][1]
        keep = G if args.no_teacher else [g for idx, g in enumerate(G) if t.get(f"single_{idx}", ((), -1.0))[1] > y0]
        if not keep: continue
        q2 = dict(q); q2["gold_all"] = G; q2["gold"] = keep
        teacher[len(train)] = t; train.append(q2); added += 1
    print("NQ distant-supervision questions kept:", added, "of", len(nqtr), flush=True)
print("train q", len(train), "teacher q", len(teacher), flush=True)


def targets(k, q):
    """reader-grounded evidence-attributable sufficiency, mixed with gold-evidence closure (G subset of S)."""
    G = set(map(tuple, q["gold"]))
    t = teacher.get(k)
    if t is None:
        from grow.teacher import build_sets
        t = {n: (S, None) for n, S in build_sets(q, random.Random(k)).items()}
    y0 = t.get("empty", ((), 0.0))[1] or 0.0
    out = {}
    for n, (S, y) in t.items():
        if n == "empty": continue
        h = float(G.issubset(set(map(tuple, S))))
        if args.no_teacher or y is None: out[n] = (S, h)
        else:
            r = max(0.0, y - y0) / max(1e-3, 1 - y0)
            out[n] = (S, r if args.teacher_only else args.mix * r + (1 - args.mix) * h)
    return out


net = GROWNet(BGE, CE, K=args.K, train_layers=args.train_layers, use_cnn=not args.no_cnn, use_set=not args.no_set,
              use_xq=not args.no_xq, use_reader=not args.no_reader, use_dep=not args.no_dep).cuda()
net.ce.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
enc_params = [p for p in net.ce.parameters() if p.requires_grad]
head_params = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith(("ce.", "bi."))]
print("trainable reader params %.1fM, head params %.2fM" % (sum(p.numel() for p in enc_params) / 1e6, sum(p.numel() for p in head_params) / 1e6), flush=True)
opt = torch.optim.AdamW(([{"params": enc_params, "lr": args.lr_enc}] if enc_params else []) + [{"params": head_params, "lr": args.lr_head}], weight_decay=0.01)
steps = args.epochs * math.ceil(len(train) / args.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))


def crf_tags(k, q, D, L):
    """reader-grounded segmentation labels. A gold sentence absorbs its neighbour iff the reader is more likely to produce
    the answer with the neighbour than without it; the whole gold document is the unit iff the reader prefers it over the
    gold sentences with their neighbours. Only comparisons of reader likelihoods are used (no thresholds)."""
    G_all = sorted(set(map(tuple, q.get("gold_all", q["gold"])))); keep = set(map(tuple, q["gold"]))
    G = sorted(keep); t = {} if args.no_teacher else teacher.get(k, {})
    y = lambda n: t[n][1] if n in t else None
    sel = set(G)
    for idx, (j, i) in enumerate(G_all):
        if (j, i) not in keep: continue
        b = y(f"single_{idx}")
        if b is None: continue
        if y(f"singleL_{idx}") is not None and y(f"singleL_{idx}") > b: sel.add((j, i - 1))
        if y(f"singleR_{idx}") is not None and y(f"singleR_{idx}") > b: sel.add((j, i + 1))
    gd, gn = y("gold_docs"), y("gold_nbr")
    if gd is not None and gn is not None and gd > gn:
        sel |= {(j, i) for j in {j for j, _ in G} for i in range(len(q["docs"][j]["sents"]))}
    tags = torch.zeros(D, L, dtype=torch.long)
    for j in range(len(q["docs"])):
        for i in range(min(L, len(q["docs"][j]["sents"]))):
            if (j, i) in sel: tags[j, i] = 2 if (j, i - 1) in sel else 1
    return tags


def run():
    net.train(); net.bi.eval(); step = 0; t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(train))); random.shuffle(order)
        for s in range(0, len(order), args.bs):
            ks = order[s:s + args.bs]; qs = [train[k] for k in ks]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, M = net.forward_batch([q["question"] for q in qs], [q["docs"] for q in qs], QPREFIX, args.Lmax)
            B, D, L = M.shape
            gold = torch.zeros(B, D, L, device=M.device)
            for b, q in enumerate(qs):
                for j, i in q["gold"]:
                    if i < L: gold[b, j, i] = 1
            lg = out["pi_logit"].float().max(1).values
            pmax = torch.sigmoid(lg).clamp(1e-5, 1 - 1e-5)
            l_cov = F.binary_cross_entropy(pmax[M], gold[M], weight=(1 + 4 * gold[M]))
            loss = args.a_cov * l_cov
            l_list = torch.tensor(0.0, device=M.device)
            if not args.no_list:
                lgf = lg.masked_fill(~M, -1e4).reshape(B, -1); gl = gold.reshape(B, -1); hv = gl.sum(1) > 0
                l_list = -(torch.logsumexp(lgf.masked_fill(gl == 0, -1e4), 1) - torch.logsumexp(lgf, 1))[hv].mean()
                loss = loss + l_list
            l_suff = torch.tensor(0.0, device=M.device)
            if args.crf:
                tg = torch.stack([crf_tags(k, q, D, L) for k, q in zip(ks, qs)]).to(M.device)
                l_suff = net.crf.nll(net._h.reshape(B * D, L, -1).float(), tg.reshape(B * D, L), M.reshape(B * D, L))
                loss = loss + l_suff
            elif not args.no_suff:
                tg = [targets(k, q) for k, q in zip(ks, qs)]
                N = max(len(t) for t in tg)
                S = torch.zeros(B, N, D, L, device=M.device); Y = torch.zeros(B, N, device=M.device); V = torch.zeros(B, N, device=M.device)
                for b, t in enumerate(tg):
                    for n, (name, (sel, y)) in enumerate(t.items()):
                        for j, i in sel:
                            if i < L: S[b, n, j, i] = 1
                        Y[b, n] = y; V[b, n] = 1
                p = log_suff(out["pi"], out["w"], out["dL"], out["dR"], S).exp().clamp(1e-5, 1 - 1e-5)
                l_suff = (F.binary_cross_entropy(p, Y, reduction="none") * V).sum() / V.sum()
                loss = loss + l_suff
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} list {l_list.item():.4f} cov {l_cov.item():.4f} suff {l_suff.item():.4f} "
                      f"w {out['w'].mean().item():.2f} dL {out['dL'][M].mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
    sd = {k: v for k, v in net.state_dict().items() if not k.startswith("bi.")}
    torch.save(sd, f"{ROOT}/ckpt/{args.out}.pt")
    print("done", flush=True)


os.makedirs(f"{ROOT}/ckpt", exist_ok=True)
json.dump(vars(args), open(f"{ROOT}/ckpt/{args.out}.json", "w"))
run()
