"""Train GROW-Net (nested reader-odds field) on the reader's own answer likelihoods with the profiled likelihood.
Data: random-coalition probes (teacherab_*, teacherfull_*) and, optionally, decision-anchored probes
(data/probe_{TAG}_{src}.pkl written by probe_rof.py).
python train_rof.py OUT [--probes TAG1,TAG2] [--no_random] [--no_pair] [--no_set] [--no_doc] [--fixed_beta]
                        [--sup reader|relevance] [--train_layers 8] [--frac 1.0] [--seed 0]"""
import sys, os, random, pickle, time, argparse, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from grow.common import *
from grow.rof import ReaderOddsField, field_value, profile_bce

CE = f"{HF}/BAAI__bge-reranker-v2-m3"
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--probes", default="")
ap.add_argument("--target", default="lik", choices=["lik", "act"])   # act: F1 of the reader's greedy answer (probe_gen.py)
ap.add_argument("--no_random", action="store_true"); ap.add_argument("--no_pair", action="store_true")
ap.add_argument("--no_set", action="store_true"); ap.add_argument("--no_doc", action="store_true")
ap.add_argument("--fixed_beta", action="store_true")      # ablation: beta_q = logit y(empty) instead of profiling
ap.add_argument("--sup", default="reader", choices=["reader", "relevance"])
ap.add_argument("--train_layers", type=int, default=8); ap.add_argument("--frac", type=float, default=1.0)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--docs_per_step", type=int, default=48)
ap.add_argument("--lr_enc", type=float, default=1e-5); ap.add_argument("--lr_head", type=float, default=3e-4)
ap.add_argument("--Lmax", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)

# held out from every training run (fidelity / regret analyses)
HOLD = {("hotpotqa", "ab"): range(11500, 12000), ("hotpotqa", "full"): range(17500, 18000), ("nq", "ab"): range(3800, 4000), ("nq", "full"): range(3800, 4000)}
HOLD_PROBE = {"hotpotqa": range(19000, 20000), "nq": range(3800, 4000)}


def load():
    items = []
    if not args.no_random:
        QS = {s: pickle.load(open(f"{DATA}/{s}_train.pkl", "rb")) for s in ("hotpotqa", "nq")}
        held = {(s, k) for (s, _), r in HOLD.items() for k in r}
        for src in ("hotpotqa", "nq"):
            for des in ("ab", "full"):
                f = f"{DATA}/{'gen/' if args.target == 'act' else ''}teacher{des}_{src}.pkl"
                if not os.path.exists(f): continue
                for k, ms in pickle.load(open(f, "rb")).items():
                    if (src, k) in held: continue
                    q = QS[src][k]
                    items.append({"src": src, "question": q["question"], "docs": q["docs"], "gold": q["gold"], "sets": ms})
    for tag in [t for t in args.probes.split(",") if t]:
        for src in ("hotpotqa", "nq"):
            f = f"{DATA}/{'gen/' if args.target == 'act' else ''}probe_{tag}_{src}.pkl"
            if not os.path.exists(f): print("missing", f); continue
            for r in pickle.load(open(f, "rb")):
                if r["k"] in HOLD_PROBE[src]: continue
                items.append({"src": src, "question": r["question"], "docs": r["docs"], "gold": r.get("gold", []), "sets": r["sets"]})
    items = [it for it in items if len(it["sets"]) > 1]
    random.Random(args.seed).shuffle(items)
    return items[:int(len(items) * args.frac)]


train = load()
print("train questions", len(train), "sets/q %.1f" % (sum(len(t["sets"]) for t in train) / len(train)), flush=True)
net = ReaderOddsField(BGE, CE, train_layers=args.train_layers, use_set=not args.no_set, use_pair=not args.no_pair,
                      use_doc=not args.no_doc).cuda()
net.ce.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
enc_params = [p for p in net.ce.parameters() if p.requires_grad]
head_params = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith(("ce.", "bi."))]
opt = torch.optim.AdamW(([{"params": enc_params, "lr": args.lr_enc}] if enc_params else []) + [{"params": head_params, "lr": args.lr_head}], weight_decay=0.01)

# batches with a bounded number of documents (probe records carry up to 20 candidate documents)
batches, cur, nd = [], [], 0
for t in train:
    if cur and nd + len(t["docs"]) > args.docs_per_step: batches.append(cur); cur, nd = [], 0
    cur.append(t); nd += len(t["docs"])
if cur: batches.append(cur)
steps = args.epochs * len(batches)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))


def loss_reader(a, b, c, M, qs):
    B, D, L = M.shape; N = max(len(q["sets"]) for q in qs)
    X = torch.zeros(B, N, D, L, device=M.device); Y = torch.zeros(B, N, device=M.device); W = torch.zeros(B, N, device=M.device)
    for bb, q in enumerate(qs):
        for n, (S, y) in enumerate(q["sets"]):
            for j, i in S:
                if j < D and i < L: X[bb, n, j, i] = 1
            Y[bb, n] = y; W[bb, n] = 1
    phi = field_value(a, b, c, X * M[:, None].float())
    if args.fixed_beta:
        y0 = torch.tensor([([y for S, y in q["sets"] if not S] or [0.5])[0] for q in qs], device=M.device).clamp(1e-4, 1 - 1e-4)
        beta = torch.log(y0 / (1 - y0))
        l = F.binary_cross_entropy_with_logits(beta[:, None] + phi, Y.clamp(1e-4, 1 - 1e-4), reduction="none")
        return ((l * W).sum(1) / W.sum(1)).mean()
    return profile_bce(phi, Y, W)[0]


def loss_relevance(a, M, qs):
    gold = torch.zeros_like(a)
    for bb, q in enumerate(qs):
        for j, i in q["gold"]:
            if j < a.shape[1] and i < a.shape[2]: gold[bb, j, i] = 1
    return F.binary_cross_entropy_with_logits(a[M], gold[M])


net.train(); net.bi.eval(); t0 = time.time(); step = 0
for ep in range(args.epochs):
    random.shuffle(batches)
    for qs in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b, c, M = net.forward_batch([q["question"] for q in qs], [q["docs"] for q in qs], args.Lmax)
            loss = loss_reader(a, b, c, M, qs) if args.sup == "reader" else loss_relevance(a, M, qs)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step(); sched.step(); step += 1
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} a+ {(a[M] > 0).float().mean().item():.3f} b {b[M].mean().item():+.3f} "
                  f"c {c[M.any(-1)].mean().item():+.2f} {time.time() - t0:.0f}s", flush=True)
os.makedirs(f"{ROOT}/ckpt", exist_ok=True)
torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("bi.")}, f"{ROOT}/ckpt/{args.out}.pt")
json.dump(vars(args), open(f"{ROOT}/ckpt/{args.out}.json", "w"))
print("done", flush=True)
