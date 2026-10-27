"""Train GROW-Net by fitting the reader's own answer likelihoods.
For every probed training question we have reader measurements y(S) on random evidence coalitions S and the closed-book
value y(0). The reader's answer log-odds are modelled as Gaussian with mean beta_q + Phi(S) and variance V(S) = sum v_i x_i,
so its expected answer probability is sigma((beta_q + Phi(S)) / sqrt(1 + pi V(S) / 8)). The only objective is the
likelihood of the measurements under this model (the S = 0 term trains the closed-book head):
    L = sum_{S != 0} BCE( sigma((beta_q + Phi(S)) / sqrt(1 + c V(S))), y(S) )  +  BCE( sigma(beta_hat(q)), y(0) ).
No relevance labels, target segmentations or auxiliary losses. Teacher: Qwen3-8B-AWQ (int4).
Probe designs: 'ab'   = Banzhaf coalitions of a small universe (gold neighbourhood + one distractor), teacher_ab.py
               'full' = Shapley coalitions of ALL candidate sentences, teacher_full.py
python train_field.py OUT [--data both|ab|full] [--no_var] [--sup reader|relevance|crf] [--link logistic|linear]
                          [--no_pair] [--no_set] [--train_layers 8] [--frac 1.0] [--seed 0]"""
import sys, os, glob, random, pickle, time, argparse, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from grow.common import *
from grow.field import GROWField, field_value, expected_logit

CE = f"{HF}/BAAI__bge-reranker-v2-m3"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--data", default="both", choices=["both", "ab", "full"]); ap.add_argument("--no_var", action="store_true")
ap.add_argument("--sup", default="reader", choices=["reader", "relevance", "crf"]); ap.add_argument("--link", default="logistic", choices=["logistic", "linear"])
ap.add_argument("--no_pair", action="store_true"); ap.add_argument("--no_set", action="store_true")
ap.add_argument("--train_layers", type=int, default=8); ap.add_argument("--frac", type=float, default=1.0)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr_enc", type=float, default=1e-5); ap.add_argument("--lr_head", type=float, default=3e-4)
ap.add_argument("--Lmax", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
DEV = os.environ.get("GROW_DEV", "cuda")
torch.manual_seed(args.seed); random.seed(args.seed)
USE_VAR = not args.no_var and args.sup == "reader" and args.link == "logistic"

# fidelity hold-out (fidelity.py): never trained on by any variant
HOLD = {("hotpotqa", "ab"): range(11500, 12000), ("hotpotqa", "full"): range(17500, 18000), ("nq", "ab"): range(3800, 4000), ("nq", "full"): range(3800, 4000)}


def load_measurements():
    """(src, k) -> list of (S, y) over all probe designs, plus y(0)."""
    designs = ["ab", "full"] if args.data == "both" else [args.data]
    out = {}
    for src in ("hotpotqa", "nq"):
        for des in designs:
            f = f"{DATA}/teacher{des}_{src}.pkl"
            if not os.path.exists(f): print("missing", f, flush=True); continue
            T = pickle.load(open(f, "rb")); held = set(HOLD[(src, des)])
            for k, ms in T.items():
                if k in held: continue
                e = out.setdefault((src, k), {"sets": [], "y0": None})
                for S, y in ms:
                    if S: e["sets"].append((tuple(S), y))
                    else: e["y0"] = y
    return out


Mx = load_measurements()
QS = {src: pickle.load(open(f"{DATA}/{src}_train.pkl", "rb")) for src in ("hotpotqa", "nq")}
held_all = {(s, k) for (s, _), r in HOLD.items() for k in r}
train = []
for (src, k), e in sorted(Mx.items()):
    if (src, k) in held_all or not e["sets"] or e["y0"] is None: continue
    q = dict(QS[src][k]); q["src"], q["sets"], q["y0"] = src, e["sets"], e["y0"]; train.append(q)
random.Random(args.seed).shuffle(train); train = train[:int(len(train) * args.frac)]
for src in ("hotpotqa", "nq"):
    t = [q for q in train if q["src"] == src]
    print(src, "questions", len(t), "measurements/q %.1f" % (sum(len(q["sets"]) for q in t) / max(1, len(t))), flush=True)
print("train questions", len(train), flush=True)

net = GROWField(BGE, CE, train_layers=args.train_layers, use_set=not args.no_set, use_pair=not args.no_pair and args.sup == "reader",
                use_crf=args.sup == "crf", use_var=USE_VAR).to(DEV)
net.ce.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
enc_params = [p for p in net.ce.parameters() if p.requires_grad]
head_params = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith(("ce.", "bi."))]
print("trainable reader %.1fM head %.2fM" % (sum(p.numel() for p in enc_params) / 1e6, sum(p.numel() for p in head_params) / 1e6), flush=True)
opt = torch.optim.AdamW(([{"params": enc_params, "lr": args.lr_enc}] if enc_params else []) + [{"params": head_params, "lr": args.lr_head}], weight_decay=0.01)
steps = args.epochs * math.ceil(len(train) / args.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
logit = lambda y: math.log(min(max(y, 1e-4), 1 - 1e-4) / (1 - min(max(y, 1e-4), 1 - 1e-4)))


def loss_reader(a, b, M, qs):
    B, D, L = M.shape; N = max(len(q["sets"]) for q in qs)
    X = torch.zeros(B, N, D, L, device=M.device); Y = torch.zeros(B, N, device=M.device); W = torch.zeros(B, N, device=M.device)
    beta = torch.tensor([logit(q["y0"]) for q in qs], device=M.device)
    for bb, q in enumerate(qs):
        for n, (S, y) in enumerate(q["sets"]):
            for j, i in S:
                if j < D and i < L: X[bb, n, j, i] = 1
            Y[bb, n] = y; W[bb, n] = 1.0 / len(q["sets"])        # every question weighs the same
    X = X * M[:, None].float(); phi = field_value(a, b, X)
    if args.link == "linear":   # datamodel-style regression of the log-likelihood gain
        y0 = torch.tensor([max(q["y0"], 1e-4) for q in qs], device=M.device)
        l = (phi - (torch.log(Y.clamp(min=1e-4)) - torch.log(y0)[:, None])) ** 2
        return (l * W).sum() / B
    V = (X * net._v[:, None]).sum((-1, -2)) if USE_VAR else torch.zeros_like(phi)
    l = F.binary_cross_entropy_with_logits(expected_logit(beta[:, None], phi, V), Y.clamp(1e-4, 1 - 1e-4), reduction="none")
    loss = (l * W).sum() / B
    if USE_VAR:   # S = 0 term of the same likelihood: the closed-book head
        y0 = torch.tensor([q["y0"] for q in qs], device=M.device).clamp(1e-4, 1 - 1e-4)
        loss = loss + F.binary_cross_entropy_with_logits(net.closed_book([q["question"] for q in qs]), y0)
    return loss


def loss_relevance(a, M, qs):
    gold = torch.zeros_like(a)
    for bb, q in enumerate(qs):
        for j, i in q["gold"]:
            if i < a.shape[2]: gold[bb, j, i] = 1
    return F.binary_cross_entropy_with_logits(a[M], gold[M])


def loss_crf(M, qs):
    """ablation: BIO CRF trained on the reader-optimal measured subset (highest y, then fewest sentences)."""
    B, D, L = M.shape; tags = torch.zeros(B, D, L, dtype=torch.long)
    for bb, q in enumerate(qs):
        S = set(max(q["sets"], key=lambda t: (t[1], -len(t[0])))[0])
        for j, i in S:
            if j < D and i < L: tags[bb, j, i] = 2 if (j, i - 1) in S else 1
    return net.crf.nll(net._h.reshape(B * D, L, -1).float(), tags.to(M.device).reshape(B * D, L), M.reshape(B * D, L))


def run():
    net.train(); net.bi.eval(); step = 0; t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(train))); random.shuffle(order)
        for s in range(0, len(order), args.bs):
            qs = [train[k] for k in order[s:s + args.bs]]
            with torch.autocast(DEV, dtype=torch.bfloat16):
                a, b, M = net.forward_batch([q["question"] for q in qs], [q["docs"] for q in qs], args.Lmax)
                loss = loss_reader(a, b, M, qs) if args.sup == "reader" else loss_relevance(a, M, qs) if args.sup == "relevance" else loss_crf(M, qs)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step(); sched.step(); step += 1
            if step % int(os.environ.get("GROW_LOG", 100)) == 0:
                extra = f" v {net._v[M].mean().item():.3f}" if USE_VAR else ""
                print(f"step {step}/{steps} loss {loss.item():.4f} a+ {(a[M] > 0).float().mean().item():.3f} "
                      f"b {b[M].mean().item():+.3f}{extra} {time.time() - t0:.0f}s", flush=True)
    torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("bi.")}, f"{ROOT}/ckpt/{args.out}.pt")
    print("done", flush=True)


os.makedirs(f"{ROOT}/ckpt", exist_ok=True)
json.dump(dict(vars(args), use_var=USE_VAR), open(f"{ROOT}/ckpt/{args.out}.json", "w"))
run()
