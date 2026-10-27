"""GROW-Net (field form): evidence selection as exact maximisation of an amortised reader log-odds field.

The reader LLM's probability of producing the answer from evidence set S is modelled as
        P_reader(a | q, S)  =  sigma( beta_q + Phi(S) ),     Phi(S) = sum_i a_i x_i + sum_i b_i x_i x_{i+1}
  beta_q   : closed-book log-odds of the reader (a property of the question only; never needed at inference),
  a_i      : first-order effect of sentence i on the reader's answer log-odds (negative for distractors: the reader
             itself pays for irrelevant context),
  b_i      : interaction of adjacent sentences i, i+1 of the same document (contextual dependency: coreference,
             conditions, list continuation; negative for redundancy).
Both potentials are produced by ONE network from the question and the candidate documents only:
  query-conditioned reader (cross-encoder over [q; document]) -> per-sentence in-context states -> one set-attention
  layer over all candidate sentences -> unary / adjacent-pair heads.
Because sigma is monotone, argmax_S P_reader = argmax_S Phi(S) for every beta_q, and on the chain the maximiser is
computed exactly by a two-state dynamic program. A sentence enters the evidence iff it raises the predicted
answer odds given its neighbours; spans (granularity) are the connected components of the maximiser."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class GROWField(nn.Module):
    def __init__(self, bi_path, ce_path, d=256, train_layers=8, use_set=True, use_pair=True, max_len=512, use_crf=False, use_var=False):
        super().__init__()
        self.use_set, self.use_pair, self.max_len, self.d, self.use_var = use_set, use_pair, max_len, d, use_var
        self.btok = AutoTokenizer.from_pretrained(bi_path); self.bi = AutoModel.from_pretrained(bi_path)
        for p in self.bi.parameters(): p.requires_grad = False          # first-stage index encoder (frozen, shared with baselines)
        self.ctok = AutoTokenizer.from_pretrained(ce_path); self.ce = AutoModelForSequenceClassification.from_pretrained(ce_path)
        H, nl = self.ce.config.hidden_size, self.ce.config.num_hidden_layers
        for n, p in self.ce.named_parameters():
            p.requires_grad = train_layers > 0 and any(f"encoder.layer.{i}." in n for i in range(nl - train_layers, nl))
        self.inp = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, d))
        self.setl = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.unary = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.pair = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        if use_var:   # epistemic variance of each sentence's effect, and the reader's closed-book log-odds (question only)
            self.var = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.beta = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, d), nn.GELU(), nn.Linear(d, 1))
        if use_crf:   # ablation only: most-likely BIO labelling instead of the utility argmax
            from grow.model import SpanCRF
            self.crf = SpanCRF(d)

    @property
    def device(self): return self.unary[0].weight.device

    @torch.no_grad()
    def encode_query(self, questions, prefix, max_len=64):
        enc = self.btok([prefix + q for q in questions], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        return F.normalize(self.bi(**enc).last_hidden_state[:, 0].float(), dim=-1)

    def read(self, questions, docs_sents, titles):
        """cross-encoder over (question, document); sentence state = mean of the reader's token states inside it."""
        texts, spans = [], []
        for title, ss in zip(titles, docs_sents):
            t = f"{title}:"; sp = []
            for s in ss:
                t += " "; a = len(t); t += s; sp.append((a, len(t)))
            texts.append(t); spans.append(sp)
        enc = self.ctok(questions, texts, padding=True, truncation="only_second", max_length=self.max_len,
                        return_offsets_mapping=True, return_tensors="pt")
        off = enc.pop("offset_mapping")
        seq_ids = [enc.sequence_ids(b) for b in range(len(texts))]
        h = self.ce(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states[-1]
        out = []
        for b, sp in enumerate(spans):
            sid = torch.tensor([x == 1 for x in seq_ids[b]]); st, en = off[b, :, 0], off[b, :, 1]
            A = torch.zeros(len(sp), h.shape[1])
            for i, (a, e) in enumerate(sp):
                A[i] = (sid & (st >= a) & (en <= e) & (en > st)).float()
            A = A / A.sum(1, keepdim=True).clamp(min=1)
            out.append(A.to(h.device, h.dtype) @ h[b])
        return out

    def forward_batch(self, questions, docs_list, Lmax):
        """returns unary a [B,D,L], adjacent interaction b [B,D,L] (b[..,i] couples i and i+1), mask M [B,D,L]."""
        B, D = len(questions), max(len(x) for x in docs_list)
        pq, ps, pt, pos = [], [], [], []
        for bb, docs in enumerate(docs_list):
            for j, d in enumerate(docs):
                pq.append(questions[bb]); ps.append(d["sents"][:Lmax]); pt.append(d["title"]); pos.append((bb, j))
        U = self.read(pq, ps, pt)
        H = self.ce.config.hidden_size
        X = torch.zeros(B, D, Lmax, H, device=self.device); M = torch.zeros(B, D, Lmax, dtype=torch.bool, device=self.device)
        for k, (bb, j) in enumerate(pos):
            n = min(len(ps[k]), U[k].shape[0]); X[bb, j, :n] = U[k][:n].to(X.dtype); M[bb, j, :n] = True
        h = self.inp(X)
        if self.use_set:
            h = self.setl(h.reshape(B, D * Lmax, -1), src_key_padding_mask=~M.reshape(B, D * Lmax)).reshape(B, D, Lmax, -1)
        h = h * M[..., None]; self._h = h
        a = self.unary(h).squeeze(-1).float().masked_fill(~M, 0.0)
        hr = F.pad(h, (0, 0, 0, 1))[:, :, 1:]; Mr = F.pad(M, (0, 1))[:, :, 1:] & M
        b = self.pair(torch.cat([h, hr, h * hr], -1)).squeeze(-1).float().masked_fill(~Mr, 0.0)
        if not self.use_pair: b = torch.zeros_like(b)
        self._v = F.softplus(self.var(h).squeeze(-1).float()).masked_fill(~M, 0.0) if self.use_var else None
        return a, b, M

    def closed_book(self, questions):
        """predicted closed-book log-odds beta_q, read from the question alone."""
        enc = self.ctok(questions, [""] * len(questions), padding=True, truncation=True, max_length=64, return_tensors="pt")
        h = self.ce(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states[-1][:, 0]
        return self.beta(h).squeeze(-1).float()


def field_value(a, b, X):
    """Phi(S) for many sets sharing potentials. a,b [B,D,L]; X [B,N,D,L] in {0,1} -> [B,N]."""
    Xr = F.pad(X, (0, 1))[..., 1:]
    return (X * a[:, None]).sum((-1, -2)) + (X * Xr * b[:, None]).sum((-1, -2))


PROBIT = math.pi / 8   # E_{z~N(m,s2)} sigma(z) ~= sigma(m / sqrt(1 + pi s2 / 8))  (MacKay's probit approximation)


def expected_logit(beta, phi, V):
    """logit of the reader's expected answer probability when the field's log-odds are Gaussian with mean beta + phi and
    variance V = sum_i v_i x_i."""
    return (beta + phi) / torch.sqrt(1 + PROBIT * V)


@torch.no_grad()
def bayes_argmax(a, b, v, M, beta, iters=30):
    """evidence maximising the reader's EXPECTED answer probability  sigma((beta + Phi(x)) / sqrt(1 + c V(x))).
    The maximiser lies on the frontier {max Phi(x) - lambda V(x)} for some exchange rate lambda between expected log-odds
    and variance; stationarity gives lambda = c (beta+Phi) / (2 (1 + c V)). For a fixed lambda the frontier point is exact on
    the chains (two-state DP); we alternate the two (a Dinkelbach-type fixed point), started from both regimes (lambda = 0
    and the rate of the all-sentence and empty selections), and keep the best iterate. a,b,v,M [B,D,L]; beta [B] -> x [B,D,L].
    When the reader is predicted to answer (beta+Phi > 0) uncertain sentences are priced out; when it is not, they are
    welcome -- the amount of evidence follows the reader's confidence."""
    B, D, L = M.shape; Mf = M.float(); Mr = F.pad(Mf, (0, 1))[..., 1:] * Mf

    def value(x):
        xf = x.float(); xr = F.pad(xf, (0, 1))[..., 1:]
        phi = (xf * a).sum((1, 2)) + (xf * xr * b).sum((1, 2)); V = (xf * v).sum((1, 2))
        return expected_logit(beta, phi, V), PROBIT * (beta + phi) / (2 * (1 + PROBIT * V))

    best = torch.full((B,), -1e9, device=a.device); xb = torch.zeros_like(M)
    full_phi = (Mf * a).sum((1, 2)) + (Mr * b).sum((1, 2)); full_V = (Mf * v).sum((1, 2))
    starts = [torch.zeros(B, device=a.device), PROBIT * (beta + full_phi) / (2 * (1 + PROBIT * full_V)), PROBIT * beta / 2]
    for lam in starts:
        for _ in range(iters):
            x = exact_argmax((a - lam[:, None, None] * v).reshape(B * D, L), b.reshape(B * D, L), M.reshape(B * D, L)).reshape(B, D, L)
            val, new = value(x); up = val > best
            best = torch.where(up, val, best); xb[up] = x[up]
            if torch.allclose(new, lam): break
            lam = new
    return xb


@torch.no_grad()
def exact_argmax(a, b, M, delta=0.0):
    """exact maximiser of sum_i (a_i - delta) x_i + sum_i b_i x_i x_{i+1} on every document chain (two-state DP).
    a,b,M [N,L] -> x [N,L] bool. delta=0 is the field itself (used everywhere except the offset-sensitivity study)."""
    a = (a - delta).masked_fill(~M, -1e4); N, L = a.shape
    d0 = torch.zeros(N, device=a.device); d1 = a[:, 0].clone(); bp0, bp1 = [], []
    for t in range(1, L):
        c1 = torch.stack([d0, d1 + b[:, t - 1]], 1)          # previous state 0 / 1 -> current 1
        c0 = torch.stack([d0, d1], 1)                           # previous state 0 / 1 -> current 0
        n1, i1 = c1.max(1); n0, i0 = c0.max(1)
        d1, d0 = n1 + a[:, t], n0; bp1.append(i1); bp0.append(i0)
    x = torch.zeros(N, L, dtype=torch.long, device=a.device); cur = (d1 > d0).long(); x[:, L - 1] = cur
    for t in range(L - 1, 0, -1):
        cur = torch.where(cur == 1, bp1[t - 1], bp0[t - 1]); x[:, t - 1] = cur
    return x.bool() & M
