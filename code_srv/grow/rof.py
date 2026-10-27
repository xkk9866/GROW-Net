"""GROW-Net core: the nested reader-odds field, its exact maximiser and its profiled (closed-book-free) likelihood.

The reader's probability of producing the answer a from evidence x (a binary vector over candidate sentences) is
        P_R(a | q, x) = sigma( beta_q + Phi_q(x) ),
        Phi_q(x) = sum_d c_d z_d(x) + sum_i a_i x_i + sum_{i~i+1} b_i x_i x_{i+1},      z_d(x) = OR_{i in d} x_i
  beta_q : closed-book log-odds (a property of the question; it never has to be known),
  a_i    : effect of sentence i on the reader's answer log-odds,
  b_i    : interaction of adjacent sentences of one document (dependency > 0, redundancy < 0),
  c_d    : effect of opening document d at all (its title / source identity, the cost of one more source).
One principle drives every part of the method: the evidence is argmax_x P_R(a | q, x).
  (i)  sigma is monotone, so the argmax is the argmax of Phi_q and does not depend on beta_q. Learning therefore only fits
       within-question log-odds contrasts: beta_q is profiled out of the likelihood of every question (profile_bce).
  (ii) units are contiguous runs of sentences inside documents, so Phi_q is a chain field with one gate per document and
       its maximiser is computed exactly by a three-state dynamic program (nested_argmax). Its runs are the granularity.
  (iii) an argmax only needs the field to be right where decisions are taken, so the reader is probed around the
       decoder's own decisions (probe_rof.py)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

NEG = -1e4


class ReaderOddsField(nn.Module):
    """question + candidate documents -> (a, b, c). The only inputs are the texts; one query-conditioned reader
    (cross-encoder initialised from bge-reranker-v2-m3) reads every (question, document) pair, one set-attention layer lets
    sentences and documents of all candidates see each other, and three heads emit the potentials."""

    def __init__(self, bi_path, ce_path, d=256, train_layers=8, use_set=True, use_pair=True, use_doc=True, max_len=512):
        super().__init__()
        self.use_set, self.use_pair, self.use_doc, self.max_len, self.d = use_set, use_pair, use_doc, max_len, d
        self.btok = AutoTokenizer.from_pretrained(bi_path); self.bi = AutoModel.from_pretrained(bi_path)
        for p in self.bi.parameters(): p.requires_grad = False          # first-stage index encoder (frozen, shared with baselines)
        self.ctok = AutoTokenizer.from_pretrained(ce_path); self.ce = AutoModelForSequenceClassification.from_pretrained(ce_path)
        H, nl = self.ce.config.hidden_size, self.ce.config.num_hidden_layers
        for n, p in self.ce.named_parameters():
            p.requires_grad = train_layers > 0 and (any(f"encoder.layer.{i}." in n for i in range(nl - train_layers, nl)) or "classifier" in n)
        self.inp = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, d))
        self.typ = nn.Embedding(2, d)                                     # 0 = sentence token, 1 = document token
        self.setl = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.unary = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.pair = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.doc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    @property
    def device(self): return self.unary[0].weight.device

    @torch.no_grad()
    def encode_query(self, questions, prefix, max_len=64):
        enc = self.btok([prefix + q for q in questions], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        return F.normalize(self.bi(**enc).last_hidden_state[:, 0].float(), dim=-1)

    def read(self, questions, docs_sents, titles):
        """cross-encoder over (question, document). Returns per pair: sentence states [n, H] (mean of the token states inside
        each sentence), the document state (first token) and the backbone's own relevance read-out (classifier logit)."""
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
        o = self.ce(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True)
        h = o.hidden_states[-1]
        out = []
        for b, sp in enumerate(spans):
            sid = torch.tensor([x == 1 for x in seq_ids[b]]); st, en = off[b, :, 0], off[b, :, 1]
            A = torch.zeros(len(sp), h.shape[1])
            for i, (a, e) in enumerate(sp):
                A[i] = (sid & (st >= a) & (en <= e) & (en > st)).float()
            A = A / A.sum(1, keepdim=True).clamp(min=1)
            out.append(A.to(h.device, h.dtype) @ h[b])
        return out, h[:, 0], o.logits.squeeze(-1)

    def forward_batch(self, questions, docs_list, Lmax):
        """returns a [B,D,L], b [B,D,L] (b[..,i] couples i and i+1), c [B,D], sentence mask M [B,D,L] (prefix masks)."""
        B, D = len(questions), max(len(x) for x in docs_list)
        pq, ps, pt, pos = [], [], [], []
        for bb, docs in enumerate(docs_list):
            for j, d in enumerate(docs):
                pq.append(questions[bb]); ps.append(d["sents"][:Lmax]); pt.append(d["title"]); pos.append((bb, j))
        U, Hd, R = self.read(pq, ps, pt)
        H = self.ce.config.hidden_size
        X = torch.zeros(B, D, Lmax, H, device=self.device); M = torch.zeros(B, D, Lmax, dtype=torch.bool, device=self.device)
        Xd = torch.zeros(B, D, H, device=self.device); Md = torch.zeros(B, D, dtype=torch.bool, device=self.device)
        rel = torch.zeros(B, D, device=self.device)
        for k, (bb, j) in enumerate(pos):
            n = min(len(ps[k]), U[k].shape[0]); X[bb, j, :n] = U[k][:n].to(X.dtype); M[bb, j, :n] = True
            Xd[bb, j] = Hd[k].to(X.dtype); Md[bb, j] = n > 0; rel[bb, j] = R[k].float()
        h = self.inp(X) + self.typ.weight[0]; hd = self.inp(Xd) + self.typ.weight[1]
        if self.use_set:
            z = torch.cat([hd, h.reshape(B, D * Lmax, -1)], 1); m = torch.cat([Md, M.reshape(B, D * Lmax)], 1)
            z = self.setl(z, src_key_padding_mask=~m)
            hd, h = z[:, :D], z[:, D:].reshape(B, D, Lmax, -1)
        h = h * M[..., None]; self._h = h
        a = self.unary(h).squeeze(-1).float().masked_fill(~M, 0.0)
        hr = F.pad(h, (0, 0, 0, 1))[:, :, 1:]; Mr = F.pad(M, (0, 1))[:, :, 1:] & M
        b = self.pair(torch.cat([h, hr, h * hr], -1)).squeeze(-1).float().masked_fill(~Mr, 0.0)
        if not self.use_pair: b = torch.zeros_like(b)
        c = (self.doc(hd).squeeze(-1).float() + rel).masked_fill(~Md, 0.0)
        if not self.use_doc: c = torch.zeros_like(c)
        return a, b, c, M


def field_value(a, b, c, X):
    """Phi(x) for many selections sharing potentials. a,b [B,D,L]; c [B,D]; X [B,N,D,L] in {0,1} -> [B,N]."""
    Xr = F.pad(X, (0, 1))[..., 1:]
    z = X.amax(-1)
    return (X * a[:, None]).sum((-1, -2)) + (X * Xr * b[:, None]).sum((-1, -2)) + (z * c[:, None]).sum(-1)


def profile_bce(phi, Y, W, iters=12):
    """likelihood of the reader's measurements with the closed-book log-odds beta_q profiled out.
    phi, Y, W [B,N] (W = 1 for a measured selection, 0 for padding; the empty selection has phi = 0).
    beta_q = argmin_beta sum_n W BCE(sigma(beta + phi_n), y_n) is a one-dimensional convex problem solved by Newton steps;
    by the envelope theorem its gradient w.r.t. the potentials is the partial gradient at beta_q, so beta_q is detached.
    Only within-question differences of phi enter the loss -- exactly the information the argmax uses."""
    with torch.no_grad():
        p_d, y_d, w = phi.detach().float(), Y.float(), W.float()
        ym = ((y_d * w).sum(1) / w.sum(1).clamp(min=1)).clamp(1e-3, 1 - 1e-3)
        beta = torch.log(ym / (1 - ym)) - (p_d * w).sum(1) / w.sum(1).clamp(min=1)
        for _ in range(iters):
            s = torch.sigmoid(beta[:, None] + p_d)
            g = ((s - y_d) * w).sum(1); hss = (s * (1 - s) * w).sum(1) + 1e-4
            beta = (beta - g / hss).clamp(-20, 20)
    l = F.binary_cross_entropy_with_logits(beta[:, None] + phi, Y.clamp(1e-4, 1 - 1e-4), reduction="none")
    return ((l * W).sum(1) / W.sum(1).clamp(min=1)).mean(), beta


@torch.no_grad()
def chain_best(a, b, M):
    """best NON-EMPTY selection on every chain: three-state DP (E = nothing selected yet, I = current sentence selected,
    O = something selected before, current not). a,b,M [N,L] -> value v* [N], selection [N,L] bool."""
    a = a.float().masked_fill(~M, NEG); N, L = a.shape; dev = a.device
    E = torch.zeros(N, device=dev); I = a[:, 0].clone(); O = torch.full((N,), NEG, device=dev)
    bpI, bpO = [], []
    for t in range(1, L):
        cI = torch.stack([E, O, I + b[:, t - 1].float()], 1)          # previous E / O / I -> I
        cO = torch.stack([O, I], 1)                                     # previous O / I -> O
        nI, iI = cI.max(1); nO, iO = cO.max(1)
        I, O = nI + a[:, t], nO; bpI.append(iI); bpO.append(iO)
    v, st = torch.stack([I, O], 1).max(1)                             # 0 = ends in I, 1 = ends in O
    st = torch.where(st == 0, torch.ones_like(st), torch.full_like(st, 2))   # state codes: 0 E, 1 I, 2 O
    x = torch.zeros(N, L, dtype=torch.bool, device=dev)
    for t in range(L - 1, -1, -1):
        x[:, t] = st == 1
        if t == 0: break
        pI = torch.tensor([0, 2, 1], device=dev)[bpI[t - 1]]            # E, O, I
        pO = torch.tensor([2, 1], device=dev)[bpO[t - 1]]               # O, I
        st = torch.where(st == 1, pI, torch.where(st == 2, pO, st))
    return v, x & M


@torch.no_grad()
def nested_argmax(a, b, c, M, delta=0.0):
    """exact maximiser of Phi over all selections: each document independently either stays closed (value 0) or is opened
    with its best non-empty chain (value c_d + v*_d). delta is a per-sentence offset used ONLY for the cost-sensitivity
    study (delta = 0 is the method). a,b,M [B,D,L]; c [B,D] -> x [B,D,L] bool, per-document gain [B,D]."""
    B, D, L = M.shape
    v, x = chain_best((a - delta).reshape(B * D, L), b.reshape(B * D, L), M.reshape(B * D, L))
    gain = (c.float() + v.reshape(B, D)).masked_fill(~M.any(-1), NEG)
    return x.reshape(B, D, L) & (gain > 0)[..., None], gain
