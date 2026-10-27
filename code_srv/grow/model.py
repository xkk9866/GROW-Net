"""GROW-Net: one granularity-elastic evidence network.

  (0) single sentence index      : every sentence of the corpus is embedded once by a frozen bi-encoder (bge-base); passage /
                                   document scores are soft-max pooled from it, so all granularities share one index and
                                   every unit keeps an exact provenance pointer (doc, a, b).
  (1) query-conditioned reader   : a cross-encoder backbone (initialised from bge-reranker-v2-m3, top layers fine-tuned) reads
                                   each candidate document *together with the question* once; mean-pooling its token states
                                   inside every sentence gives query-conditioned, in-context sentence states u_i.
  (2) discourse-structure layers : query FiLM -> gated dilated 1-D CNN along the sentence axis (coreference, conditions,
                                   continuation) -> query-token cross-attention -> cross-document set attention (inter-hop).
  (3) sufficiency head           : K information-need slots (gates w_k), coverage pi_{k,i}, dependencies delta^L/R_i and
        Suff(S|q) = prod_k [1 - prod_{i in S}(1 - pi_{k,i} rho_i(S))]^{w_k},   rho_i(S) = (1-dL_i[i-1 notin S])(1-dR_i[i+1 notin S]).
Granularity is never chosen explicitly: it is the solution of  max_S log Suff(S|q) - lambda tok(S)  (select_ew.py)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

NFEAT = 8


class GatedDilatedConv(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.conv = nn.Conv1d(d, 2 * d, 3, padding=dil, dilation=dil)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, m):  # x [N,L,d], m [N,L]
        y = self.conv((x * m[..., None]).transpose(1, 2)).transpose(1, 2)
        a, g = y.chunk(2, -1)
        return self.ln(x + a * torch.sigmoid(g)) * m[..., None]


class GROWNet(nn.Module):
    def __init__(self, bi_path, ce_path, d=256, K=4, train_layers=8, use_cnn=True, use_set=True, use_xq=True, use_reader=True,
                 use_dep=True, max_len=512):
        super().__init__()
        self.K, self.d, self.max_len = K, d, max_len
        self.use_cnn, self.use_set, self.use_xq, self.use_reader, self.use_dep = use_cnn, use_set, use_xq, use_reader, use_dep
        # (0) frozen bi-encoder (index + query)
        self.btok = AutoTokenizer.from_pretrained(bi_path); self.bi = AutoModel.from_pretrained(bi_path)
        for p in self.bi.parameters(): p.requires_grad = False
        H = self.bi.config.hidden_size
        # (1) query-conditioned reader
        self.ctok = AutoTokenizer.from_pretrained(ce_path); self.ce = AutoModelForSequenceClassification.from_pretrained(ce_path)
        H2 = self.ce.config.hidden_size
        nl = self.ce.config.num_hidden_layers
        for n, p in self.ce.named_parameters():
            keep = any(f"encoder.layer.{i}." in n for i in range(nl - train_layers, nl)) or "classifier" in n
            p.requires_grad = keep and use_reader and train_layers > 0
        # (2) structure
        self.inp = nn.Linear(H + H2, d)
        self.feat = nn.Linear(NFEAT, d)
        self.film = nn.Linear(H, 2 * d)
        self.conv = nn.ModuleList([GatedDilatedConv(d, r) for r in (1, 2, 4)])
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.qtok_proj = nn.Linear(H, d)
        self.xq = nn.MultiheadAttention(d, 4, batch_first=True); self.xq_ln = nn.LayerNorm(d)
        self.setl = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        # (3) sufficiency head
        self.slots = nn.Parameter(torch.randn(K, d) * 0.02)
        self.slot_attn = nn.MultiheadAttention(d, 4, batch_first=True); self.slot_ln = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.U = nn.Linear(d, d)
        self.pi_bias = nn.Parameter(torch.full((K,), -2.0))
        self.dep = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 2))
        self.qg = nn.Linear(H, d)
        self.crf = SpanCRF(d)

    @property
    def device(self): return self.U.weight.device

    # ------------------------------------------------------------------ (0) bi-encoder views (frozen)
    @torch.no_grad()
    def encode_query(self, questions, prefix, max_len=64):
        enc = self.btok([prefix + q for q in questions], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        h = self.bi(**enc).last_hidden_state
        return F.normalize(h[:, 0].float(), dim=-1), h.float(), enc["attention_mask"].bool()

    @torch.no_grad()
    def encode_sents(self, texts, max_len=96, bs=512):
        out = []
        for s in range(0, len(texts), bs):
            enc = self.btok(texts[s:s + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            out.append(F.normalize(self.bi(**enc).last_hidden_state[:, 0].float(), dim=-1))
        return torch.cat(out)

    # ------------------------------------------------------------------ (1) query-conditioned reader
    def read(self, questions, docs_sents, titles):
        """questions[j], docs_sents[j], titles[j] for each (question, document) pair.
        returns list of [n_j, H2] sentence states and [N] document relevance logits."""
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
        h = o.hidden_states[-1]; logit = o.logits.squeeze(-1)
        out = []
        for b, sp in enumerate(spans):
            sid = torch.tensor([x == 1 for x in seq_ids[b]]); st, en = off[b, :, 0], off[b, :, 1]
            A = torch.zeros(len(sp), h.shape[1])
            for i, (a, e) in enumerate(sp):
                A[i] = (sid & (st >= a) & (en <= e) & (en > st)).float()
            A = A / A.sum(1, keepdim=True).clamp(min=1)
            out.append(A.to(h.device, h.dtype) @ h[b])
        return out, logit

    # ------------------------------------------------------------------ (2)+(3)
    def structure(self, q, qtok, qmask, Z, Uc, feats, M):
        """q [B,H] qtok [B,T,H] qmask [B,T]; Z [B,D,L,H] bi-encoder sentence vectors; Uc [B,D,L,H2] reader states;
        feats [B,D,L,NFEAT]; M [B,D,L] bool."""
        B, D, L, _ = Z.shape
        Mf = M.float()
        if not self.use_reader: Uc = torch.zeros_like(Uc)
        g, b = self.film(q).chunk(2, -1)
        h = self.inp(torch.cat([Z, Uc], -1)) * (1 + g[:, None, None]) + b[:, None, None] + self.feat(feats)
        h = h.reshape(B * D, L, -1); m = Mf.reshape(B * D, L)
        if self.use_cnn:
            for c in self.conv: h = c(h, m)
        else:
            h = (h + self.ffn(h)) * m[..., None]
        h = h.reshape(B, D * L, -1)
        qt = self.qtok_proj(qtok)
        if self.use_xq:
            a, _ = self.xq(h, qt, qt, key_padding_mask=~qmask)
            h = self.xq_ln(h + a)
        if self.use_set:
            h = self.setl(h, src_key_padding_mask=~M.reshape(B, D * L))
        h = (h * Mf.reshape(B, D * L)[..., None]).reshape(B, D, L, -1)
        self._h = h
        s = self.slots[None].expand(B, -1, -1)
        a, _ = self.slot_attn(s, qt, qt, key_padding_mask=~qmask)
        a = self.slot_ln(s + a)
        w = torch.sigmoid(self.gate(a)).squeeze(-1)
        pi_logit = torch.einsum("bkd,bnld->bknl", a, self.U(h)) / math.sqrt(self.d) + self.pi_bias[None, :, None, None]
        pi_logit = pi_logit.masked_fill(~M[:, None], -20.0)
        hl = F.pad(h, (0, 0, 1, 0))[:, :, :-1]; hr = F.pad(h, (0, 0, 0, 1))[:, :, 1:]
        qg = self.qg(q)[:, None, None]
        dl = self.dep(torch.cat([h, hl, h * qg], -1))[..., 0]
        dr = self.dep(torch.cat([h, hr, h * qg], -1))[..., 1]
        Ml = F.pad(M, (1, 0))[:, :, :-1]; Mr = F.pad(M, (0, 1))[:, :, 1:]
        neg = torch.tensor(-20.0, device=dl.device, dtype=dl.dtype)
        dl = torch.where(Ml & M, dl, neg); dr = torch.where(Mr & M, dr, neg)
        if not self.use_dep: dl = torch.full_like(dl, -20.0); dr = torch.full_like(dr, -20.0)
        return {"pi": torch.sigmoid(pi_logit.float()), "pi_logit": pi_logit, "w": w.float(),
                "dL": torch.sigmoid(dl.float()), "dR": torch.sigmoid(dr.float())}

    # ------------------------------------------------------------------ convenience: full forward for B questions
    def forward_batch(self, questions, docs_list, qprefix, Lmax, bi_cache=None):
        """questions [B]; docs_list[b] = list of docs (dict title, sents). bi_cache: optional per-doc precomputed bi vectors."""
        B, D = len(questions), max(len(x) for x in docs_list)
        q, qtok, qmask = self.encode_query(questions, qprefix)
        pairs_q, pairs_s, pairs_t, pos = [], [], [], []
        for b, docs in enumerate(docs_list):
            for j, d in enumerate(docs):
                pairs_q.append(questions[b]); pairs_s.append(d["sents"][:Lmax]); pairs_t.append(d["title"]); pos.append((b, j))
        Hb, H2 = self.bi.config.hidden_size, self.ce.config.hidden_size
        Z = torch.zeros(B, D, Lmax, Hb, device=self.device); Uc = torch.zeros(B, D, Lmax, H2, device=self.device)
        M = torch.zeros(B, D, Lmax, dtype=torch.bool, device=self.device); dlog = torch.zeros(B, D, device=self.device)
        if bi_cache is None:
            txt = [f"{t}: {s}" for t, ss in zip(pairs_t, pairs_s) for s in ss]
            zz = self.encode_sents(txt); o = 0
        if self.use_reader:
            U, lg = self.read(pairs_q, pairs_s, pairs_t)
        for k, (b, j) in enumerate(pos):
            n = len(pairs_s[k]); M[b, j, :n] = True
            if bi_cache is None: Z[b, j, :n] = zz[o:o + n]; o += n
            else: Z[b, j, :n] = bi_cache[k][:n].to(self.device, Z.dtype)
            if self.use_reader: Uc[b, j, :n] = U[k][:n].to(Uc.dtype); dlog[b, j] = lg[k].float()
        feats = make_feats(q, Z, M, dlog)
        return self.structure(q, qtok, qmask, Z, Uc, feats, M), M


def log_suff(pi, w, dL, dR, S, eps=1e-6):
    """pi [B,K,D,L], w [B,K], dL/dR [B,D,L], S [B,N,D,L] in {0,1}  ->  log Suff [B,N]"""
    Sl = F.pad(S, (1, 0))[..., :-1]; Sr = F.pad(S, (0, 1))[..., 1:]
    rho = (1 - dL[:, None] * (1 - Sl)) * (1 - dR[:, None] * (1 - Sr))
    p = (pi[:, None] * rho[:, :, None]).clamp(max=1 - eps)
    miss = (torch.log1p(-p) * S[:, :, None]).sum((-1, -2))
    cov = torch.log((-torch.expm1(miss)).clamp(min=eps))
    return (w[:, None] * cov).sum(-1)


def make_feats(qz, Z, M, dlog):
    """scalar features: bi-encoder sim, pooled doc score, doc rank, rel. position, is-first, n-sents, sim-docmax, reader doc logit."""
    B, D, L, _ = Z.shape
    s = torch.einsum("bh,bdlh->bdl", qz.float(), Z.float()).masked_fill(~M, -1.0)
    dsc = torch.logsumexp((s / 0.05).masked_fill(~M, -1e4), -1) * 0.05
    dmax = s.max(-1).values
    rank = dsc.argsort(-1, descending=True).argsort(-1).float()
    n = M.float().sum(-1, keepdim=True).clamp(min=1)
    pos = torch.arange(L, device=Z.device).float()[None, None].expand(B, D, L)
    f = torch.stack([s, dsc[..., None].expand(-1, -1, L), (rank / D)[..., None].expand(-1, -1, L), pos / n,
                     (pos == 0).float(), torch.log(n).expand(-1, -1, L) / 3, s - dmax[..., None], (dlog / 5)[..., None].expand(-1, -1, L)], -1)
    return f * M[..., None].float()


# ====================================================================== v4: evidence segmentation CRF head
TAGS = 3  # O, B, I


class SpanCRF(nn.Module):
    """Linear-chain CRF over the sentence sequence of every candidate document. The BIO grammar (I only after B/I) is the
    only structural constraint; all emission and transition potentials are learned. Viterbi decoding returns the exact
    jointly optimal segmentation, i.e. the evidence spans and therefore their granularity."""

    def __init__(self, d):
        super().__init__()
        self.emit = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, TAGS))
        self.trans = nn.Parameter(torch.zeros(TAGS, TAGS)); self.start = nn.Parameter(torch.zeros(TAGS))
        mask = torch.zeros(TAGS, TAGS); mask[0, 2] = -1e4          # O -> I forbidden
        self.register_buffer("gram", mask); st = torch.zeros(TAGS); st[2] = -1e4; self.register_buffer("gram_start", st)

    def potentials(self, h):  # h [N,L,d]
        return self.emit(h).float(), self.trans + self.gram, self.start + self.gram_start

    def seq_loglik(self, e, T, s, tags, m):
        """log p(tags) for many tag sequences sharing emissions. e [N,L,3]; tags [N,R,L]; m [N,L] -> [N,R] (unnormalised score)."""
        N, R, L = tags.shape
        ee = e[:, None].expand(N, R, L, TAGS)
        sc = s[tags[..., 0]] + ee[..., 0, :].gather(-1, tags[..., :1]).squeeze(-1)
        for t in range(1, L):
            mt = m[:, t][:, None].float()
            sc = sc + (T[tags[..., t - 1], tags[..., t]] + ee[..., t, :].gather(-1, tags[..., t:t + 1]).squeeze(-1)) * mt
        return sc

    def log_partition(self, e, T, s, m):
        alpha = s[None] + e[:, 0]
        for t in range(1, e.shape[1]):
            na = torch.logsumexp(alpha[:, :, None] + T[None], 1) + e[:, t]
            alpha = torch.where(m[:, t][:, None], na, alpha)
        return torch.logsumexp(alpha, 1)

    def nll(self, h, tags, m):
        """tags [N,L] long, m [N,L] bool (prefix masks). returns mean negative log-likelihood over sequences with >=1 token."""
        e, T, s = self.potentials(h)
        N, L, _ = e.shape
        alpha = s[None] + e[:, 0]
        score = s[tags[:, 0]] + e[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, L):
            mt = m[:, t]
            na = torch.logsumexp(alpha[:, :, None] + T[None], 1) + e[:, t]
            alpha = torch.where(mt[:, None], na, alpha)
            sc = T[tags[:, t - 1], tags[:, t]] + e[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + sc * mt.float()
        logZ = torch.logsumexp(alpha, 1)
        v = m[:, 0]
        return ((logZ - score) * v.float()).sum() / v.float().sum().clamp(min=1)

    @torch.no_grad()
    def viterbi(self, h, m):
        e, T, s = self.potentials(h)
        N, L, _ = e.shape
        dp = s[None] + e[:, 0]; bp = []
        for t in range(1, L):
            cand = dp[:, :, None] + T[None]
            best, arg = cand.max(1)
            nd = best + e[:, t]
            mt = m[:, t][:, None]
            dp = torch.where(mt, nd, dp); bp.append(torch.where(mt, arg, torch.arange(TAGS, device=e.device)[None].expand(N, -1)))
        last = dp.argmax(1); path = [last]
        for b in reversed(bp):
            last = b.gather(1, last[:, None]).squeeze(1); path.append(last)
        return torch.stack(path[::-1], 1) * m.long()   # [N,L]  0=O 1=B 2=I
