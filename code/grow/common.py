"""Shared utilities: paths, benchmark containers, multi-granularity units, vanilla dense encoder, token counting, QA metrics."""
import os, re, string, pickle, collections, json, glob
import numpy as np
import torch

ROOT = os.path.expanduser(os.environ.get("GROW_ROOT", "~/research/grow_www27"))
DATA = f"{ROOT}/data"
HF = f"{DATA}/hf"
EMB = f"{ROOT}/emb"
RES = f"{ROOT}/results"
BGE = f"{HF}/BAAI__bge-base-en-v1.5"
READER = os.environ.get("GROW_READER", f"{HF}/Qwen__Qwen3-8B-AWQ")
QPREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_SENTS = 3  # multi-hop corpora: passage = window of 3 consecutive sentences of a document
for d in (EMB, RES): os.makedirs(d, exist_ok=True)


def load_bench(name):
    return pickle.load(open(f"{DATA}/{name}_eval.pkl", "rb"))


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); pickle.dump(obj, open(path, "wb"))


# ---------------------------------------------------------------- multi-granularity units with provenance
def sent_text(doc, i):
    return f"{doc['title']}: {doc['sents'][i]}"


def span_text(doc, a, b):
    return f"{doc['title']}: " + " ".join(doc["sents"][a:b])


def doc_passages(doc):
    """passage spans (sentence index ranges). Corpora that come with native passages store them in doc['pas']."""
    if "pas" in doc: return doc["pas"]
    n = len(doc["sents"]); return [(s, min(n, s + PASSAGE_SENTS)) for s in range(0, n, PASSAGE_SENTS)]


class Units:
    """Flat sentence / passage / document arrays + cross-granularity provenance map
    (global sentence id <-> (doc, local idx) <-> passage id)."""

    def __init__(self, docs):
        self.docs = docs
        sd, si, pd_, ps = [], [], [], []
        self.doc_start = np.zeros(len(docs) + 1, dtype=np.int64)
        self.sent_pas = []
        for k, d in enumerate(docs):
            self.doc_start[k] = len(sd)
            pas = doc_passages(d)
            for j, (a, b) in enumerate(pas):
                for i in range(a, b): self.sent_pas.append(len(pd_))
                pd_.append(k); ps.append((a, b))
            for i in range(len(d["sents"])): sd.append(k); si.append(i)
        self.doc_start[len(docs)] = len(sd)
        self.sent_doc, self.sent_idx = np.array(sd), np.array(si)
        self.pas_doc, self.pas_span = np.array(pd_), ps
        self.sent_pas = np.array(self.sent_pas)

    def texts(self, level):
        if level == "sent": return [sent_text(self.docs[d], i) for d, i in zip(self.sent_doc, self.sent_idx)]
        if level == "pas": return [span_text(self.docs[d], a, b) for d, (a, b) in zip(self.pas_doc, self.pas_span)]
        if level == "doc": return [span_text(d, 0, len(d["sents"])) for d in self.docs]

    def gid(self, doc, i): return int(self.doc_start[doc] + i)


# ---------------------------------------------------------------- vanilla dense encoder (baselines)
class Encoder:
    def __init__(self, path=BGE, device="cuda"):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, bs=256, max_len=256, prefix=""):
        out = []
        order = np.argsort([-len(t) for t in texts])
        for s in range(0, len(texts), bs):
            batch = [prefix + texts[j] for j in order[s:s + bs]]
            enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            out.append(torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu())
        emb = torch.cat(out).numpy(); inv = np.empty_like(order); inv[order] = np.arange(len(order))
        return emb[inv]


def cached(path, fn):
    if os.path.exists(path): return np.load(path)
    x = fn(); np.save(path, x); return x


def dense_search(qemb, cemb, k, bs=512, device="cuda"):
    """exact inner-product search on GPU, corpus processed in shards (fp16)."""
    q_all = torch.from_numpy(np.asarray(qemb, dtype=np.float16)).to(device)
    best_v = torch.full((len(q_all), k), -1e4, device=device); best_i = torch.zeros((len(q_all), k), dtype=torch.long, device=device)
    shard = 2_000_000
    for c0 in range(0, len(cemb), shard):
        C = torch.from_numpy(np.asarray(cemb[c0:c0 + shard], dtype=np.float16)).to(device)
        for s in range(0, len(q_all), bs):
            sc = q_all[s:s + bs] @ C.T
            v, i = sc.float().topk(min(k, C.shape[0]), dim=1)
            v = torch.cat([best_v[s:s + bs], v], 1); i = torch.cat([best_i[s:s + bs], i + c0], 1)
            vv, o = v.topk(k, dim=1); best_v[s:s + bs] = vv; best_i[s:s + bs] = i.gather(1, o)
        del C; torch.cuda.empty_cache()
    return best_v.cpu().numpy(), best_i.cpu().numpy()


# ---------------------------------------------------------------- token counting (reader tokenizer)
_TOK = None


def ntok(text):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(READER)
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def ntok_batch(texts):
    global _TOK
    if _TOK is None: ntok("x")
    return [len(x) for x in _TOK(texts, add_special_tokens=False)["input_ids"]] if texts else []


# ---------------------------------------------------------------- QA metrics (SQuAD/HotpotQA official normalisation)
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, golds):
    return float(max(normalize_answer(pred) == normalize_answer(g) for g in golds))


def f1(pred, golds):
    best = 0.0
    p = normalize_answer(pred)
    for g in golds:
        g = normalize_answer(g)
        if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g: continue
        pt, gt = p.split(), g.split()
        common = collections.Counter(pt) & collections.Counter(gt); ns = sum(common.values())
        if ns == 0: continue
        pr, rc = ns / len(pt), ns / len(gt); best = max(best, 2 * pr * rc / (pr + rc))
    return best


def acc(pred, golds):
    p = normalize_answer(pred)
    return float(any(normalize_answer(g) in p for g in golds if normalize_answer(g)))


def has_answer(text, golds):
    t = normalize_answer(text)
    return float(any(normalize_answer(g) in t for g in golds if normalize_answer(g)))


# ---------------------------------------------------------------- simple sentence splitter (for corpora without sentence annotation)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sents(text, min_chars=20):
    parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    out = []
    for p in parts:  # merge very short fragments into previous sentence
        if out and len(p) < min_chars: out[-1] += " " + p
        else: out.append(p)
    return out or [text.strip()]


# ---------------------------------------------------------------- context formatting (shared by teacher, reader and all methods)
def format_context(sel, docs, max_docs=None):
    """sel: dict doc_id -> sorted list of selected local sentence indices (or list of (doc, idx)).
    Contiguous runs are joined; gaps are marked with ' ... ' so provenance and discourse order are preserved.
    Docs keep the insertion order of `sel` (= retrieval rank)."""
    if isinstance(sel, list):
        d = {}
        for a, i in sel: d.setdefault(a, set()).add(i)
        sel = {a: sorted(v) for a, v in d.items()}
    blocks = []
    for k, (a, idx) in enumerate(sel.items()):
        if max_docs and k >= max_docs: break
        doc = docs[a]; idx = sorted(idx); parts, prev = [], None
        for i in idx:
            if prev is not None and i != prev + 1: parts.append("...")
            parts.append(doc["sents"][i]); prev = i
        blocks.append(f"[{len(blocks)+1}] {doc['title']}: " + " ".join(parts))
    return "\n".join(blocks)


SYS_PROMPT = ("Answer the question based on the given documents. Only give me the short answer (an entity, number, date, "
              "short phrase, or yes/no) and do not output any other words.")


def user_prompt(question, ctx):
    if ctx: return f"Documents:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"
