"""Collect every reader-evaluation line in logs/ into results/harvest.json: {reader: {dataset: {tag: metrics}}} (latest file wins)."""
import glob, os, re, json
os.chdir(os.path.expanduser("~/research/grow_www27"))
pat = re.compile(r"^(?:(\S+-AWQ\S*|Qwen3-\S+)\s+)?(hotpotqa|2wiki|nq)\s+(\S+)\s+(EM=.*)$")
out = {}
files = sorted(glob.glob("logs/**/*.log", recursive=True), key=os.path.getmtime)
for fn in files:
    base = os.path.basename(fn)
    for ln in open(fn, errors="replace"):
        m = pat.match(ln.strip())
        if not m: continue
        rd, d, tag, rest = m.groups()
        if rd is None:
            rd = "Qwen3-4B-AWQ" if "readers" in base and False else "Qwen__Qwen3-8B-AWQ"
        if "readers" in base and m.group(1) is None:
            continue  # reader logs without a reader prefix are ambiguous
        kv = dict(re.findall(r"(\w+)=([-\d.]+)", rest.split("|")[0]))
        met = {k: float(v) for k, v in kv.items()}
        g = re.search(r"gran s/p/d=([\d.]+)/([\d.]+)/([\d.]+)", rest)
        if g: met.update(g_sent=float(g[1]), g_pas=float(g[2]), g_doc=float(g[3]))
        met["_src"] = fn
        out.setdefault(rd, {}).setdefault(d, {})[tag] = met
json.dump(out, open("results/harvest.json", "w"), indent=0)
for rd in out:
    print(rd, {d: len(v) for d, v in out[rd].items()})
