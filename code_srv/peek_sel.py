import pickle, sys, itertools
s = pickle.load(open(sys.argv[1], "rb"))
print(type(s), len(s) if hasattr(s, "__len__") else "")
if isinstance(s, dict):
    for k, v in itertools.islice(s.items(), 2):
        print("KEY", k, "->", str(v)[:1500])
else:
    print(str(s[0])[:2500])
if len(sys.argv) > 2:
    p = pickle.load(open(sys.argv[2], "rb"))
    print("PRED keys", list(p.keys()) if isinstance(p, dict) else type(p))
    for k in (p.keys() if isinstance(p, dict) else []):
        print(k, str(p[k])[:300])
