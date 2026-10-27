import os, sys, time
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
D = "Y:/xkk_workspace/workspace/WWW/8991ARAG/data/hf"
jobs = [
 ("BAAI/bge-base-en-v1.5","model",None),
 ("BeIR/nq","dataset",None), ("BeIR/nq-qrels","dataset",None),
 ("google-research-datasets/nq_open","dataset",None),
]
for b in ["scifact","nfcorpus","fiqa","scidocs","arguana","trec-covid"]:
    jobs += [(f"BeIR/{b}","dataset",None),(f"BeIR/{b}-qrels","dataset",None)]
jobs += [("Qwen/Qwen2.5-7B-Instruct","model",None)]
for rid, rtype, pat in jobs:
    t=time.time()
    for k in range(3):
        try:
            p = snapshot_download(repo_id=rid, repo_type=rtype, local_dir=f"{D}/{rid.replace('/','__')}", allow_patterns=pat, max_workers=8)
            print("OK", rid, p, f"{time.time()-t:.0f}s", flush=True); break
        except Exception as e:
            print("ERR", rid, k, repr(e)[:300], flush=True); time.sleep(5)
