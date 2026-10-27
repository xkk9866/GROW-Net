#!/bin/bash
# Queue A (GPU3): theta selection -> main eval (HotpotQA/2Wiki/NQ) -> reader robustness -> BEIR
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
TH="0.6,0.7,0.8,0.9,0.95,0.98,0.99"
[ -f results/sel_hotpotqa_tune_v3_t0.99.pkl ] || $P code/grow/grow_infer.py v3 hotpotqa --split tune --theta $TH
GROW_READER=$R14 $P code/grow/eval_reader.py hotpotqa v3_t0.6,v3_t0.7,v3_t0.8,v3_t0.9,v3_t0.95,v3_t0.98,v3_t0.99,mog --split tune --no_cb
$P - <<'PY'
import json; m=json.load(open("results/metrics_hotpotqa_tune_Qwen3-14B-AWQ.json"))
c={k:v for k,v in m.items() if k.startswith("v3_t")}; best=max(c,key=lambda k:(round(c[k]["F1"],1),-c[k]["Tokens"]))
open("results/theta.txt","w").write(best.split("_t")[1]); print("best theta",best,c[best])
PY
T=$(cat results/theta.txt)
BASE="sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,mog,rerank@3,provence"
for ds in hotpotqa 2wiki nq; do
  $P code/grow/grow_infer.py v3 $ds --split eval --theta $TH
  $P code/grow/grow_infer.py v3 $ds --split eval --theta $T --no_closure --tag v3noclos
  $P code/grow/grow_infer.py v3 $ds --split eval --theta $T --no_promote --tag v3noprom
  GROW_READER=$R14 $P code/grow/eval_reader.py $ds $BASE,v3_t0.6,v3_t0.7,v3_t0.8,v3_t0.9,v3_t0.95,v3_t0.98,v3_t0.99,v3noclos_t$T,v3noprom_t$T --split eval
done
for r in /home/yaokang/models/Qwen3-4B-AWQ data/hf/Qwen__Qwen3-8B-AWQ; do
  [ -f $r/config.json ] || continue
  for ds in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $ds doc@3,rerank@3,provence,mog,pas@3,sentwin,v3_t$T --split eval; done
done
$P code/grow/beir_eval.py v3 scifact,nfcorpus,arguana,scidocs,fiqa,trec-covid
echo QA_DONE
