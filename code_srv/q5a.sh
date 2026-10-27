#!/bin/bash
# GPU2: NQ baselines (reader), CRF-vs-independent ablation (final config), reader robustness
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
BASE="sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,mog,rerank@3,provence"
$P code/grow/baselines.py nq --split eval
GROW_READER=$R14 $P code/grow/eval_reader.py nq $BASE,v4_t1.0 --split eval
$P code/grow/grow_infer.py v4 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --indep --tag v4indep
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d v4indep_t1.0 --no_cb; done
for r in /home/yaokang/models/Qwen3-4B-AWQ data/hf/Qwen__Qwen3-8B-AWQ; do
  [ -f $r/config.json ] || continue
  for d in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $d doc@3,rerank@3,provence,mog,sentwin,automerge,v4hop_t1.0 --split eval; done
done
echo Q5A_DONE
