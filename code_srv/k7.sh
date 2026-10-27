#!/bin/bash
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GROW_GPU_UTIL=0.5 GROW_MAXLEN=16384; P=.venv/bin/python
R4=/home/yaokang/models/Qwen3-4B-AWQ; RL=/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
L=logs/r/k7.log
for r in $R4 $RL; do for d in hotpotqa 2wiki nq; do
  $P code/grow/eval_reader.py $d r3_d0 --reader $r --no_cb 2>&1 | grep EM= | sed "s|^|$(basename $r) |" >> $L
done; done
echo K7_DONE >> $L
