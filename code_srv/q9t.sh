#!/bin/bash
# GPU7 only: random-ablation reader teacher with Qwen3-8B-AWQ (int4)  (HotpotQA 12K x 24, NQ 4K x 24); resumable, retried
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
P=.venv/bin/python; export GROW_READER=data/hf/Qwen__Qwen3-8B-AWQ GROW_GPU_UTIL=0.80
for src in hotpotqa nq; do
  n=12000; [ $src = nq ] && n=4000
  for try in 1 2 3 4 5; do
    [ -f data/teacherab_$src.pkl ] && break
    timeout 6h $P code/grow/teacher_ab.py $src $n 24 >> logs/teacherab_$src.log 2>&1
  done
done
echo Q9T_DONE
