#!/bin/bash
# wait for both teacher shards, then train the full model (GPU7) and server-side ablations (GPU6, 1 epoch each)
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1
until [ -f data/teacher_0of2.pkl ] && [ -f data/teacher_1of2.pkl ]; do sleep 60; done
sleep 30
CUDA_VISIBLE_DEVICES=7 .venv/bin/python code/grow/train.py full > logs/train_full.log 2>&1 &
( while pgrep -f "baselines[.]py" > /dev/null; do sleep 60; done
  for ab in "full1ep" "no_cnn --no_cnn" "no_set --no_set" "no_xq --no_xq" "no_ctx --no_ctx" "no_dep --no_dep"; do
    set -- $ab; name=$1; shift
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python code/grow/train.py ab_$name --epochs 1 $@ > logs/train_ab_$name.log 2>&1
  done ) &
wait
