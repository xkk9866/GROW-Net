#!/bin/bash
# GPU3 (shared with another job): ablation trainings + inference (no LLM needed)
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
until [ -f data/teacher_nq_0of1.pkl ]; do sleep 120; done
for ab in "noset --no_set" "gold --no_teacher" "hotonly" "frozen --train_layers 0" "noreader --no_reader" "withcnn"; do
  set -- $ab; n=$1; shift
  extra="--nq 4000"; [ $n = hotonly ] && extra=""
  cnn="--no_cnn --no_xq"; [ $n = withcnn ] && cnn=""
  [ -f ckpt/v5ab_$n.pt ] || $P code/grow/train.py v5ab_$n --crf $cnn $extra $@ > logs/train_v5ab_$n.log 2>&1
  $P code/grow/grow_infer.py v5ab_$n hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --tag v5ab_$n
done
echo ABL_INFER_DONE
