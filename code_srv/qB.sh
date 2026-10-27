#!/bin/bash
# Queue B (GPU2): component ablations (10K questions, 1 epoch, identical protocol) -> inference -> reader eval
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
rm -f ckpt/ab_*
$P code/grow/baselines.py nq --split eval
$P code/grow/baselines.py hotpotqa,2wiki --split eval --only mog
$P code/grow/baselines.py hotpotqa,2wiki --split tune --only mog
for ab in "ref" "no_cnn --no_cnn" "no_set --no_set" "no_xq --no_xq" "no_reader --no_reader" "no_dep --no_dep" "K1 --K 1" "no_suff --no_suff" "no_teacher --no_teacher" "teacher_only --teacher_only" "no_list --no_list" "frozen_reader --train_layers 0"; do
  set -- $ab; n=$1; shift
  [ -f ckpt/ab_$n.pt ] || $P code/grow/train.py ab_$n --epochs 1 --n_train 10000 $@ > logs/train_ab_$n.log 2>&1
done
until [ -f results/theta.txt ]; do sleep 120; done; T=$(cat results/theta.txt)
TAGS=""
for n in ref no_cnn no_set no_xq no_reader no_dep K1 no_suff no_teacher teacher_only no_list frozen_reader; do
  $P code/grow/grow_infer.py ab_$n hotpotqa,2wiki --split eval --theta $T; TAGS="$TAGS,ab_${n}_t$T"
done
for ds in hotpotqa 2wiki; do GROW_READER=/home/yaokang/models/Qwen3-14B-AWQ $P code/grow/eval_reader.py $ds ${TAGS#,} --split eval --no_cb; done
echo QB_DONE
