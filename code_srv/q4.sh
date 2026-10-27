#!/bin/bash
# v4 pipeline on GPU2 only
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
BASE="sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,mog,rerank@3,provence"
[ -f ckpt/v4.pt ] || $P code/grow/train.py v4 --crf > logs/train_v4.log 2>&1
$P code/grow/grow_infer.py v4 hotpotqa,2wiki,nq --split eval --theta 1.0
GROW_READER=$R14 $P code/grow/eval_reader.py hotpotqa v4_t1.0 --split eval --no_cb
GROW_READER=$R14 $P code/grow/eval_reader.py 2wiki v4_t1.0 --split eval --no_cb
$P code/grow/baselines.py nq --split eval
GROW_READER=$R14 $P code/grow/eval_reader.py nq $BASE,v3_t0.9,v3_t0.95,v3_t0.98,v3_t0.99,v4_t1.0 --split eval
echo MAIN_DONE
for ab in "no_cnn --no_cnn" "no_set --no_set" "no_xq --no_xq" "no_reader --no_reader" "frozen --train_layers 0" "goldonly --no_teacher"; do
  set -- $ab; n=$1; shift
  [ -f ckpt/v4ab_$n.pt ] || $P code/grow/train.py v4ab_$n --crf --n_train 10000 $@ > logs/train_v4ab_$n.log 2>&1
done
[ -f ckpt/v4ab_ref.pt ] || $P code/grow/train.py v4ab_ref --crf --n_train 10000 > logs/train_v4ab_ref.log 2>&1
T=""
for n in ref no_cnn no_set no_xq no_reader frozen goldonly; do $P code/grow/grow_infer.py v4ab_$n hotpotqa,2wiki --split eval --theta 1.0; T="$T,v4ab_${n}_t1.0"; done
for ds in hotpotqa 2wiki; do GROW_READER=$R14 $P code/grow/eval_reader.py $ds ${T#,} --split eval --no_cb; done
for r in /home/yaokang/models/Qwen3-4B-AWQ data/hf/Qwen__Qwen3-8B-AWQ; do
  [ -f $r/config.json ] || continue
  for ds in hotpotqa 2wiki; do GROW_READER=$r $P code/grow/eval_reader.py $ds doc@3,rerank@3,provence,mog,sentwin,v3_t0.98,v4_t1.0 --split eval; done
done
echo Q4_DONE
