#!/bin/bash
# GPU4 only: NQ teacher -> GROW-Net v6 (reader-utility structured distillation) -> inference -> readers -> ablations
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
[ -f data/teacher_nq_0of1.pkl ] || GROW_READER=$R14 GROW_GPU_UTIL=0.72 GROW_MAXBT=2048 $P code/grow/teacher.py 0 1 all nq > logs/teacher_nq.log 2>&1
ls data/teacher_nq_0of1.pkl || exit 1
BASE="--crf --no_cnn --no_xq"
[ -f ckpt/v6.pt ] || $P code/grow/train.py v6 $BASE --rusd --tgt argmax --nq 4000 > logs/train_v6.log 2>&1
$P code/grow/grow_infer.py v6 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --tag v6hop > logs/inf_v6.log 2>&1
$P code/grow/grow_infer.py v6 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --tag v6nohop >> logs/inf_v6.log 2>&1
$P code/grow/grow_infer.py v6 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --indep --tag v6indep >> logs/inf_v6.log 2>&1
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d v6hop_t1.0,v6nohop_t1.0,v6indep_t1.0 --no_cb; done > logs/ev_v6.log 2>&1
echo MAIN_DONE >> logs/ev_v6.log
# ablations: 10K HotpotQA + 4K NQ, one epoch, identical decoding
for ab in "ref --rusd --tgt argmax" "nokl --tgt argmax" "gold --tgt gold --no_teacher" "hotonly --rusd --tgt argmax --nonq" "noset --rusd --tgt argmax --no_set" "noreader --rusd --tgt argmax --no_reader"; do
  set -- $ab; n=$1; shift; a="$*"; nq="--nq 4000"; b="$BASE"
  [[ $a == *--nonq* ]] && nq="" && a=${a/--nonq/}
  [[ $a == *--withcnn* ]] && b="--crf --no_xq" && a=${a/--withcnn/}
  [ -f ckpt/v6ab_$n.pt ] || $P code/grow/train.py v6ab_$n $b $a $nq --n_train 10000 > logs/train_v6ab_$n.log 2>&1
  $P code/grow/grow_infer.py v6ab_$n hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --tag v6ab_$n >> logs/inf_v6ab.log 2>&1
done
T=""; for n in ref nokl gold hotonly noset noreader; do T="$T,v6ab_${n}_t1.0"; done
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d ${T#,} --no_cb; done > logs/ev_v6ab.log 2>&1
for r in /home/yaokang/models/Qwen3-4B-AWQ /home/yaokang/models/Qwen3-8B-AWQ; do
  for d in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $d v6hop_t1.0 --no_cb; done
done > logs/ev_v6_readers.log 2>&1
echo Q8_DONE >> logs/ev_v6_readers.log
