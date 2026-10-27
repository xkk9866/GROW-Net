#!/bin/bash
# GPU2: teacher shard 1 -> ablation trainings -> ablation evaluation (final inference configuration)
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
until [ -f data/nq_train.pkl ]; do sleep 60; done
[ -f data/teacher_nq_1of2.pkl ] || GROW_READER=$R14 GROW_GPU_UTIL=0.72 GROW_MAXBT=2048 $P code/grow/teacher.py 1 2 all nq
until [ -f data/teacher_nq_0of2.pkl ]; do sleep 60; done
for ab in "noset --no_set" "gold --no_teacher" "hotonly" "frozen --train_layers 0" "noreader --no_reader" "withcnn"; do
  set -- $ab; n=$1; shift
  extra="--nq 12000"; [ $n = hotonly ] && extra=""
  cnn="--no_cnn --no_xq"; [ $n = withcnn ] && cnn=""
  [ -f ckpt/v5ab_$n.pt ] || $P code/grow/train.py v5ab_$n --crf $cnn $extra $@ > logs/train_v5ab_$n.log 2>&1
done
T=""
for n in noset gold hotonly frozen noreader withcnn; do
  $P code/grow/grow_infer.py v5ab_$n hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --tag v5ab_$n; T="$T,v5ab_${n}_t1.0"
done
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d ${T#,} --no_cb; done
echo Q6B_DONE
