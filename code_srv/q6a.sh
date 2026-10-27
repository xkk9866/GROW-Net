#!/bin/bash
# GPU3: NQ distant-supervision data -> teacher shard 0 -> train v5 -> main evaluation -> reader robustness
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
[ -f data/nq_train.pkl ] || $P code/grow/prep_nq_train.py 12000
[ -f data/teacher_nq_0of2.pkl ] || GROW_READER=$R14 GROW_GPU_UTIL=0.72 GROW_MAXBT=2048 $P code/grow/teacher.py 0 2 all nq
until [ -f data/teacher_nq_1of2.pkl ]; do sleep 60; done
[ -f ckpt/v5.pt ] || $P code/grow/train.py v5 --crf --no_cnn --no_xq --nq 12000 > logs/train_v5.log 2>&1
$P code/grow/grow_infer.py v5 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --tag v5hop
$P code/grow/grow_infer.py v5 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --tag v5nohop
$P code/grow/grow_infer.py v5 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --indep --tag v5indep
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d v5hop_t1.0,v5nohop_t1.0,v5indep_t1.0 --no_cb; done
for r in /home/yaokang/models/Qwen3-4B-AWQ data/hf/Qwen__Qwen3-8B-AWQ; do
  for d in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $d v5hop_t1.0 --no_cb; done
done
echo Q6A_DONE
