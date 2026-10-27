#!/bin/bash
# GPU2: teacher (single shard) -> train v5 -> v5 inference -> all LLM reader evaluations
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
rm -f data/teacher_nq_*of2.pkl
[ -f data/teacher_nq_0of1.pkl ] || GROW_READER=$R14 GROW_GPU_UTIL=0.80 GROW_MAXBT=2048 $P code/grow/teacher.py 0 1 all nq
[ -f ckpt/v5.pt ] || $P code/grow/train.py v5 --crf --no_cnn --no_xq --nq 4000 > logs/train_v5.log 2>&1
$P code/grow/grow_infer.py v5 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --tag v5hop
$P code/grow/grow_infer.py v5 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --tag v5nohop
$P code/grow/grow_infer.py v5 hotpotqa,2wiki,nq --split eval --theta 1.0 --K_docs 20 --hop2 --indep --tag v5indep
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d v5hop_t1.0,v5nohop_t1.0,v5indep_t1.0 --no_cb; done
for r in /home/yaokang/models/Qwen3-4B-AWQ data/hf/Qwen__Qwen3-8B-AWQ; do
  for d in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $d v5hop_t1.0 --no_cb; done
done
until grep -q ABL_INFER_DONE logs/q7a.log; do sleep 120; done
T=""; for n in noset gold hotonly frozen noreader withcnn; do T="$T,v5ab_${n}_t1.0"; done
for d in hotpotqa 2wiki nq; do GROW_READER=$R14 $P code/grow/eval_reader.py $d ${T#,} --no_cb; done
echo Q7B_DONE
