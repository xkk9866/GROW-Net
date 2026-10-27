#!/bin/bash
# Re-evaluate every Qwen3-8B baseline into one log (metrics json files were clobbered by concurrent writers).
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GROW_GPU_UTIL=0.85 GROW_MAXLEN=16384; P=.venv/bin/python
BASE="sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,mog,rerank@1,rerank@2,rerank@3,rerank@5,rerank2hop@3,provence,r0_d0,r3_d0"
$P code/grow/eval_reader.py $2 $BASE 2>&1 | grep EM= >> logs/r/kb_$2.log
echo KB_DONE >> logs/r/kb_$2.log
