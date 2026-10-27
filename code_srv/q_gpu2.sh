#!/bin/bash
# GPU2 queue: theta sweep on HotpotQA-tune, then main evaluation on HotpotQA / 2Wiki
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2
export GROW_READER=/home/yaokang/models/Qwen3-14B-AWQ
P=.venv/bin/python
for th in 0.3 0.5 0.7 0.85; do $P code/grow/grow_infer.py full hotpotqa --split tune --theta $th --tag grow_t$th; done
$P code/grow/eval_reader.py hotpotqa grow_t0.3,grow_t0.5,grow_t0.7,grow_t0.85,sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,rerank@3,provence,mog --split tune
echo Q2_DONE
