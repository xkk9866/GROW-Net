#!/bin/bash
# usage: k8.sh GPU READER_PATH DATASET TAGS   -> appends EM lines to logs/r/k8.log
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GROW_GPU_UTIL=0.85 GROW_MAXLEN=16384; P=.venv/bin/python
$P code/grow/eval_reader.py $3 $4 --reader $2 --no_cb 2>&1 | grep EM= | sed "s|^|$(basename $2) |" >> logs/r/k8.log
echo "K8_DONE $1 $3" >> logs/r/k8.log
