#!/bin/bash
cd ~/research/grow_www27
echo "--- running job scripts"; ps -eo pid,args | grep -E 'code/k[0-9]+(run)?\.sh' | grep -v grep
echo "--- evals"; ps -eo pid,args | grep eval_reader | grep -v grep | cut -c1-160
echo "--- k8.log"; cut -c1-150 logs/r/k8.log 2>/dev/null
echo "--- gpus"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
