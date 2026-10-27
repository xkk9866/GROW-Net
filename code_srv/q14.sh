#!/bin/bash
# GPU7 only: full-candidate reader probe (Qwen3-8B-AWQ, int4). Resumable; a stall watchdog restarts a hung engine.
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
P=.venv/bin/python; export GROW_READER=data/hf/Qwen__Qwen3-8B-AWQ GROW_GPU_UTIL=0.80 GROW_MAXBT=2048 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for job in "hotpotqa 12000 6000" "nq 0 4000"; do
  set -- $job; src=$1
  ( while [ ! -f data/teacherfull_$src.pkl ]; do sleep 300
      [ -f logs/teacherfull_$src.log ] || continue
      age=$(( $(date +%s) - $(stat -c %Y logs/teacherfull_$src.log) ))
      [ $age -gt 2400 ] && pgrep -f "teacher_full.py $src" > /dev/null && { echo "$(date) stall $src -> kill"; pkill -9 -f "teacher_full.py $src"; }
    done ) >> logs/watchdog_full.log 2>&1 &
  for try in 1 2 3 4 5 6; do
    [ -f data/teacherfull_$src.pkl ] && break
    timeout 8h $P code/grow/teacher_full.py $1 $2 $3 16 >> logs/teacherfull_$src.log 2>&1
  done
done
echo TEACHERFULL_DONE
