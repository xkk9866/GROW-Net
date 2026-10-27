#!/bin/bash
# GPU7 only: resume the NQ teacher (stall watchdog watches the NQ log only), then the full q12 pipeline
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
P=.venv/bin/python; export GROW_READER=data/hf/Qwen__Qwen3-8B-AWQ GROW_GPU_UTIL=0.80
( while [ ! -f data/teacherab_nq.pkl ]; do sleep 300
    age=$(( $(date +%s) - $(stat -c %Y logs/teacherab_nq.log) ))
    [ $age -gt 2400 ] && pgrep -f "teacher_ab.py nq" > /dev/null && { echo "$(date) stall -> kill"; pkill -9 -f "teacher_ab.py nq"; }
  done ) >> logs/watchdog_nq.log 2>&1 &
for try in 1 2 3 4 5 6; do
  [ -f data/teacherab_nq.pkl ] && break
  timeout 4h $P code/grow/teacher_ab.py nq 4000 24 >> logs/teacherab_nq.log 2>&1
done
[ -f data/teacherab_nq.pkl ] || { echo NQ_TEACHER_FAIL; exit 1; }
echo Q9T_DONE >> logs/q9t.log
bash code/q12.sh > logs/q12.log 2>&1
