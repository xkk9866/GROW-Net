#!/bin/bash
cd ~/research/grow_www27
for p in $(pgrep -f "bash code/q14.sh"); do kill $p; done
for p in $(pgrep -f "grow/teacher_full.py"); do kill -9 $p; done
sleep 8; nvidia-smi --query-gpu=index,memory.used --format=csv | sed -n 9p
(setsid nohup bash code/q14.sh > logs/q14.log 2>&1 &)
sleep 2; ps -u yaokang -o pid,cmd | grep -E "q14|teacher_full" | grep -v grep
