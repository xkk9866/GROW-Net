#!/bin/bash
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 GROW_READER=/home/yaokang/models/Qwen3-14B-AWQ
rm -f data/teacher_*of2.pkl
CUDA_VISIBLE_DEVICES=7 GROW_GPU_UTIL=0.80 nohup .venv/bin/python code/grow/teacher.py 0 2 > logs/teacher0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 GROW_GPU_UTIL=0.55 nohup .venv/bin/python code/grow/teacher.py 1 2 > logs/teacher1.log 2>&1 &
