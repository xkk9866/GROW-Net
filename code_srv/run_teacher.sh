#!/bin/bash
cd ~/research/grow_www27
export PYTHONPATH=code HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/python code/grow/teacher.py 0 2 > logs/teacher0.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup .venv/bin/python code/grow/teacher.py 1 2 > logs/teacher1.log 2>&1 &
