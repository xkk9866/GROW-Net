#!/bin/bash
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/python code/grow/train.py full > logs/train_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup bash -c ".venv/bin/python code/grow/baselines.py hotpotqa,2wiki --split eval; .venv/bin/python code/grow/baselines.py hotpotqa,2wiki --split tune; .venv/bin/python code/grow/baselines.py nq --split eval" > logs/baselines.log 2>&1 &
