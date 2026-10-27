#!/bin/bash
cd ~/research/grow_www27
for p in $(pgrep -f "code/dl_server.py"); do kill $p; done
export HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1
nohup .venv/bin/python code/dl_server.py > logs/dl.log 2>&1 &
