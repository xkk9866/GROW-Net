#!/bin/bash
cd ~/research/grow_www27; export PYTHONPATH=code; P=.venv/bin/python
for d in hotpotqa 2wiki nq; do $P code/grow/bootstrap.py $d r3_d0 rerank@3,rerank@5,rerank2hop@3,doc@5,automerge,sentwin,mog,provence,r0_d0; done > logs/r/boot3.log 2>&1
cat logs/r/boot3.log
