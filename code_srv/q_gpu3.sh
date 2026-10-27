#!/bin/bash
# GPU3 queue: NQ baselines (vanilla multi-granularity indexes) and GROW NQ index
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3
P=.venv/bin/python
$P code/grow/baselines.py hotpotqa,2wiki --split eval --only mog
$P code/grow/baselines.py hotpotqa,2wiki --split tune --only mog
$P code/grow/baselines.py nq --split eval
$P -c "import sys; sys.path.insert(0,'code'); from grow.grow_infer import *; import pickle; net,_=load_net('full'); B=pickle.load(open(f'{DATA}/nq_eval.pkl','rb')); build_index(net,'full','nq',B['docs'])"
echo Q3_DONE
