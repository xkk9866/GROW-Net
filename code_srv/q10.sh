#!/bin/bash
# GPU7 only: field-fidelity test on held-out teacher questions, run alongside q9 as checkpoints appear (small memory)
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
P=.venv/bin/python
until [ -f data/teacherab_nq.pkl ]; do sleep 120; done
$P code/grow/fidelity.py gold_count,size > logs/fidelity.log 2>&1
until [ -f ckpt/f1.pt ]; do sleep 120; done; sleep 60
$P code/grow/fidelity.py f1 >> logs/fidelity.log 2>&1
for n in ref nopair noset frozen relev linear; do
  until [ -f ckpt/fab_$n.pt ]; do sleep 120; done; sleep 60
  $P code/grow/fidelity.py fab_$n >> logs/fidelity.log 2>&1
done
echo Q10_DONE >> logs/fidelity.log
