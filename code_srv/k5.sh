#!/bin/bash
# GPU5: missing ablation eval, then r3 extras (frontier deltas, no re-reading, regret, bootstrap).
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GROW_GPU_UTIL=0.85 GROW_MAXLEN=16384; P=.venv/bin/python
L=logs/r/k5.log
for d in hotpotqa 2wiki nq; do $P code/grow/eval_reader.py $d ab3_nodoc_d0 --no_cb 2>&1 | grep EM= >> $L; done
echo NODOC_DONE >> $L
until grep -q "JOBDONE r3$" logs/r/evf.log 2>/dev/null; do sleep 60; done
$P code/grow/infer_rof.py r3 hotpotqa,2wiki,nq --deltas=-0.5,0.25,1 > logs/r/inf_r3x.log 2>&1
for d in hotpotqa 2wiki nq; do $P code/grow/eval_reader.py $d r3_d-0.5,r3_d0.25,r3_d1 --no_cb 2>&1 | grep EM= >> $L; done
$P code/grow/infer_rof.py r3 hotpotqa,2wiki,nq --no_hop2 --tag r3nh > logs/r/inf_r3nh.log 2>&1
for d in hotpotqa 2wiki nq; do $P code/grow/eval_reader.py $d r3nh_d0 --no_cb 2>&1 | grep EM= >> $L; done
$P code/grow/regret.py r0,r3,ab3_ref,ab3_nopair,ab3_rand,ab3_fixb,ab3_noset > logs/r/regret3.log 2>&1
for d in hotpotqa 2wiki nq; do $P code/grow/bootstrap.py $d r3_d0 rerank@3,rerank@5,rerank2hop@3,doc@5,automerge,mog,provence,r0_d0 >> logs/r/boot3.log 2>&1; done
echo K5_DONE >> $L
