#!/bin/bash
# GPU3: component ablations under the final inference configuration (20 candidates + evidence-conditioned second hop)
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R14=/home/yaokang/models/Qwen3-14B-AWQ
T=""
for n in ref no_cnn no_set no_xq no_reader frozen goldonly; do
  $P code/grow/grow_infer.py v4ab_$n hotpotqa,2wiki --split eval --theta 1.0 --K_docs 20 --hop2 --tag v4abh_$n; T="$T,v4abh_${n}_t1.0"
done
$P code/grow/grow_infer.py v4 hotpotqa,2wiki --split eval --theta 1.0 --K_docs 20 --tag v4nohop
for d in hotpotqa 2wiki; do GROW_READER=$R14 $P code/grow/eval_reader.py $d ${T#,} --split eval --no_cb; done
echo Q5B_DONE
