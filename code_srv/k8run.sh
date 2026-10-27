#!/bin/bash
cd ~/research/grow_www27
R4=/home/yaokang/models/Qwen3-4B-AWQ; RL=/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
(bash code/k8.sh 1 $R4 hotpotqa r3_d0; bash code/k8.sh 1 $R4 2wiki r3_d0; bash code/k8.sh 1 $R4 nq r3_d0) &
(bash code/k8.sh 2 $RL hotpotqa r3_d0; bash code/k8.sh 2 $RL 2wiki r3_d0) &
(bash code/k8.sh 3 $RL nq r3_d0) &
(bash code/k8.sh 4 $RL nq doc@5,automerge,mog,rerank@3,rerank@5,rerank2hop@3,provence) &
wait
echo K8ALL >> logs/r/k8.log
