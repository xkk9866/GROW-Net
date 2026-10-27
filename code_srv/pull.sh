#!/bin/bash
cd ~/research/grow_www27/results
F="metrics_*.json"
for d in hotpotqa 2wiki nq; do for t in r3_d0 rerank@3 rerank@5 doc@5 rerank2hop@3; do F="$F pred_${d}_eval_${t}_Qwen__Qwen3-8B-AWQ.pkl"; done; done
tar czf /tmp/p.tgz $F 2>/dev/null; ls -la /tmp/p.tgz
