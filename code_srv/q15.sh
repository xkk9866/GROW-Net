#!/bin/bash
# GPU7 only, strictly sequential. Every LLM is int4 (AWQ) and < 14B; probe + main reader Qwen3-8B-AWQ.
# GROW-Net g1: full-candidate probe + universe probe, Gaussian (uncertainty-aware) reader field, expected-utility decision.
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R8=data/hf/Qwen__Qwen3-8B-AWQ; R4=/home/yaokang/models/Qwen3-4B-AWQ; RL=/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
BASE="sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,mog,rerank@1,rerank@2,rerank@3,rerank@5,rerank2hop@3,provence"
DS=hotpotqa,2wiki,nq
until grep -q TEACHERFULL_DONE logs/q14.log; do sleep 120; done
# ---------------------------------------------------------------- main model
[ -f ckpt/g1.pt ] || $P code/grow/train_field.py g1 > logs/train_g1.log 2>&1
$P code/grow/fidelity.py gold_count,size,f1,g1 > logs/fidelity_g.log 2>&1
$P code/grow/infer_field.py g1 $DS --deltas=-0.5,0,0.5 --tag g1 --save_scores > logs/inf_g1.log 2>&1
$P code/grow/infer_field.py g1 $DS --decision mean --tag g1m >> logs/inf_g1.log 2>&1
$P code/grow/infer_field.py g1 $DS --no_hop2 --K1 20 --tag g1nh >> logs/inf_g1.log 2>&1
$P code/grow/infer_field.py g1 $DS --unary --tag g1u >> logs/inf_g1.log 2>&1
for d in hotpotqa 2wiki nq; do GROW_READER=$R8 $P code/grow/eval_reader.py $d g1_d0,g1_d-0.5,g1_d0.5,g1m_d0,g1nh_d0,g1u_d0,f1_d0 --no_cb; done > logs/ev_g1.log 2>&1
echo MAIN_DONE
# ---------------------------------------------------------------- ablations (half of the probed questions, identical decoding)
for ab in "ref" "ab --data ab" "full --data full" "novar --no_var" "nopair --no_pair" "noset --no_set" "relev --sup relevance" "frozen --train_layers 0"; do
  set -- $ab; n=$1; shift
  [ -f ckpt/gab_$n.pt ] || $P code/grow/train_field.py gab_$n --frac 0.5 $@ > logs/train_gab_$n.log 2>&1
  $P code/grow/infer_field.py gab_$n $DS --tag gab_$n >> logs/inf_gab.log 2>&1
  [ $n = relev ] || $P code/grow/fidelity.py gab_$n >> logs/fidelity_g.log 2>&1
done
T=""; for n in ref ab full novar nopair noset relev frozen; do T="$T,gab_${n}_d0"; done
for d in hotpotqa 2wiki nq; do GROW_READER=$R8 $P code/grow/eval_reader.py $d ${T#,} --no_cb; done > logs/ev_gab.log 2>&1
echo ABL_DONE
# ---------------------------------------------------------------- seeds
for s in 1 2; do
  [ -f ckpt/g1s$s.pt ] || $P code/grow/train_field.py g1s$s --seed $s > logs/train_g1s$s.log 2>&1
  $P code/grow/infer_field.py g1s$s $DS --tag g1s$s >> logs/inf_seed.log 2>&1
done
for d in hotpotqa 2wiki nq; do GROW_READER=$R8 $P code/grow/eval_reader.py $d g1s1_d0,g1s2_d0 --no_cb; done > logs/ev_seed.log 2>&1
echo SEED_DONE
# ---------------------------------------------------------------- unseen readers
for r in $R4 $RL; do
  for d in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $d $BASE,g1_d0 --reader $r; done
done > logs/ev_readers.log 2>&1
echo Q15_DONE
