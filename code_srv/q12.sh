#!/bin/bash
# GPU7 only, strictly sequential (never two GPU jobs at once). Every LLM is int4 (AWQ) and < 14B; teacher + main reader Qwen3-8B-AWQ.
cd ~/research/grow_www27; export PYTHONPATH=code HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R8=data/hf/Qwen__Qwen3-8B-AWQ; R4=/home/yaokang/models/Qwen3-4B-AWQ; RL=/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
BASE="sent@5,sent@10,pas@3,pas@5,doc@3,doc@5,sentwin,automerge,c2f,mog,rerank@1,rerank@2,rerank@3,rerank@5,rerank2hop@3,provence"
until grep -q Q9T_DONE logs/q9t.log; do sleep 30; done
$P code/grow/baselines.py hotpotqa,2wiki,nq --only rerank@1,rerank@2,rerank@5,rerank2hop@3 > logs/base_new.log 2>&1
$P code/grow/fidelity.py gold_count,size > logs/fidelity.log 2>&1
[ -f ckpt/f1.pt ] || $P code/grow/train_field.py f1 > logs/train_f1.log 2>&1
$P code/grow/fidelity.py f1 >> logs/fidelity.log 2>&1
$P code/grow/infer_field.py f1 hotpotqa,2wiki,nq --deltas=-1,-0.5,0,0.5,1 --tag f1 --save_scores > logs/inf_f1.log 2>&1
$P code/grow/infer_field.py f1 hotpotqa,2wiki,nq --unary --tag f1u >> logs/inf_f1.log 2>&1
$P code/grow/infer_field.py f1 hotpotqa,2wiki,nq --no_hop2 --K1 20 --tag f1nh >> logs/inf_f1.log 2>&1
for d in hotpotqa 2wiki nq; do GROW_READER=$R8 $P code/grow/eval_reader.py $d $BASE,f1_d0,f1_d-1,f1_d-0.5,f1_d0.5,f1_d1,f1u_d0,f1nh_d0; done > logs/ev_f1.log 2>&1
echo MAIN_DONE
for ab in "ref" "nopair --no_pair" "noset --no_set" "frozen --train_layers 0" "relev --sup relevance" "crf --sup crf" "linear --link linear"; do
  set -- $ab; n=$1; shift
  [ -f ckpt/fab_$n.pt ] || $P code/grow/train_field.py fab_$n --n_hot 6000 --n_nq 2000 $@ > logs/train_fab_$n.log 2>&1
  $P code/grow/infer_field.py fab_$n hotpotqa,2wiki,nq --tag fab_$n >> logs/inf_fab.log 2>&1
  [ $n = crf ] || $P code/grow/fidelity.py fab_$n >> logs/fidelity.log 2>&1
done
T=""; for n in ref nopair noset frozen relev crf linear; do T="$T,fab_${n}_d0"; done
for d in hotpotqa 2wiki nq; do GROW_READER=$R8 $P code/grow/eval_reader.py $d ${T#,} --no_cb; done > logs/ev_fab.log 2>&1
echo ABL_DONE
for s in 1 2; do
  [ -f ckpt/f1s$s.pt ] || $P code/grow/train_field.py f1s$s --seed $s > logs/train_f1s$s.log 2>&1
  $P code/grow/infer_field.py f1s$s hotpotqa,2wiki,nq --tag f1s$s >> logs/inf_seed.log 2>&1
done
for d in hotpotqa 2wiki nq; do GROW_READER=$R8 $P code/grow/eval_reader.py $d f1s1_d0,f1s2_d0 --no_cb; done > logs/ev_seed.log 2>&1
echo SEED_DONE
for r in $R4 $RL; do
  for d in hotpotqa 2wiki nq; do GROW_READER=$r $P code/grow/eval_reader.py $d $BASE,f1_d0 --reader $r; done
done > logs/ev_readers.log 2>&1
echo Q12_DONE
