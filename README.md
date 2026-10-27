# GROW-Net

GROW-Net learns how much evidence to retrieve for a question from a reader's answer quality. It scores sentences, adjacent sentence pairs and complete documents, then uses structured decoding to select the context passed to the reader.

## Code layout

| Path | Purpose |
| --- | --- |
| `code_srv/grow/rof.py` | Reader-odds field and structured decoding |
| `code_srv/grow/train_rof.py` | Training with reader feedback |
| `code_srv/grow/infer_rof.py` | Evidence selection and context-length sweeps |
| `code_srv/grow/probe_gen.py`, `probe_rof.py` | Reader measurements used for supervision |
| `code_srv/grow/eval_reader.py` | Answer, evidence and context-cost evaluation |
| `code/grow/` | Additional implementations and experiment utilities |
| `repos/beir/` | BEIR retrieval and evaluation code |

The commands below use the implementation in `code_srv/grow/`. Keep its modules together when running an experiment.

## Environment and inputs

Use a Python environment with a CUDA-enabled PyTorch installation. The main pipeline also uses Transformers, NumPy, pandas, datasets and vLLM; BEIR comparisons have their own dependencies in `repos/beir/`.

Run the examples in Bash from the repository root:

```bash
export GROW_ROOT="$PWD"
```

`code_srv/grow/common.py` defines the data and model paths. Place the BGE encoder and reranker in the directories it expects under `data/hf/`, and set `GROW_READER` to your local reader model when generating answers. Download HotpotQA, 2WikiMultihopQA and Natural Questions from their dataset providers. The `prep_multihop.py` and `prep_nq*.py` scripts convert these inputs to the project's pickle format; check their dataset-cache paths before running them.

Training needs processed questions and reader measurements. For the answer-quality target, measurements are read from `data/gen/teacherab_<dataset>.pkl` and `data/gen/teacherfull_<dataset>.pkl`; `probe_gen.py` generates these measurements from the prepared probe sets. Additional probe tags can be supplied with `--probes`.

## Train and retrieve

Once those inputs are prepared:

```bash
python code_srv/grow/train_rof.py grow_act --target act --seed 0
python code_srv/grow/infer_rof.py grow_act hotpotqa,2wiki --deltas 0 --tag grow_act
python code_srv/grow/eval_reader.py hotpotqa grow_act_d0
```

`--target act` uses the F1 of the reader's greedy answer. `--target lik` selects the likelihood-based target. The output argument is a checkpoint name: training writes `ckpt/grow_act.pt` and its configuration, while inference writes selection files under `results/`. Pass a comma-separated list to `--deltas` to compare context lengths.

Datasets, model weights, generated measurements and experiment outputs are prepared locally. This repository contains the implementation and experiment scripts, without manuscript or publication-figure files.

## License

The BEIR source in `repos/beir/` retains its Apache 2.0 license and notices. A license for the project-specific code has not yet been specified.
