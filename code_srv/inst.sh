#!/bin/bash
cd ~/research/grow_www27; export PATH=$HOME/.local/bin:$PATH
source .venv/bin/activate
UV_HTTP_TIMEOUT=300 uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple "vllm==0.8.5" "transformers==4.51.3" sentence-transformers pandas pyarrow faiss-cpu scikit-learn nltk matplotlib "huggingface_hub[hf_transfer]" && echo INSTALL_DONE
