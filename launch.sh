#!/bin/bash
set -euo pipefail

echo ">>> Preparing environment"
sudo apt-get update -y

echo ">>> Installing Python dependencies (CUDA build)"
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install datasets tokenizers sentencepiece accelerate wandb matplotlib tensorboard pyyaml

export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_DATASETS_CACHE}"

echo ">>> Starting training"
python3 train.py --config config.yaml
