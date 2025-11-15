#!/bin/bash
set -euo pipefail

echo ">>> Preparing environment"

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo ">>> Virtual environment not found. Running setup..."
    bash setup_venv.sh
fi

echo ">>> Activating virtual environment..."
source venv/bin/activate

# Set environment variables
export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_DATASETS_CACHE}"

# Set PyTorch CUDA memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> Starting training"
python train.py --config config.yaml
