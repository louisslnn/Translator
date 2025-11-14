#!/bin/bash
set -e

echo ">>> Updating system"
sudo apt-get update -y

echo ">>> Installing Python dependencies"
pip install --upgrade pip
pip install datasets sentencepiece accelerate wandb matplotlib

echo ">>> Starting training"
python train.py --config config.yaml
