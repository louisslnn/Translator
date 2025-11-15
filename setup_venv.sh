#!/bin/bash
set -euo pipefail

echo ">>> Setting up virtual environment"

# Check if venv already exists
if [ -d "venv" ]; then
    echo ">>> Virtual environment already exists. Skipping creation."
else
    echo ">>> Creating virtual environment..."
    python3 -m venv venv
fi

echo ">>> Activating virtual environment..."
source venv/bin/activate

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing other dependencies..."
pip install -r requirements.txt

echo ">>> Virtual environment setup complete!"
echo ">>> To activate the venv manually, run: source venv/bin/activate"

