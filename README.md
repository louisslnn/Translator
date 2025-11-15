# Translator

A transformer-based neural machine translation model for English-French translation.

## Setup

### Option 1: Automated Setup (Recommended)
Run the setup script to create a virtual environment and install all dependencies:

```bash
bash setup_venv.sh
```

Then activate the virtual environment:
```bash
source venv/bin/activate
```

### Option 2: Manual Setup
1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Training

### Using the launch script (automatically uses venv):
```bash
bash launch.sh
```

### Manual training (with venv activated):
```bash
source venv/bin/activate
python train.py --config config.yaml
```

### Training with custom options:
```bash
source venv/bin/activate
python train.py --config config.yaml --device cuda:0 --mixed_precision bf16
```

## Configuration

Edit `config.yaml` to adjust training parameters like batch size, learning rate, number of epochs, etc.

## Notes

- The model uses gradient checkpointing by default to reduce memory usage
- Mixed precision training (bf16) is enabled by default for faster training
- Model checkpoints are saved to the `weights/` directory
- TensorBoard logs are saved to the `runs/` directory