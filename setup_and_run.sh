#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for GRPO-Zero training..."

# Initialize conda for bash
echo "Initializing conda..."
eval "$($HOME/miniconda3/bin/conda init bash)"
source ~/.bashrc

# Create and activate conda environment
echo "Creating and activating conda environment..."
conda create -n grpo-zero python=3.10 -y || true
conda activate grpo-zero

# Install base dependencies with specific versions
echo "Installing dependencies..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge transformers=4.37.2 tensorboard pandas pyyaml
conda install -y numpy==1.24.3  # Specific version to avoid compatibility issues
pip install --force-reinstall numpy==1.24.3  # Make sure numpy is the right version
pip install attrdict timm einops

# Clone and install DeepSeek VL2
echo "Installing DeepSeek VL2..."
if [ ! -d "DeepSeek-VL2" ]; then
    git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
fi

# Install DeepSeek VL2 from local directory
cd DeepSeek-VL2
pip install .
cd ..

# Create directories for logs and checkpoints
echo "Creating directories..."
mkdir -p logs
mkdir -p ckpt

echo "Setup complete!"
echo
echo "To run training, execute these commands in your terminal:"
echo "1. source ~/.bashrc"
echo "2. conda activate grpo-zero"
echo "3. ./run_train.sh --config config_24GB.yaml"
echo
echo "Or simply run: source setup_and_run.sh && ./run_train.sh --config config_24GB.yaml" 