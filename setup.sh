#!/bin/bash

# Exit on error
set -e

echo "Setting up GRPO-Zero environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda not found. Please install conda first."
    exit 1
fi

# Create and activate conda environment
echo "Creating conda environment..."
conda create -n grpo-zero python=3.10 -y || true
eval "$(conda shell.bash hook)"
conda activate grpo-zero

# Install base dependencies
echo "Installing base dependencies..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge transformers=4.37.2 tensorboard numpy pandas pyyaml

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

# Create a run script that sets up PYTHONPATH
echo "Creating run script..."
cat > run_train.sh << 'EOL'
#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate grpo-zero

# Add DeepSeek VL2 to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/DeepSeek-VL2

# Run training script with provided arguments
python train.py "$@"
EOL

# Make run script executable
chmod +x run_train.sh

echo "Setup complete!"
echo "To run training:"
echo "1. First time: source ~/.bashrc"
echo "2. Then: ./run_train.sh --config config_24GB.yaml" 