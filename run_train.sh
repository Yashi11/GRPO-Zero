#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate grpo-zero

# Add DeepSeek VL2 to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/DeepSeek-VL2

# Run training script with provided arguments
python train.py "$@"
