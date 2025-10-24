#!/bin/bash

# Example script to run DDP training with torchrun
# Usage: ./run_ddp.sh [config_file] [num_gpus]

CONFIG_FILE=${1:-"config/nomad_L_gaussian_norm-224.yaml"}
NUM_GPUS=${2:-$(nvidia-smi --list-gpus | wc -l)}
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

echo "Starting DDP training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_FILE"
echo "World size: $NUM_GPUS"

export NCCL_P2P_DISABLE=1

# Run with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    train_ddp.py \
    --config $CONFIG_FILE

echo "Training completed!"


