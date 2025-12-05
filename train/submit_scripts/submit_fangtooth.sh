#!/bin/bash

CONFIG_FILE=${1}
NUM_GPUS=${2:-$(nvidia-smi --list-gpus | wc -l)}

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE argument is required"
    echo "Usage: $0 <CONFIG_FILE> [NUM_GPUS]"
    exit 1
fi
source activate nomad_train
cd /home/alexnwang/visualnav-transformer/train

./run.sh ${CONFIG_FILE} ${NUM_GPUS}