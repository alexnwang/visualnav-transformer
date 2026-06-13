#!/bin/bash
set -e
HOST=${HOSTNAME%%.*}
LOG="/home/anw2067/ddp_probe_dinov3_${HOST}.log"
cd /home/anw2067/hdp
rm -f "$LOG"
NUM_GPUS=4 ./submit_scripts/run_nymeria.sh train \
    env.data_folder=/scratch/anw2067/nymeria_visibility_matrix \
    env.data_split_train=/home/anw2067/visualnav-transformer/train/data_splits/nymeria/train \
    env.data_split_test=/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test \
    env.gaussian_normalization_stats_path=/home/anw2067/visualnav-transformer/train/nymeria_nomad_mean_var_stats.json \
    env.num_workers=16 \
    method.encoder_backbone=dinov3-s \
    log=false \
    n_epochs=1 \
    eval_freq_epochs=99 \
    save_freq_epochs=99 \
    log_freq=20 \
    >"$LOG" 2>&1
