#!/bin/bash
set -u
SIF=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
OVERLAY=/scratch/anw2067/nymeria.sqf
TRAIN=/home/anw2067/visualnav-transformer/train
SCRIPT=$TRAIN/paper_figure_generations/figure_planning_highres.py
TASKS=$TRAIN/data_splits/nymeria/test/viz_shortlists/talk_peva_resume3.pkl
PV_SRC="$TRAIN/logs/cem/2026_05_31_02_49_57:peva_cem-h8-n64-t8-v0.5-o8-N1-ds64-dist8-8"
echo "######## PEVA RESUME (tammy/elizabeth/michael) ########"
singularity exec --nv --overlay ${OVERLAY}:ro ${SIF} bash -lc \
  "conda activate nomad_train2 && python $SCRIPT -a peva --tasks_file $TASKS --source_log_dir $PV_SRC -o 8 -H 8 -n 64 -t 8 -v 0.5 -N 1 --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask"
echo "PEVA_RESUME_DONE"
