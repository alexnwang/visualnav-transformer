#!/bin/bash
set -u
SIF=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
OVERLAY=/scratch/anw2067/nymeria.sqf
TRAIN=/home/anw2067/visualnav-transformer/train
TASKS=$TRAIN/data_splits/nymeria/test/viz_shortlists/talk_8tasks.pkl
SCRIPT=$TRAIN/paper_figure_generations/figure_planning_highres.py
WP_SRC="$TRAIN/logs/cem/2026_05_31_02_49_18:waypoint_cem-h1-n64-t8-v0.3-o8-N64-ds64-dist8-8"
echo "######## WAYPOINT R=256 ########"
singularity exec --nv --overlay ${OVERLAY}:ro ${SIF} bash -lc \
  "conda activate nomad_train2 && python $SCRIPT -a waypoint --tasks_file $TASKS --source_log_dir $WP_SRC --log_dir $TRAIN/logs/planning_figure_hr_R256 -o 8 -H 1 -n 64 -t 8 -v 0.3 -N 64 -R 256 --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask"
echo "WAYPOINT_R256_DONE"
