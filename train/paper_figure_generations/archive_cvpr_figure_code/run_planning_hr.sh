#!/bin/bash
# Render high-res planning figures for the talk8 set, both algos.
# Reuses the FINISHED plan_cem.py runs (saved mu_history) -> skips the ~7h CEM search,
# only re-rolls the world model + renders on the 1408 frame.
# Runs on a GPU node, inside singularity with the nymeria.sqf overlay (224 RGB frames live there).
set -u
SIF=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
OVERLAY=/scratch/anw2067/nymeria.sqf
TRAIN=/home/anw2067/visualnav-transformer/train
TASKS=$TRAIN/data_splits/nymeria/test/viz_shortlists/talk_8tasks.pkl
SCRIPT=$TRAIN/paper_figure_generations/figure_planning_highres.py
WP_SRC="$TRAIN/logs/cem/2026_05_31_02_49_18:waypoint_cem-h1-n64-t8-v0.3-o8-N64-ds64-dist8-8"
PV_SRC="$TRAIN/logs/cem/2026_05_31_02_49_57:peva_cem-h8-n64-t8-v0.5-o8-N1-ds64-dist8-8"

run() {
  singularity exec --nv --overlay ${OVERLAY}:ro ${SIF} bash -lc \
    "conda activate nomad_train2 && python $SCRIPT $*"
}

echo "######## WAYPOINT ########"
run -a waypoint --tasks_file "$TASKS" --source_log_dir "$WP_SRC" \
    -o 8 -H 1 -n 64 -t 8 -v 0.3 -N 64 -R 64 \
    --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask

echo "######## PEVA ########"
run -a peva --tasks_file "$TASKS" --source_log_dir "$PV_SRC" \
    -o 8 -H 8 -n 64 -t 8 -v 0.5 -N 1 \
    --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask

echo "ALL_PLANNING_HR_DONE"
