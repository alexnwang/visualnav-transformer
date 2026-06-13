#!/bin/bash
set -e
SING="singularity exec --nv --overlay /scratch/anw2067/nymeria.sqf:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
PY=/home/anw2067/visualnav-transformer/train/plan_policy_viz.py
SHORT=/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test/viz_shortlists
VIZ=${SHORT}/viz_top50
for pair in hand:tasks_top50_hand.pkl balanced_hand:tasks_top50_balanced_hand.pkl lateral:tasks_top50_lateral.pkl balanced_lateral:tasks_top50_balanced_lateral.pkl; do
  s=${pair%%:*}; f=${pair##*:}
  echo "=== rendering $s ==="
  $SING bash -l -c "conda activate nomad_train2 && python ${PY} --nomad_model draw_mask -N 64 --num_workers 4 --tasks_file ${SHORT}/${f} --log_dir ${VIZ}/${s}"
done
echo "ALL DONE"
