#!/bin/bash
SING="singularity exec --nv --overlay /scratch/anw2067/nymeria.sqf:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
SHORT=/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test/viz_shortlists
$SING bash -l -c "conda activate nomad_train2 && python -c \"from PIL import Image; im=Image.open('/scratch/anw2067/nymeria_visibility_matrix/20230817_s0_brittney_powell_act1_tdosac/images/3425.jpg'); print('NATIVE JPEG SIZE:', im.size)\""
$SING bash -l -c "conda activate nomad_train2 && python /home/anw2067/visualnav-transformer/train/plan_policy_viz.py --nomad_model draw_mask -N 64 --num_workers 2 --tasks_file ${SHORT}/tasks_tdosac_s3425_s3426.pkl --log_dir /home/anw2067/visualnav-transformer/train/logs/policy_viz/vicinity_tdosac"
echo "ALL DONE"
