#!/bin/bash
set -e
LOG=/home/anw2067/cond_ablation_diagnostic.log
rm -f "$LOG"
IMAGE="/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
OVERLAY="/scratch/anw2067/nymeria.sqf"
singularity exec --nv --overlay "${OVERLAY}:ro" "${IMAGE}" \
  bash -l -c "conda activate nomad_train2 && export PYTHONPATH=/home/anw2067/visualnav-transformer/train:\${PYTHONPATH:-} && python -u /home/anw2067/cond_ablation_diagnostic.py" \
  >"$LOG" 2>&1
