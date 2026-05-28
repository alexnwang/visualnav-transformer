#!/bin/bash

# One-off submit: 2x A100, 16 cpu/gpu, 100GB/gpu. No race tag.

CONFIG_FILE=${1}

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE argument is required"
    echo "Usage: $0 <CONFIG_FILE>"
    exit 1
fi
source activate nomad_train
cd /home/anw2067/visualnav-transformer/train

sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --time=36:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=nomad-a100x2
#SBATCH --output=/home/anw2067/slurm_logs/nomad-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/nomad-%j.err
#SBATCH --account=torch_pr_230_tandon_advanced

cd /home/anw2067/visualnav-transformer/train
singularity exec --nv \
  --overlay /scratch/anw2067/nymeria.sqf:ro \
  --overlay /scratch/anw2067/egodex_processed_v2_clean.sqf:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  bash -l -c "conda activate nomad_train && ./torch_run.sh ${CONFIG_FILE} 2"
EOF
