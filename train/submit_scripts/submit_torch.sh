#!/bin/bash

CONFIG_FILE=${1}

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE argument is required"
    echo "Usage: $0 <CONFIG_FILE>"
    exit 1
fi
source activate nomad_train
cd /home/anw2067/visualnav-transformer/train

# Race tag: shared across siblings so first-to-start can scancel the rest.
RACE_TAG="nomad-$(date +%s)-$$"

submit_one() {
    local gpu_type=$1
    local num_gpus=$2
    local cpus=$3
    local mem_gb=$4

    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --constraint=${gpu_type}
#SBATCH --time=36:00:00
#SBATCH --mem=${mem_gb}GB
#SBATCH --job-name=${RACE_TAG}
#SBATCH --output=/home/anw2067/slurm_logs/nomad-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/nomad-%j.err
#SBATCH --account=torch_pr_230_tandon_advanced

scancel --state=PENDING --jobname=${RACE_TAG} -u \$USER

cd /home/anw2067/visualnav-transformer/train
singularity exec --nv --overlay /scratch/anw2067/nymeria.sqf:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif bash -l -c "conda activate nomad_train && ./torch_run.sh ${CONFIG_FILE} ${num_gpus}"
EOF
}

# gpu_type  num_gpus  cpus  mem_gb
submit_one l40s 4 64 400
submit_one a100 4 64 400
submit_one h100 2 40 400
