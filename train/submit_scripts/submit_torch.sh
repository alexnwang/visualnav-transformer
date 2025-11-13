#!/bin/bash

CONFIG_FILE=${1}
NUM_GPUS=${2:-$(nvidia-smi --list-gpus | wc -l)}

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE argument is required"
    echo "Usage: $0 <CONFIG_FILE> [NUM_GPUS]"
    exit 1
fi
source activate nomad_train
cd /home/anw2067/visualnav-transformer/train

sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=$((16 * NUM_GPUS))
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --constraint=l40s
#SBATCH --time=36:00:00
#SBATCH --mem=$((100 * NUM_GPUS))GB
#SBATCH --job-name=nomad
#SBATCH --output=/home/anw2067/slurm_logs/nomad-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/nomad-%j.err
#SBATCH --account=torch_pr_230_tandon_advanced

cd /home/anw2067/visualnav-transformer/train
./torch_run.sh ${CONFIG_FILE} ${NUM_GPUS}
EOF