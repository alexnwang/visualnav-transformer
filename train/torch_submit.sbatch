#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --time=70:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=nomad
#SBATCH --output=slurm_logs/proprioception_L_gaussian_norm-224-ctxt1-%j.out
#SBATCH --error=slurm_logs/proprioception_L_gaussian_norm-224-ctxt1-%j.err

cd /home/anw2067/visualnav-transformer/train

./torch_run.sh config/torch/proprioception_L_gaussian_norm-224-ctxt1.yaml 2