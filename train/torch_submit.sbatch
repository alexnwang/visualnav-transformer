#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --time=36:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=nomad
#SBATCH --output=/home/anw2067/slurm_logs/proprioception_L_gaussian_norm-224-ctxt1-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/proprioception_L_gaussian_norm-224-ctxt1-%j.err

cd /home/anw2067/visualnav-transformer/train

# ./torch_run.sh config/torch/minimal-nomad-lr1e-3.yaml 4
# ./torch_run.sh config/torch/minimal-nomad-4_12-alpha0.yaml 4
# ./torch_run.sh config/torch/minimal-nomad-4_12.yaml 4