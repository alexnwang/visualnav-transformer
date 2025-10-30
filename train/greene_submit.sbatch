#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|h100
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=nomad
#SBATCH --output=/home/anw2067/visualnav-transformer/slurm_logs/nomad-%j.out
#SBATCH --error=/home/anw2067/visualnav-transformer/slurm_logs/nomad-%j.err
#SBATCH --account=pr_359_tandon_advanced

cd /home/anw2067/visualnav-transformer/train

./torch_run.sh config/greene/minimal-nomad-proprioception-cat8.yaml 2
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat4.yaml 2