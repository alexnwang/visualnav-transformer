#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=nomad
#SBATCH --output=/home/anw2067/visualnav-transformer/slurm_logs/nomad-%j.out
#SBATCH --error=/home/anw2067/visualnav-transformer/slurm_logs/nomad-%j.err
#SBATCH --account=pr_359_tandon_advanced

cd /home/anw2067/visualnav-transformer/train

# 11/11/2025 trying to improve unpooled representations using 3dposembed
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3.yaml 2
./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 2

# random baseline
# ./torch_run.sh config/greene/random-cat8-dinov3.yaml 1

# dinov3 unpool full!
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-dinov3_unpool_full-big_transformer.yaml 2
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-r50.yaml 2

# checking how things work after cat8 and preserve_updown and stuff
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8.yaml 2

# proprioceptionv2 + bugged preserve_updown code
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-preserve_updown.yaml 2

# ./torch_run.sh config/greene/minimal-nomad-big_encoder.yaml 2
# ./torch_run.sh config/greene/minimal-nomad-big_transformer.yaml 2
# ./torch_run.sh config/greene/minimal-nomad-big_denoiser.yaml 2

# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8.yaml 2
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat4.yaml 2