#!/bin/bash

source activate nomad_train2
cd /home/anw2067/visualnav-transformer/train

SLURM_HEADER="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --output=/home/anw2067/slurm_logs/cem_final/plan_cem-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/cem_final/plan_cem-%j.err
#SBATCH --account=torch_pr_230_tandon_advanced
"

########################################################
# 01/27 CEM heldout environments
########################################################
world_size=2
for rank in $(seq 0 $((world_size - 1))); do
sbatch <<EOF
${SLURM_HEADER}
#SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
cd /home/anw2067/visualnav-transformer/train
python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask_heldout --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
EOF
done
########################################################
# 03/26 2 runs to get a sense of waypoint vs waypoint_point3d
########################################################
# rank=0
# world_size=1
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle
# EOF

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle
# EOF

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle
# EOF

########################################################
# 03/26 Slight misfire run of different lengths
########################################################
# for distance in 8 12 16 20; do
# world_size=1
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done
# done

########################################################
# 01/27 CEM heldout environments
########################################################
# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask_heldout --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF
# done


########################################################
# 01/27 CEM heldout environments
########################################################
# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model heldout --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-heldout-o6-n8-t2-v0.05-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model heldout --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF
# done

########################################################
# 01/26 CEM n=8 lol because 16 performed too well
########################################################

# # N = 8
# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-draw_mask-o6-n8-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n8-t2-v0.05-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF
# done

########################################################
# 01/26 CEM higher peva variance
########################################################
# N = 64
# world_size=5
# for rank in $(seq 0 $((world_size - 1))); do

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n64-t2-v0.15-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.15 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 40 --shuffle
# EOF
# done

# # N = 32
# world_size=4
# for rank in 0 1 2 3; do

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n32-t2-v0.15-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 32 -t 2 -v 0.15 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 50 --shuffle
# EOF
# done

# # N = 16
# world_size=2
# for rank in 0 1; do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n16-t2-v0.15-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.15 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF
# done

###########################
# 01/24 CEM performance sweep
###########################
# N = 64
# world_size=5
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-draw_mask-o6-n64-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 40 --shuffle
# EOF

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n64-t2-v0.05-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 40 --shuffle
# EOF
# done

# N = 32
# world_size=4
# for rank in 0 1 2 3; do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-draw_mask-o6-n32-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 32 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 50 --shuffle
# EOF

# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n32-t2-v0.05-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 32 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 50 --shuffle
# EOF
# done

# N = 16
# world_size=2
# for rank in 0 1; do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-draw_mask-o6-n16-t4-v0.3-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-draw_mask-o6-n16-t2-v0.05-N64-ds64-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --rank ${rank} --world_size ${world_size} --num_samples_to_plan 100 --shuffle
# EOF
# done

###########################
# 01/21 less compute with gravity preserving model 
###########################

# for opt_steps in 2 4; do
#     for num_samples in 64 100; do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-gravity-o${opt_steps}-n${num_samples}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o ${opt_steps} -H 1 -n ${num_samples} -t 4 -v 0.3 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 --nomad_model gravity
# EOF
#     done 
# done

###########################
# 01/16 less compute2
###########################

# for opt_steps in 2 4; do
#     for num_samples in 64 100; do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o${opt_steps}-n${num_samples}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o ${opt_steps} -H 1 -n ${num_samples} -t 4 -v 0.3 -N 64 --peva_context_size 8 --peva_diffusion_steps 64
# EOF
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o${opt_steps}-n${num_samples}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o ${opt_steps} -H 8 -n ${num_samples} -t 2 -v 0.05 -N 64 --peva_context_size 8 --peva_diffusion_steps 64
# EOF
#     done 
# done

###########################
# 01/16 goal timestep offset
###########################

# for goal_timestep_offset in 8 12 15; do
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-H1-gt${goal_timestep_offset}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 8 -H 1 -n 100 -t 4 -v 0.3 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 --goal_timestep_offset ${goal_timestep_offset}
# EOF
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-H2-gt${goal_timestep_offset}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 4 -H 2 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 --goal_timestep_offset ${goal_timestep_offset}
# EOF
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-H8-gt${goal_timestep_offset}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 8 -H 8 -n 100 -t 2 -v 0.05 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 --goal_timestep_offset ${goal_timestep_offset}
# EOF
# sbatch <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-H${goal_timestep_offset}-gt${goal_timestep_offset}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 4 -H ${goal_timestep_offset} -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 --goal_timestep_offset ${goal_timestep_offset}
# EOF
# done 
###########################
# older runs
###########################

# for topk in 2; do
#     for v in 0.005 0.01 0.05; do

# sbatch <<EOF
# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=h200
# #SBATCH --time=24:00:00
# #SBATCH --mem=80GB
# #SBATCH --job-name=plan-peva_cem-t${topk}-v${v}
# #SBATCH --output=/home/anw2067/slurm_logs/plan_cem-%j.out
# #SBATCH --error=/home/anw2067/slurm_logs/plan_cem-%j.err
# #SBATCH --account=torch_pr_230_tandon_advanced

# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -H 8 -n 100 -t ${topk} -v ${v} -N 64 --peva_context_size 8 --peva_diffusion_steps 64 -a peva
# EOF
# done
# done

# for topk in 2 4; do
#     for horizon in 1; do
#         for diffusion_steps in 64; do
#             for v in 0.3 ; do

# if [[ "$diffusion_steps" == "128" && "$horizon" == "2" ]]; then
#     continue
# fi

# sbatch <<EOF
# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=h200
# #SBATCH --time=24:00:00
# #SBATCH --mem=80GB
# #SBATCH --job-name=plan_cem-t${topk}-v${v}-h${horizon}-d${diffusion_steps}
# #SBATCH --output=/home/anw2067/slurm_logs/plan_cem-%j.out
# #SBATCH --error=/home/anw2067/slurm_logs/plan_cem-%j.err
# #SBATCH --account=torch_pr_230_tandon_advanced

# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -H ${horizon} -n 100 -t ${topk} -v ${v} -N 64 --peva_context_size 8 --peva_diffusion_steps ${diffusion_steps} -a waypoint
# EOF
# done
# done
# done
# done

# sbatch <<EOF
# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=h200
# #SBATCH --time=12:00:00
# #SBATCH --mem=80GB
# #SBATCH --job-name=plan_cem-CHEATMETRIC_leafxyz_as_cost-H1-n64-t4-v0.3
# #SBATCH --output=/home/anw2067/slurm_logs/plan_cem-%j.out
# #SBATCH --error=/home/anw2067/slurm_logs/plan_cem-%j.err
# #SBATCH --account=torch_pr_230_tandon_advanced

# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 -a waypoint --use_leafxyz_as_cost
# EOF

# sbatch <<EOF
# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=h200
# #SBATCH --time=12:00:00
# #SBATCH --mem=80GB
# #SBATCH --job-name=plan_cem-CHEATMETRIC_leafxyz_as_cost-H8-n64-t2-v0.05
# #SBATCH --output=/home/anw2067/slurm_logs/plan_cem-%j.out
# #SBATCH --error=/home/anw2067/slurm_logs/plan_cem-%j.err
# #SBATCH --account=torch_pr_230_tandon_advanced

# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 8 --peva_diffusion_steps 64 -a peva --use_leafxyz_as_cost
# EOF