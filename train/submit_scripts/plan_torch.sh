#!/bin/bash

source activate nomad_train2
cd /home/anw2067/visualnav-transformer/train

SING="singularity exec --nv --overlay /scratch/anw2067/nymeria.sqf:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"

SLURM_HEADER="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100
#SBATCH --time=8:00:00
#SBATCH --mem=100GB
#SBATCH --output=/home/anw2067/slurm_logs/cem_final/plan_cem-%j.out
#SBATCH --error=/home/anw2067/slurm_logs/cem_final/plan_cem-%j.err
#SBATCH --account=torch_pr_230_tandon_advanced
"


# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 100 --shuffle
# python plan_cem.py -a waypoint3d -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank 0 --world_size 1 --num_samples_to_plan 100 --shuffle

# singularity exec --nv --overlay /scratch/anw2067/nymeria.sqf:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif bash -l -c "conda activate nomad_train2 && python plan_cem_viz.py -a waypoint -n 128 -o 16 -t 8 -v 0.5 -R 32 -H 1 --peva_diffusion_steps 250 --nomad_model draw_mask --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"

########################################################
# 04/26, rerender_cem_viz local one-liner — best mu only (MJE + WP), SRC_B + SRC_C
########################################################

CEM_VIZ_ROOT=/scratch/anw2067/nomad-logs/cem_viz
RERENDER_ARGS="--steps best --peva_context_size 7 --nomad_model draw_mask"
${SING} bash -l -c "conda activate nomad_train2 && cd /home/anw2067/visualnav-transformer/train && python paper_figure_generations/rerender_cem_viz.py --source_log_dir ${CEM_VIZ_ROOT}/2026_04_18_12_49_12:viz_waypoint_cem-h1-n8-t8-v0.5-o12-R128-ds64-visds250-dist8-8:ws2-r0 --tasks 0.887_20230724_s1_justin_heath_act0_5gtnkm-s1760-g1768 0.822_20230905_s1_elizabeth_morgan_act3_smhnlg-s2794-g2802 0.609_20231122_s1_harold_copeland_act2_k1ngjh-s1758-g1766 ${RERENDER_ARGS} && python paper_figure_generations/rerender_cem_viz.py --source_log_dir ${CEM_VIZ_ROOT}/2026_04_15_13_19_52:viz_waypoint_cem-h1-n8-t8-v0.5-o12-R128-ds64-visds250-dist8-8 --tasks 0.781_20230817_s1_rebecca_ward_act2_39a7o2-s1124-g1132 ${RERENDER_ARGS}"

########################################################
# 04/26, rerender_cem_viz across three existing waypoint runs (draw_mask, ctx6) — sbatch
########################################################

# CEM_VIZ_ROOT=/scratch/anw2067/nomad-logs/cem_viz

# SRC_A="${CEM_VIZ_ROOT}/2026_04_18_18_49_47:viz_waypoint_cem-h1-n8-t8-v0.5-o12-R128-ds64-visds250-dist8-8:ws2-r1"
# TASKS_A=(
#     0.390_20230817_s1_rebecca_ward_act2_39a7o2-s1124-g1132
#     0.374_20230829_s0_ray_humphrey_act4_7lkmhe-s2687-g2695
# )
# sbatch --time=4:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=rerender_cem_viz-srcA-2tasks
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python paper_figure_generations/rerender_cem_viz.py --source_log_dir ${SRC_A} --tasks ${TASKS_A[@]} --steps all --peva_context_size 7 --nomad_model draw_mask"
# EOF

# SRC_B="${CEM_VIZ_ROOT}/2026_04_18_12_49_12:viz_waypoint_cem-h1-n8-t8-v0.5-o12-R128-ds64-visds250-dist8-8:ws2-r0"
# TASKS_B=(
#     0.822_20230905_s1_elizabeth_morgan_act3_smhnlg-s2794-g2802
#     0.609_20231122_s1_harold_copeland_act2_k1ngjh-s1758-g1766
# )
# sbatch --time=4:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=rerender_cem_viz-srcB-3tasks
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python paper_figure_generations/rerender_cem_viz.py --source_log_dir ${SRC_B} --tasks ${TASKS_B[@]} --steps all --peva_context_size 7 --nomad_model draw_mask"
# EOF

# SRC_C="${CEM_VIZ_ROOT}/2026_04_15_13_19_52:viz_waypoint_cem-h1-n8-t8-v0.5-o12-R128-ds64-visds250-dist8-8"
# TASKS_C=(
#     0.781_20230817_s1_rebecca_ward_act2_39a7o2-s1124-g1132
# )
# sbatch --time=4:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=rerender_cem_viz-srcC-1task
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python paper_figure_generations/rerender_cem_viz.py --source_log_dir ${SRC_C} --tasks ${TASKS_C[@]} --steps all --peva_context_size 7 --nomad_model draw_mask"
# EOF

########################################################
# 04/21, plan_cem_viz waypoint draw_mask dist8 on 3 cherry-picked tasks (ctx6, matches peva viz split)
########################################################

# TARGET_TRACKS=(
#     20230817_s1_rebecca_ward_act2_39a7o2-s1124-g1132
#     20230905_s1_elizabeth_morgan_act3_smhnlg-s2794-g2802
#     20231122_s1_harold_copeland_act2_k1ngjh-s1758-g1766
# )
# sbatch --time=4:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=cem_viz-waypoint-targeted-n8-o12-R128-ctx6
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem_viz.py -a waypoint -n 8 -o 12 -t 8 -v 0.5 -R 128 -H 1 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask --min_dist_cat 8 --max_dist_cat 8 --target_tracks ${TARGET_TRACKS[@]}"
# EOF

########################################################
# 04/20, heldout eval: peva + waypoint(heldout) + waypoint(draw_mask_heldout), n8 64 tasks
########################################################

# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-heldout-o6-n8-H8-t2-v0.05-N64-ds64-dist8-64tasks
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model heldout --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-H1-t4-v0.3-N64-ds64-dist8-64tasks
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model heldout --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-draw_mask_heldout-o6-n8-H1-t4-v0.3-N64-ds64-dist8-64tasks
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask_heldout --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

########################################################
# 04/20, waypoint_cem 128 tasks single gpu (h200 only, 48h)
########################################################

# sbatch --time=48:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --constraint=h200
# #SBATCH --job-name=plan-waypoint_cem-o6-n32-H1-t4-v0.3-N64-ds64-dist8-128tasks
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 32 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

########################################################
# 04/19, gt-waypoint policy+PEVA rollout on targeted tracks (no CEM planning)
########################################################

# singularity exec --nv --overlay /scratch/anw2067/nymeria.sqf:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif bash -l -c "conda activate nomad_train2 && python paper_figure_generations/gt_waypoint_policy_rollouts.py --nomad_model draw_mask --peva_context_size 7 --min_dist_cat 8 --max_dist_cat 8 -N 64 --use_peva --peva_diffusion_steps 64 --target_tracks 20231009_s0_clayton_bradley_act0_8iksyy-794 20231019_s0_douglas_martin_act1_n6a4yk-3375 20231018_s0_scott_hutchinson_act3_46oe4h-1008 20231019_s0_douglas_martin_act3_rsqq7a-56 20231027_s1_stacie_cross_act2_kijh3i-431 20231027_s1_stacie_cross_act2_kijh3i-753 20231110_s0_thomas_brown_act3_pisdac-591 20231110_s0_thomas_brown_act3_pisdac-1387 20231113_s0_patricia_gutierrez_act6_209mth-4338 20231212_s0_paul_arellano_act3_oj31oo-2967 --no_skeleton_text"
# sbatch --time=2:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=gt_waypoint_rollout-targeted-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python paper_figure_generations/gt_waypoint_policy_rollouts.py --nomad_model draw_mask --peva_context_size 7 --min_dist_cat 8 --max_dist_cat 8 -N 64 --use_peva --peva_diffusion_steps 64 --target_tracks ${TARGET_TRACKS[@]}"
# EOF

########################################################
# 04/19, extend mje/dreamsim_vs_cem_iterations to 200 tasks (72 more after 128)
########################################################

# n8 dist8 (ws=1, tasks 128..199 = 72 more), waypoint + waypoint_point3d + peva
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n8-H1-t4-v0.3-N64-ds64-dist8-ext72
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 200 --skip_tasks 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n8-H1-t4-v0.3-N64-ds64-dist8-ext72
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank 0 --world_size 1 --num_samples_to_plan 200 --skip_tasks 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H8-t2-v0.05-N64-ds64-dist8-ext72
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 200 --skip_tasks 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# n16 dist8 (ws=1, tasks 128..199 = 72 more), waypoint + waypoint_point3d + peva
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n16-H1-t4-v0.3-N64-ds64-dist8-ext72
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 200 --skip_tasks 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n16-H1-t4-v0.3-N64-ds64-dist8-ext72
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank 0 --world_size 1 --num_samples_to_plan 200 --skip_tasks 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=28:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8-ext72
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 200 --skip_tasks 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# n64 dist8 (ws=2, skip 64 + 36 per rank = 72 more total), waypoint + waypoint_point3d + peva
# for rank in 0 1; do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-ext72-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 2 --num_samples_to_plan 100 --skip_tasks 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-ext72-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank ${rank} --world_size 2 --num_samples_to_plan 100 --skip_tasks 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-ext72-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 2 --num_samples_to_plan 100 --skip_tasks 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/17, gt_waypoint_policy_rollouts with PEVA WM rollout, draw_mask dist8
########################################################

# sbatch --time=8:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=gt_wp_policy_rollouts-draw_mask-N64-ds32-dist8-peva
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python paper_figure_generations/grt_waypoint_policy_rollouts.py --nomad_model draw_mask -N 64 --peva_context_size 7 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8 --use_peva"
# EOF

########################################################
# 04/17, finish peva n16 dist8 (resume last 23 tasks: partial idx 105 + 22 remaining)
########################################################

# sbatch --time=8:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8-resume23
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --skip_tasks 105 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

########################################################
# 04/18, plan_cem_viz ws=2 peva draw_mask dist8
########################################################

# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=12:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=cem_viz-peva-ws${world_size}-n8-o12-R128-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem_viz.py -a peva -n 8 -o 12 -t 2 -v 0.05 -R 128 -H 8 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/17, plan_cem_viz ws=2 waypoint draw_mask dist8
########################################################

# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=12:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=cem_viz-ws${world_size}-n8-o12-R128-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem_viz.py -a waypoint -n 8 -o 12 -t 8 -v 0.5 -R 128 -H 1 -N 64 --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/16, plan_policy_only draw_mask 200 tasks dist8
########################################################

# sbatch --time=00:30:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan_policy_only-draw_mask-200-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_policy_only.py --nomad_config /home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/config.yaml --nomad_checkpoint /home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth -o 6 -N 64 --num_samples_to_plan 200 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

########################################################
# 04/15, rerun n8/n16/n64 with 128 tasks (off-by-1 fix)
########################################################

# n8 dist8 (ws=1, 128 tasks), waypoint + waypoint_point3d + peva
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n8-H1-t4-v0.3-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n8-H1-t4-v0.3-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H8-t2-v0.05-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# # n16 dist8 (ws=1, 128 tasks), waypoint + waypoint_point3d + peva
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n16-H1-t4-v0.3-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n16-H1-t4-v0.3-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=28:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 128 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# # n64 dist8 (ws=4, 32 per rank = 128 total), waypoint + waypoint_point3d + peva
# for rank in 0 1 2 3; do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 4 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank ${rank} --world_size 4 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 4 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/15, cherry-pick best CEM viz, 6 workers x 32 tasks
########################################################

# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=12:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=cem_viz-n8-o16-R128-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem_viz.py -a waypoint -n 8 -o 12 -t 8 -v 0.5 -R 128 -H 1 -N 64 --peva_diffusion_steps 64 --peva_vis_diffusion_steps 250 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/13, n8 dist8 200 tasks (ws=2, 100 per rank), waypoint + peva
########################################################

# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=12:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n8-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=12:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/13, rerun n16 (ws=1) and n64 (ws=2) with correct num_samples_to_plan
########################################################

# n16 dist8, waypoint + peva (world_size=1, 64 samples total)
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n16-H1-t4-v0.3-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# n64 dist8, waypoint + peva (world_size=2, 32 samples per rank = 64 total)
# for rank in 0 1; do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 2 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_point3d_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint_point3d -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model 3d_mask --rank ${rank} --world_size 2 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 2 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

# for rank in 1; do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 2 --num_samples_to_plan 32 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/11, resubmit failed peva ranks 2,3,4 (OpenGL render crash)
########################################################

# # n64 dist8, peva only — ranks 2,3,4 (world_size=5)
# for rank in 2 3 4; do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size 5 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

########################################################
# 04/10, resubmit ds128 dist8 peva (OOM in OpenGL render)
########################################################

# ds128 dist8, peva only
# sbatch --time=32:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H8-t2-v0.05-N64-ds128-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 128 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

########################################################
# 04/10, resubmit failed jobs (inode issue) with singularity overlay
########################################################

# n64 dist8, waypoint + peva
# world_size=5
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a waypoint -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF
# done

# # n16 dist8, peva only (rank0)
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8-rank0
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

# # ds128 dist8, peva only
# sbatch --time=32:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H8-t2-v0.05-N64-ds128-dist8
# cd /home/anw2067/visualnav-transformer/train
# ${SING} bash -l -c "conda activate nomad_train2 && python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 128 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8"
# EOF

########################################################
# Everything above this line uses ${SING} (singularity overlay).
# Everything below used direct data access (/scratch/anw2067/nymeria).
########################################################

########################################################
# 04/08, n64 peva and waypoint for dist8
########################################################
# world_size=5
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n64-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n64-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 64 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8
# EOF
# done

########################################################
# 04/08, n16 peva and waypoint for dist8
########################################################
# world_size=2
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n16-H1-t4-v0.3-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8
# EOF
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n16-H8-t2-v0.05-N64-ds64-dist8-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 16 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8
# EOF
# done

# ########################################################
# # 04/08, dist8 with 128 diffusion steps
# ########################################################
# sbatch --time=32:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n8-H1-t4-v0.3-N64-ds128-dist8
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 128 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8
# EOF
# sbatch --time=32:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H8-t2-v0.05-N64-ds128-dist8
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H 8 -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 128 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat 8 --max_dist_cat 8
# EOF

# ########################################################
# # 04/08, peva and waypoint planning for distances 6 10 14 18
# ########################################################
# for distance in 6; do # 10 14 18; do
# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-o6-n8-H1-t4-v0.3-N64-ds64-dist${distance}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 8 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# sbatch --time=28:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-o6-n8-H${distance}-t2-v0.05-N64-ds64-dist${distance}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H ${distance} -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done

########################################################
# 04/01, opt_steps=0 initial baseline (dreamsim_init only, no CEM)
########################################################
# for distance in 8 12 16 20; do
# sbatch --time=2:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-init-o0-N64-ds64-dist${distance}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 0 -H 1 -n 1 -t 1 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# sbatch --time=2:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-init-o0-N64-ds64-dist${distance}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 0 -H ${distance} -n 1 -t 1 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank 0 --world_size 1 --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done

########################################################
# 03/31, rerun some and add dist8 n16
########################################################
# for distance in 8; do
# world_size=1
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n16-H1-t4-v0.3-N64-ds64-dist${distance}-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done
# done

# for distance in 20; do
# world_size=1
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-heldout-o6-n8-H8-t2-v0.05-N64-ds64-dist${distance}-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H ${distance} -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done
# done

########################################################
# 03/29, lets run a long horizon waypoint planning
########################################################
# world_size=1
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=24:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n8-H2-t4-v0.3-N64-ds250-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 2 -n 64 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 250 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat 16 --max_dist_cat 16
# EOF
# done

########################################################
# 03/29, lets run a long horizon peva planning and a few more samples for waypoint to see if it improves.
########################################################
# for distance in 12 16 20; do
# world_size=1
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-peva_cem-heldout-o6-n8-H8-t2-v0.05-N64-ds64-dist${distance}-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a peva -o 6 -H ${distance} -n 8 -t 2 -v 0.05 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done
# done

# for distance in 12 16 20; do
# world_size=1
# for rank in $(seq 0 $((world_size - 1))); do
# sbatch --time=16:00:00 <<EOF
# ${SLURM_HEADER}
# #SBATCH --job-name=plan-waypoint_cem-heldout-o6-n16-H1-t4-v0.3-N64-ds64-dist${distance}-rank${rank}
# cd /home/anw2067/visualnav-transformer/train
# python plan_cem.py -a waypoint -o 6 -H 1 -n 16 -t 4 -v 0.3 -N 64 --peva_context_size 7 --peva_diffusion_steps 64 --nomad_model draw_mask --rank ${rank} --world_size ${world_size} --num_samples_to_plan 64 --shuffle --min_dist_cat ${distance} --max_dist_cat ${distance}
# EOF
# done
# done


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