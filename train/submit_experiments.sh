#!/bin/bash

# 12/04 returning 2d conditiong mask to 50/50 via 2d5050 
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs-goal2d5050.yaml 4
# 12/05 and the drawing model
./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs-goaldraw.yaml 4

# 12/03/2025 2d conditioning model on fangtooth (65495b7ce189755d0ad5bd40ac9173da526aac1f)
# ./submit_scripts/submit_fangtooth.sh config/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs-goal2d.yaml 8

# 12/02/2025 point conditioning model (f05448679025013f022cd544290adb3fa28ec436)
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs-goalPoint.yaml 4

# 12/01/2025 lowered lr (b08c0869545c49b74a409191facea07aabff26b7)
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-cheat.yaml 4 # lower lr because the 1e-3 is fluctuating

# 12/01/2025 cheat model post fix (cdafef8f5383278548a0dede70c06c7ff585689c)
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs-cheat.yaml 4
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4 # fixed the angular distance metrics

# 11/20/2025 running the regression model again, but with fixed angular distance metrics
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat4-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat2-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4


# 11/17/2025 trying to train a regression model with current pose as target
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat8-current_pose.yaml 4

# 11/13/2025 trying to train a cheat model
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs-cheat.yaml 4

# 11/13/2025 trying to train a regression model
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat4-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4
# ./submit_scripts/submit_torch.sh config/torch/regression-proprioception-cat2-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4

# 11/12/2025 trying to improve unpooled representations using 3dposembed and pooling only the current/goal observations
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4
# ./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3.yaml 4

################################
## Greene experiments         ##
################################
# 11/11/2025 trying to improve unpooled representations using 3dposembed
# ./torch_run.sh config/greene/minimal-nomad-propriocepretion-cat8-dinov3_unpool_3dposemb-proj-lr1e-3.yaml 2
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 2

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

################################
## Fangtooth/D4 experiments   ##
################################
#11/09 higher lr
# ./run.sh config/minimal-nomad_proprioception-cat8-dinov3_unpool-nonlinear_proj-tsfmr8x8-lr5e-3.yaml 8
# ./run.sh config/minimal-nomad_proprioception-cat8-dinov3_unpool-nonlinear_proj-lr1e-3.yaml 8

# ./run.sh config/minimal-nomad_proprioception-cat8-dinov3_unpool-nonlinear_proj-tsfmr8x8.yaml 8
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh config/d4-minimal-nomad_proprioception-cat8-dinov3_unpool-nonlinear_proj-tsfmr8x8.yaml 4
# ./run.sh config/minimal-nomad_proprioception-cat8-dinov3_unpool_full.yaml 8
# ./run.sh config/minimal-nomad_proprioception-cat8-r50.yaml 8
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh config/d4-minimal-nomad_proprioception-cat8-dinov3.yaml 4
# ./run.sh config/minimal-nomad_proprioception-cat8-preserve_updown.yaml 8

# # 08/14 nomad-L with 128 image size
# torchrun --nproc_per_node=2 train_ddp.py -c config/nomad_L_gaussian_norm-224.yaml
# CUDA_VISIBLE_DEVICES=5,6 python train.py -c config/nomad_L_gaussian_norm-224.yaml

###########################
## Old torch experiments ##
###########################

# ./torch_run.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3.yaml 4
# ./torch_run.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4

# ./torch_run.sh config/torch/minimal-nomad-proprioception-pred4.yaml 4

# ./torch_run.sh config/torch/minimal-overfit.yaml 4
# ./torch_run.sh config/torch/minimal-overfit-8steps.yaml 4
# ./torch_run.sh config/torch/minimal-overfit-8steps-proprioception.yaml 4
# ./torch_run.sh config/torch/minimal-nomad-proprioception-nopool.yaml 4


# ./torch_run.sh config/torch/minimal-nomad-lr1e-3.yaml 4
# ./torch_run.sh config/torch/minimal-nomad-4_12-alpha0.yaml 4
# ./torch_run.sh config/torch/minimal-nomad-4_12.yaml 4