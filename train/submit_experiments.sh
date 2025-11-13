#!/bin/bash

# 11/12/2025 trying to improve unpooled representations using 3dposembed and pooling only the current/goal observations
./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3-pool_curr_obs.yaml 4
./submit_scripts/submit_torch.sh config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3.yaml 4

################################
## Greene experiments         ##
################################
# 11/11/2025 trying to improve unpooled representations using 3dposembed
# ./torch_run.sh config/greene/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr1e-3.yaml 2
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