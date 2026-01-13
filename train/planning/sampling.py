import copy
import json

from diffusers.models import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dreamsim import dreamsim
import torch
from torchvision import transforms
from torchvision.utils import save_image
import wandb
import yaml
import numpy as np
import math
from peva.diffusion import create_diffusion
from peva.models import CDiT_models
from planning.cem import CEMPlanner
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.data.vint_dataset import ViNT_Nymeria_Dataset
from vint_train.models.nomad.conditional_uned1dnomad import ConditionalUnet1D_NoMaD
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.nymeria_training_utils import (
    unnormalize_data_smpl_pose_gaussian,
)
from planning.utils import _compute_pose_and_loss
from vint_train.training.nymeria_training_utils import get_action_smpl_torch, normalize_data_smpl_pose
from planning.utils import draw_waypoints

def waypoint_sample(policy_model, policy_diffusion,
                    peva_model, peva_diffusion, peva_vae, peva_stats,
                    waypoints, context_poses, curr_obs, goal_obs,
                    policy_pred_horizon, policy_action_dim,
                    image_size, 
                    policy_context_size, peva_context_size, peva_latent_size,
                    device,
                    skip_last_peva=False,
                    gt_deltas=None,
                    first_pose=None):
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    wm_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    B, W = waypoints.shape[:2]
    delta_accum = torch.zeros(B, W, policy_pred_horizon, policy_action_dim, device=device) # B, W*8, 48
    goal_obs_accum = torch.zeros(B, W, 3, image_size, image_size, device=device)
    if skip_last_peva:
        gen_frames_accum = torch.zeros(B, (W-1), policy_pred_horizon, 3, image_size, image_size, device=device)
    else:
        gen_frames_accum = torch.zeros(B, W, policy_pred_horizon, 3, image_size, image_size, device=device)
        
    wm_obs = wm_norm(curr_obs.flatten(0, 1)).unflatten(0, (B, -1))
    for w in range(W):
        policy_obs = imagenet_norm(curr_obs[:, -policy_context_size:].flatten(0, 1)).unflatten(0, (B, policy_context_size))
        # goal_obs = imagenet_norm(curr_obs[:, -1])
        goal_obs_accum[:, w] = goal_obs = draw_waypoints(curr_obs[:, -1], waypoints[:, w])
        goal_obs = imagenet_norm(goal_obs)
        # goal_obs = imagenet_norm(goal_obs)
        
        deltas = policy_sample(policy_model, policy_diffusion,
                    policy_obs, goal_obs,
                    context_poses[:, :policy_context_size], 
                    policy_pred_horizon, policy_action_dim, device) # B, 8, 48
        delta_accum[:, w] = deltas
    
        if skip_last_peva and w == W-1:
            continue 
        
        # print("pre_peva_norm", torch.abs(deltas).sum(1).sum(1))
        peva_normalized_deltas = normalize_data_smpl_pose(deltas, peva_stats)
        # peva_normalized_deltas = deltas
        # print("post_peva_norm", torch.abs(peva_normalized_deltas).sum(1).sum(1))
        for t in range(policy_pred_horizon):
            x_cond = torch.zeros(curr_obs.shape[0], peva_context_size+1, curr_obs.shape[2], curr_obs.shape[3], curr_obs.shape[4], device=device)
            x_cond[:, :peva_context_size] = wm_obs[:, -peva_context_size:]
            
            curr_delta = peva_normalized_deltas[:, t:t+1].repeat(1, peva_context_size, 1,) 
            
            rel_const = 1. / (64-(-64))  # distance is set in eval, but fixed to [8, 8] for now.
            rel_const = rel_const * 1 # multiply the rel_const by rollout_stride 
            rel_t = (torch.ones(curr_obs.shape[0], peva_context_size, device=device) * rel_const)
            x_pred = model_forward_wrapper(
                (peva_model, peva_diffusion, peva_vae),
                x_cond,
                curr_delta,
                peva_latent_size,
                device,
                num_cond=curr_obs.shape[1],
                rel_t=rel_t,
                progress=True
            )
            x_pred = x_pred[:, None] # B, 1, 3, H, W
            wm_obs = torch.cat([wm_obs, x_pred], dim=1)
            x_pred_unnorm = x_pred * 0.5 + 0.5
            gen_frames_accum[:, w, t] = x_pred_unnorm[:, 0]
            curr_obs = torch.cat([curr_obs[:, 1:], x_pred_unnorm], dim=1)
    
    return gen_frames_accum, delta_accum, goal_obs_accum

@torch.no_grad()
def model_forward_wrapper(all_models, curr_obs, curr_delta, latent_size, device, num_cond, rel_t=None, progress=False):
    model, diffusion, vae = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)
    
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            raise ValueError("Not implemented")

        x = x.flatten(0,1)
        y = y.flatten(0, 1)
        rel_t = rel_t.flatten(0, 1)
        
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond]
        z = torch.randn(x.shape[0], 4, latent_size, latent_size, device=device)
        t_cond = torch.zeros(x.shape[0], x_cond.shape[1], device=device)
        model_kwargs = dict(y=y, rel_t=rel_t, num_cond=num_cond, x_cond=x_cond, t_cond=t_cond, x_clean=x.flatten(0, 1))
        samples = diffusion.p_sample_loop(model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device) # B, 4, latent_size, latent_size; samples a single image
        samples = vae.decode(samples / 0.18215).sample # B, 3, image_size, image_size 
        return torch.clip(samples, -1., 1.)
    
def peva_sample(peva_model, peva_diffusion, peva_vae, peva_stats,
                curr_obs, deltas,
                peva_context_size, peva_latent_size,
                image_size, device):
    wm_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    B, T = deltas.shape[:2]
    gen_frames_accum = torch.zeros(B, T, 3, image_size, image_size, device=device)
    wm_obs = wm_norm(curr_obs.flatten(0, 1)).unflatten(0, (B, -1))
    peva_normalized_deltas = normalize_data_smpl_pose(deltas, peva_stats)
    for t in range(T):
        x_cond = torch.zeros(curr_obs.shape[0], peva_context_size+1, curr_obs.shape[2], curr_obs.shape[3], curr_obs.shape[4], device=device)
        x_cond[:, :peva_context_size] = wm_obs[:, -peva_context_size:]
        
        curr_delta = peva_normalized_deltas[:, t:t+1].repeat(1, peva_context_size, 1,) 
        
        rel_const = 1. / (64-(-64))  # distance is set in eval, but fixed to [8, 8] for now.
        rel_const = rel_const * 1 # multiply the rel_const by rollout_stride 
        rel_t = (torch.ones(curr_obs.shape[0], peva_context_size, device=device) * rel_const)
        x_pred = model_forward_wrapper(
            (peva_model, peva_diffusion, peva_vae),
            x_cond,
            curr_delta,
            peva_latent_size,
            device,
            num_cond=curr_obs.shape[1],
            rel_t=rel_t,
            progress=True
        )
        x_pred = x_pred[:, None] # B, 1, 3, H, W
        wm_obs = torch.cat([wm_obs, x_pred], dim=1)
        x_pred_unnorm = x_pred * 0.5 + 0.5 # B, 1, 3, H, W
        gen_frames_accum[:, t] = x_pred_unnorm[:, 0]
        curr_obs = torch.cat([curr_obs[:, 1:], x_pred_unnorm], dim=1)
    return gen_frames_accum, deltas

@torch.no_grad()
def policy_sample(
    model: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor, # or None
    context_poses: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    device: torch.device,
):
    """
    Generate model output (conditioned, unconditioned, distance) for the given batch of images.
    Outputs are DELTAS and are NOT unnormalized or scaled.
    """
    N = batch_obs_images.shape[0]
    timesteps = noise_scheduler.timesteps.to(device)
    
    # assume goal is always provided.
    no_mask = torch.zeros((batch_goal_images.shape[0],), device=device).long()
    obsgoal_cond = model("vision_encoder",
                         obs_img=batch_obs_images,
                         goal_img=batch_goal_images, 
                         input_goal_mask=no_mask,
                         context_poses=context_poses, goal_coordinates=None)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (N, pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output # B, pred_horizon, action_dim
    # print("iter10", torch.abs(diffusion_output).sum(1).sum(1))
    for k in timesteps:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]),
            global_cond=obsgoal_cond,
            goal_pose=None
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
        # print(f"iter{k}", torch.abs(diffusion_output).sum(1).sum(1))
    
    # print("pre_policy_norm", torch.abs(diffusion_output).sum(1).sum(1))
    diffusion_output = unnormalize_data_smpl_pose_gaussian(diffusion_output.flatten(0, 1)).unflatten(0, (N, pred_horizon))
    # print("post_policy_norm", torch.abs(diffusion_output).sum(1).sum(1))
    return diffusion_output