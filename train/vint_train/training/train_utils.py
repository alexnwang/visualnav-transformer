import argparse
from vint_train.data.misc import XsensSkeleton, XSensConstants
from vint_train.training.nymeria_training_utils import forward_kinematics_wrapper, get_action_smpl_torch, get_delta_smpl, normalize_data_smpl_pose, plot_images_and_actions_full_body, unnormalize_data_smpl_pose, unnormalize_data_smpl_pose_gaussian
import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = torch.from_numpy(np.expand_dims(data_config['action_stats'][key], axis=0))

# for key in data_config['action_stats']:
#     ACTION_STATS[key] = np.array(data_config['action_stats'][key])


# Train utils for NOMAD

def _compute_3d_joint_metrics(
    model_output_dict: Dict[str, torch.Tensor],
    batch_deltas_gt: torch.Tensor,
    action_mask: torch.Tensor,
    first_pose: torch.Tensor,
    xsens_skel: XsensSkeleton,
):
    uc_deltas = model_output_dict['uc_actions']
    gc_deltas = model_output_dict['gc_actions']
    
    # actions == rpy angles for all joints + pelvis xyz
    uc_actions = get_action_smpl_torch(first_pose, uc_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
    gc_actions = get_action_smpl_torch(first_pose, gc_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
    gt_actions = get_action_smpl_torch(first_pose, batch_deltas_gt, XSensConstants.upper_body_num_parts) # B, T, 48
    
    def _compute_pose_and_loss(actn, gt_actn, skel, actn_mask=None):
        gt_xyz, gt_rpy = forward_kinematics_wrapper(gt_actn, skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, num_segments, 3
        pred_xyz, pred_rpy = forward_kinematics_wrapper(actn, skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, num_segments, 3
        res = {}
        for i, body_part_name in enumerate(XSensConstants.part_names[:XSensConstants.upper_body_num_parts]):
            R_gt = R.from_euler('xyz', gt_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
            R_pred = R.from_euler('xyz', pred_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
            ang_dist = torch.from_numpy((R_gt.inv() * R_pred).magnitude() / np.pi * 180).to(actn.device).float() # B
            xyz_dist = torch.norm(gt_xyz[:, i, :] - pred_xyz[:, i, :], dim=-1) # B
            if actn_mask is not None:
                ang_dist = ang_dist * actn_mask
                xyz_dist = xyz_dist * actn_mask
            res[f"{body_part_name}-angular_distance"] = ang_dist
            res[f"{body_part_name}-xyz_distance"] = xyz_dist
        return res
    
    uc_3d_joint_metrics_dict = _compute_pose_and_loss(uc_actions[:, -1], gt_actions[:, -1], xsens_skel, action_mask)
    gc_3d_joint_metrics_dict = _compute_pose_and_loss(gc_actions[:, -1], gt_actions[:, -1], xsens_skel, action_mask)
    init_3d_joint_metrics_dict = _compute_pose_and_loss(first_pose[:, -1], gt_actions[:, -1], xsens_skel, action_mask)
    
    return {
        **{f"uc-{key}": value for key, value in uc_3d_joint_metrics_dict.items()},
        **{f"gc-{key}": value for key, value in gc_3d_joint_metrics_dict.items()},
        **{f"init-{key}": value for key, value in init_3d_joint_metrics_dict.items()},
    }


def reduce_metrics(mdict):
    for key, value in mdict.items():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        mdict[key] = value / torch.distributed.get_world_size()
    return mdict

def train_nomad(
    config: dict,
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    lr_scheduler: torch.optim.lr_scheduler,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    rank: int = 0,
):
    """
    Train the model for one epoch.

    Args:
        config: dict
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    loggers = {}
    
    if rank == 0:
        tepoch = tqdm.tqdm(dataloader, desc="Train Batch", leave=False)
    else:
        tepoch = dataloader
    for i, data in enumerate(tepoch):
        batch_obs_images = data["obs_image_transformed"].to(device, non_blocking=True)          # batch_obs_images_transformed shape: torch.Size([256, (context_size+1), 3, *image_size])
        batch_goal_images = data["goal_image_transformed"].to(device, non_blocking=True)        # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
        deltas = data["deltas"]                                                                 # actions shape: torch.Size([256, 8, 48]) # 8 actions, each with 48 dimensions
        context_poses = data["context_poses"].to(device, non_blocking=True)                     # context_poses shape: torch.Size([256, context_size+1, 45]) # context poses
        distance = data["distance"].to(device, non_blocking=True).float()                       # distance shape: torch.Size([256])
        goal_pos = data["goal_pos"].to(device, non_blocking=True)                               # goal_pos shape: torch.Size([256, 1, 48]) # single position
        action_mask = data["action_mask"].to(device, non_blocking=True)                         # action_mask shape: torch.Size([256]) # if valid action, I guess
        first_pose = data["first_pose"]                                                         # first_pose shape: torch.Size([256, 1, 48]),
        gt_actions_with_initial = data["gt_actions_with_initial"].to(device, non_blocking=True) # gt_actions_with_initial shape: torch.Size([256, 1, 48]),
        obs_images = data["obs_images"]                                                         # batch_obs_images_transformed shape: torch.Size([256, (context_size+1) * 3, *image_size])
        goal_image = data["goal_image"]                                                         # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
        goal_image_coords = data["goal_image_coords"].to(device, non_blocking=True)             # goal_image_coords shape: torch.Size([256, 4, 2])

        naction = deltas.to(device, non_blocking=True).float()
        B = deltas.shape[0]
        goal_mask = (torch.rand((B,), device=device) < goal_mask_prob).long() # 1 if goal mask, 0 if no mask
        
        goal_visible_mask = 1. - (goal_image_coords != -1).all(dim=2).to(torch.float32) # B, K, 2 -> B, K (K = # parts)
        goal_coordinates = None
        if config.get("goal_type", None) in ["2d", "2d5050"]:
            goal_coordinates = torch.stack(
                [goal_image_coords[:, XSensConstants.part_names.index(part_name)] for part_name in ["Pelvis","Head", "R_Hand", "L_Hand"]]
            , dim=1).flatten(1, 2) # B, 4*2
            if config.get("goal_type", None) == "2d": # only modify the goal_mask for 2d goals, rather, keep it at 50/50 for 2d5050
                nonvisible_goal_mask = 1. - (goal_image_coords == -1).all(dim=-1).all(dim=-1).to(torch.float32) # 1 if goal is visible, 0 if not
                goal_mask = (1.-((1.-goal_mask)*nonvisible_goal_mask)).long() # if goal is not visible, require goal masking
        if config.get("goal_type", None) == "point":
            goal_pos_xyz = data["goal_pose_xyz"].to(device, non_blocking=True)[:, 0] # B, 15, 3
            goal_pose = torch.cat(
                [goal_pos_xyz[:, XSensConstants.part_names.index(part)] for part in ["Pelvis", "Head", "R_Hand", "L_Hand"]]
            , dim=-1) # B, 4*3
        else:
            # goal_pose = gt_actions_with_initial[:, 0]
            goal_pose = goal_pos[:, 0]

        obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
        # Predict distance
        dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
        dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

        # Sample noise to add to actions
        noise = torch.randn(naction.shape, device=device)

        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each diffusion iteration
        noisy_action = noise_scheduler.add_noise(
            naction, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond, goal_pose=goal_pose * (1 - goal_mask[:, None]))

        def action_reduce(unreduced_loss: torch.Tensor):
            # Reduce over non-batch dimensions to get loss per batch element
            while unreduced_loss.dim() > 1:
                unreduced_loss = unreduced_loss.mean(dim=-1)
            assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
            return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

        # L2 loss
        diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
        
        # Total loss
        loss = alpha * dist_loss + (1-alpha) * diffusion_loss

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        
        if config.get("clipping", False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_norm", 1.0))
        
        optimizer.step()
        lr_scheduler.step()

        # Update Exponential Moving Average of the model weights
        ema_model.step(model)
        
        if isinstance(tepoch, tqdm.tqdm):   
            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        # In-step evaluations
        if i % print_log_freq == 0 or (image_log_freq != 0 and i % image_log_freq == 0):
            model_output_dict = model_output(
                ema_model.averaged_model, noise_scheduler,
                batch_obs_images, batch_goal_images, goal_pose, context_poses,
                pred_horizon=deltas.shape[1], action_dim=deltas.shape[2],
                num_samples=1,device=device,
                goal_coordinates=goal_coordinates,
            )
            model_output_dict["gc_actions"] = unnormalize_data_smpl_pose_gaussian(
                model_output_dict["gc_actions"].flatten(0, 1)
            ).unflatten(0, (B, -1))
            model_output_dict["uc_actions"] = unnormalize_data_smpl_pose_gaussian(
                model_output_dict["uc_actions"].flatten(0, 1)
            ).unflatten(0, (B, -1))
            
            # unnormalize from gaussian for loss metrics and visualizationse
            deltas = unnormalize_data_smpl_pose_gaussian(deltas.flatten(0, 1)).unflatten(0, (B, -1))
        
            # Compute metrics
            if i % print_log_freq == 0:
                _3dp_metrics = _compute_3d_joint_metrics(model_output_dict, deltas.to(device), action_mask.to(device), first_pose.to(device), XsensSkeleton())
                if torch.distributed.is_initialized(): # Reduce all metrics across ranks by averaging
                    _3dp_metrics = reduce_metrics(_3dp_metrics)
                
                data_log = {}
                data_log['uc-leaf-xyz'], data_log['uc-leaf-angular'] = 0, 0
                data_log['gc-leaf-xyz'], data_log['gc-leaf-angular'] = 0, 0
                data_log['gc-leaf-xyz-visible'], data_log['gc-leaf-xyz-notVisible'] = [], []
                data_log['gc-leaf-angular-visible'], data_log['gc-leaf-angular-notVisible'] = [], []
                for key, value in _3dp_metrics.items():
                    if "init" in key:
                        if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                            data_log[f"segm_leaf_init/{key}"] = value.mean().item()
                        else:
                            data_log[f"segm_init/{key}"] = value.mean().item()
                    elif "uc" in key or "gc" in key:
                        if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                            data_log[f"segm_leaf/{key}"] = value.mean().item()
                            if "uc" in key: 
                                if "xyz" in key: data_log['uc-leaf-xyz'] += value.mean().item() / 4.
                                elif "angular" in key: data_log['uc-leaf-angular'] += value.mean().item() / 4.
                            elif "gc" in key:
                                if "xyz" in key: data_log['gc-leaf-xyz'] += value.mean().item() / 4.
                                elif "angular" in key: data_log['gc-leaf-angular'] += value.mean().item() / 4.
                        else:
                            data_log[f"segm/{key}"] = value.mean().item()
                    
                    if "gc" in key:
                        part_index = XSensConstants.part_names.index(key[3:].split("-")[0])
                        vis_val = (value[goal_visible_mask[:, part_index] == 1]).mean().item()
                        not_vis_val = (value[goal_visible_mask[:, part_index] == 0]).mean().item()
                        if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                            if "xyz" in key: 
                                data_log[f"gc-leaf-xyz-visible"].append(vis_val)
                                data_log[f"gc-leaf-xyz-notVisible"].append(not_vis_val)
                            elif "angular" in key: 
                                data_log[f"gc-leaf-angular-visible"].append(vis_val)
                                data_log[f"gc-leaf-angular-notVisible"].append(not_vis_val)
                            data_log[f"segm_leaf_byVis/vis-{key}"] = vis_val
                            data_log[f"segm_leaf_byVis/notVis-{key}"] = not_vis_val
                        else:
                            data_log[f"segm_byVis/vis-{key}"] = vis_val
                            data_log[f"segm_byVis/notVis-{key}"] = not_vis_val
                
                for key in ["gc-leaf-xyz-visible", "gc-leaf-xyz-notVisible", "gc-leaf-angular-visible", "gc-leaf-angular-notVisible"]:
                    data_log[key] = np.nanmean(data_log[key])

                if use_wandb and i % wandb_log_freq == 0 and rank == 0:
                    wandb.log(data_log, commit=False)

            if image_log_freq != 0 and i % image_log_freq == 0 and rank == 0:
                batch_viz_obs_images = TF.resize(obs_images[:, -1], VISUALIZATION_IMAGE_SIZE[::-1])
                batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
                path = os.path.join(project_folder, f"epoch_{epoch}", "train")
                os.makedirs(path, exist_ok=True)
                
                for idx_ in range(5):
                    plot_fname = plot_images_and_actions_full_body(
                        image_plot_dir=path,
                        name=f"batch{i}_idx{idx_}",
                        cur_obs_image=batch_viz_obs_images[idx_],
                        cur_goal_image=batch_viz_goal_images[idx_],
                        cur_first_pose=first_pose[idx_],
                        gt_deltas=deltas[idx_],
                        deltas={"uncond": model_output_dict["uc_actions"][idx_], "goalcond": model_output_dict["gc_actions"][idx_]},
                        xsens_skel=XsensSkeleton()
                    )
                    if use_wandb and i % wandb_log_freq == 0:
                        wandb.log({f"train_vis/trajectory_gif_ex{idx_}": wandb.Video(plot_fname, format="gif")}, commit=False)

        # logging
        reduced_values = {
            "total_loss": loss.clone(),
            "dist_loss": dist_loss.clone(),
            "diffusion_loss": diffusion_loss.clone(),
            "goal_mask_prob": goal_mask.float().mean().clone(),
        }
        if torch.distributed.is_initialized():
            for key, value in reduced_values.items():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                reduced_values[key] = reduced_values[key] / torch.distributed.get_world_size()
            
        if use_wandb and i % wandb_log_freq == 0 and rank == 0:
            wandb.log({
                **{key: value.item() for key, value in reduced_values.items()},
                "lr": optimizer.param_groups[0]["lr"]
            })
                        
        if use_wandb and (i % wandb_log_freq == 0 or i % image_log_freq == 0 or i % print_log_freq == 0) and rank == 0:
            wandb.log({}, commit=True)  # Commit the batch log to wandb

@torch.no_grad()
def evaluate_nomad(
    config: dict,
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
    rank: int = 0,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        config: dict
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    eval_model = ema_model.to(device)
    
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)
    
    loggers = {}

    # Accumulate metrics for averaging
    rand_mask_loss_list = []
    no_mask_loss_list = []
    goal_mask_loss_list = []
    all_data_logs = []

    if rank == 0:
        tepoch = tqdm.tqdm(
            itertools.islice(dataloader, num_batches), 
            total=num_batches, 
            dynamic_ncols=True, 
            desc=f"Evaluating {eval_type} for epoch {epoch}", 
            leave=False)
    else:
        tepoch = itertools.islice(dataloader, num_batches)
        
    for i, data in enumerate(tepoch):
        batch_obs_images = data["obs_image_transformed"].to(device, non_blocking=True)          # batch_obs_images_transformed shape: torch.Size([256, (context_size+1), 3, *image_size])
        batch_goal_images = data["goal_image_transformed"].to(device, non_blocking=True)        # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
        deltas = data["deltas"]                                                                 #  actions shape: torch.Size([256, 8, 48]) # 8 actions, each with 48 dimensions
        context_poses = data["context_poses"].to(device, non_blocking=True)                     # context_poses shape: torch.Size([256, context_size+1, 45]) # context poses
        distance = data["distance"].to(device, non_blocking=True)                               # distance shape: torch.Size([256])
        goal_pos = data["goal_pos"].to(device, non_blocking=True)                               # goal_pos shape: torch.Size([256, 1, 48]) # single position
        action_mask = data["action_mask"].to(device)                                            # action_mask shape: torch.Size([256]) # if valid action, I guess
        first_pose = data["first_pose"]                                                         # first_pose shape: torch.Size([256, 1, 48]),
        gt_actions_with_initial = data["gt_actions_with_initial"].to(device, non_blocking=True) # gt_actions_with_initial shape: torch.Size([256, 1, 48]),
        obs_images = data["obs_images"]                                                         # batch_obs_images_transformed shape: torch.Size([256, (context_size+1) * 3, *image_size])
        goal_image = data["goal_image"]                                                         # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
        goal_image_coords = data["goal_image_coords"].to(device, non_blocking=True)             # goal_image_coords shape: torch.Size([256, 4, 2])
        
        B = deltas.shape[0]

        # Generate random goal mask
        rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
        goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
        no_mask = torch.zeros_like(rand_goal_mask).long().to(device)
        
        naction = deltas.to(device, non_blocking=True).float()
        
        goal_visible_mask = 1. - (goal_image_coords != -1).all(dim=2).to(torch.float32) # B, K, 2 -> B, K (K = # parts)
        goal_coordinates = None
        if config.get("goal_type", None) in ["2d", "2d5050"]:
            goal_coordinates = torch.stack(
                [goal_image_coords[:, XSensConstants.part_names.index(part_name)] for part_name in ["Pelvis", "Head", "R_Hand", "L_Hand"]]
            , dim=1).flatten(1, 2) # B, 4*2
        if config.get("goal_type", None) == "point":
            goal_pos_xyz = data["goal_pose_xyz"].to(device, non_blocking=True)[:, 0] # B, 15, 3
            goal_pose = torch.cat(
                [goal_pos_xyz[:, XSensConstants.part_names.index(part)] for part in ["Pelvis", "Head", "R_Hand", "L_Hand"]]
            , dim=-1) # B, 4*3
        else:
            # goal_pose = gt_actions_with_initial[:, 0]
            goal_pose = goal_pos[:, 0]
            
        # fix masking for waypoint_masking setting
        # instead of setting masking by the goal_mask, set the goal images (to the goal if unmasked, to the latest obs if masked)
        if config.get("waypoint_masking", None) == "uniform":
            rand_goals_binary = torch.rand((B,), device=device) < 0.5
            rand_mask_goal_images = torch.where(rand_goals_binary[:, None, None, None], batch_obs_images[:, -1], batch_goal_images)
            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=rand_mask_goal_images, input_goal_mask=no_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_obs_images[:, -1], input_goal_mask=no_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
        else:
            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)

        obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)

        # Sample noise to add to actions
        noise = torch.randn(naction.shape, device=device)

        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        noisy_actions = noise_scheduler.add_noise(
            naction, noise, timesteps)

        ### RANDOM MASK ERROR ###
        # Predict the noise residual
        rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond, goal_pose=goal_pose * (1 - rand_goal_mask[:, None]))
        
        # L2 loss
        rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
        
        ### NO MASK ERROR ###
        # Predict the noise residual
        no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond, goal_pose=goal_pose * (1 - no_mask[:, None]))
        
        # L2 loss
        no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

        ### GOAL MASK ERROR ###
        # predict the noise residual
        goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond, goal_pose=goal_pose * (1 - goal_mask[:, None]))
        
        # L2 loss
        goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
        
        # Accumulate losses
        reduce_dict = {
            "rand_mask_loss": rand_mask_loss.clone(),
            "no_mask_loss": no_mask_loss.clone(),
            "goal_mask_loss": goal_mask_loss.clone(),
        }
        if torch.distributed.is_initialized():
            for key, value in reduce_dict.items():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                reduce_dict[key] = reduce_dict[key] / torch.distributed.get_world_size()
            
        rand_mask_loss_list.append(reduce_dict["rand_mask_loss"].item())
        no_mask_loss_list.append(reduce_dict["no_mask_loss"].item())
        goal_mask_loss_list.append(reduce_dict["goal_mask_loss"].item())

        if isinstance(tepoch, tqdm.tqdm):
            tepoch.set_postfix(loss=rand_mask_loss.item())

        # Accumulate metrics for averaging at the end
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            batch_obs_images,
            batch_goal_images,
            goal_pose,
            context_poses,
            pred_horizon=deltas.shape[1],
            action_dim=deltas.shape[2],
            num_samples=1,
            device=device,
            goal_coordinates=goal_coordinates,
            waypoint_masking=config.get("waypoint_masking", None),
        )
        model_output_dict["gc_actions"] = unnormalize_data_smpl_pose_gaussian(
            model_output_dict["gc_actions"].flatten(0, 1)
        ).unflatten(0, (B, -1))
        model_output_dict["uc_actions"] = unnormalize_data_smpl_pose_gaussian(
            model_output_dict["uc_actions"].flatten(0, 1)
        ).unflatten(0, (B, -1))
        
        # unnormalize from gaussian for loss metrics and visualizations
        deltas = unnormalize_data_smpl_pose_gaussian(deltas.flatten(0, 1)).unflatten(0, (B, -1))
        
        _3dp_metrics = _compute_3d_joint_metrics(model_output_dict, deltas.to(device), action_mask.to(device), first_pose.to(device), XsensSkeleton())
        if torch.distributed.is_initialized(): # Reduce all metrics across ranks by averaging
            _3dp_metrics = reduce_metrics(_3dp_metrics)
        
        data_log = {}
        data_log["eval/uc-leaf-xyz"], data_log['eval/uc-leaf-angular'] = 0, 0
        data_log["eval/gc-leaf-xyz"], data_log['eval/gc-leaf-angular'] = 0, 0
        data_log['eval/gc-leaf-xyz-visible'], data_log['eval/gc-leaf-xyz-notVisible'] = [], []
        data_log['eval/gc-leaf-angular-visible'], data_log['eval/gc-leaf-angular-notVisible'] = [], []
        for key, value in _3dp_metrics.items():
            if "init" in key:
                if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                    data_log[f"eval_segm_leaf_init/{key}"] = value.mean().item()
                else:
                    data_log[f"eval_segm_init/{key}"] = value.mean().item()
            elif "uc" in key or "gc" in key:
                if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                    data_log[f"eval_segm_leaf/{key}"] = value.mean().item()
                    if "uc" in key: 
                        if "xyz" in key: data_log['eval/uc-leaf-xyz'] += value.mean().item() / 4.
                        elif "angular" in key: data_log['eval/uc-leaf-angular'] += value.mean().item() / 4.
                    elif "gc" in key:
                        if "xyz" in key: data_log['eval/gc-leaf-xyz'] += value.mean().item() / 4.
                        elif "angular" in key: data_log['eval/gc-leaf-angular'] += value.mean().item() / 4.
                else:
                    data_log[f"eval_segm/{key}"] = value.mean().item()
            
            if "gc" in key:
                part_index = XSensConstants.part_names.index(key[3:].split("-")[0])
                vis_val = (value[goal_visible_mask[:, part_index] == 1]).mean().item()
                not_vis_val = (value[goal_visible_mask[:, part_index] == 0]).mean().item()
                if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                    if "xyz" in key: 
                        data_log[f"eval/gc-leaf-xyz-visible"].append(vis_val)
                        data_log[f"eval/gc-leaf-xyz-notVisible"].append(not_vis_val)
                    elif "angular" in key: 
                        data_log[f"eval/gc-leaf-angular-visible"].append(vis_val)
                        data_log[f"eval/gc-leaf-angular-notVisible"].append(not_vis_val)
                    data_log[f"eval_segm_leaf_byVis/vis-{key}"] = vis_val
                    data_log[f"eval_segm_leaf_byVis/notVis-{key}"] = not_vis_val
                else:
                    data_log[f"eval_segm_byVis/vis-{key}"] = vis_val
                    data_log[f"eval_segm_byVis/notVis-{key}"] = not_vis_val
                    
        for key in ["eval/gc-leaf-xyz-visible", "eval/gc-leaf-xyz-notVisible", "eval/gc-leaf-angular-visible", "eval/gc-leaf-angular-notVisible"]:
            data_log[key] = np.nanmean(data_log[key])
    
        all_data_logs.append(data_log)
        if i == 0 and rank == 0:
            batch_viz_obs_images = TF.resize(obs_images[:, -1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            path = os.path.join(project_folder, f"epoch_{epoch}", "eval")
            os.makedirs(path, exist_ok=True)
            for idx_ in range(min(10, B)):
                plot_fname = plot_images_and_actions_full_body(
                    image_plot_dir=path,
                    name=f"batch{i}_idx{idx_}",
                    cur_obs_image=batch_viz_obs_images[idx_],
                    cur_goal_image=batch_viz_goal_images[idx_],
                    cur_first_pose=first_pose[idx_],
                    gt_deltas=deltas[idx_],
                    deltas={"uncond": model_output_dict['uc_actions'][idx_], "goalcond": model_output_dict['gc_actions'][idx_]},
                    xsens_skel=XsensSkeleton()
                )
                if use_wandb:
                    wandb.log({f"eval_vis/trajectory_gif_ex{idx_}": wandb.Video(plot_fname, format="gif")}, commit=False)

    # At the end, log averaged metrics to wandb
    if use_wandb and rank == 0:
        # Average all_data_logs if present
        avg_data_log = {}
        if all_data_logs:
            keys = set().union(*all_data_logs)
            for key in keys:
                vals = [d[key] for d in all_data_logs if key in d]
                if vals:
                    avg_data_log[key] = float(np.nanmean(vals))

        avg_rand_mask_loss = np.mean(rand_mask_loss_list) if rand_mask_loss_list else 0.0
        avg_no_mask_loss = np.mean(no_mask_loss_list) if no_mask_loss_list else 0.0
        avg_goal_mask_loss = np.mean(goal_mask_loss_list) if goal_mask_loss_list else 0.0

        # Add the diffusion losses
        avg_data_log["eval/diffusion_loss (random masking)"] = avg_rand_mask_loss
        avg_data_log["eval/diffusion_loss (no masking)"] = avg_no_mask_loss
        avg_data_log["eval/diffusion_loss (goal masking)"] = avg_goal_mask_loss

        wandb.log(avg_data_log)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    goal_pose: torch.Tensor,
    context_poses: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    goal_coordinates: torch.Tensor = None,
    waypoint_masking: Optional[str] = None,
):
    """
    Generate model output (conditioned, unconditioned, distance) for the given batch of images.
    Outputs are DELTAS and are NOT unnormalized or scaled.
    """
    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    
    if waypoint_masking == "uniform":
        obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_obs_images[:, -1], input_goal_mask=no_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
    else:
        obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)
    
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask, context_poses=context_poses, goal_coordinates=goal_coordinates)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond,
            goal_pose=goal_pose * (1 - goal_mask[:, None])
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    B, T  = diffusion_output.shape[0], diffusion_output.shape[1]
    uc_actions = diffusion_output

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond,
            goal_pose=goal_pose * (1-no_mask[:, None])
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    gc_actions = diffusion_output
    

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }
