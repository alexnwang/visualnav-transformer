from vint_train.data.misc import XsensSkeleton, XSensConstants
from vint_train.training.nymeria_training_utils import get_action_smpl_torch, get_delta_smpl, normalize_data_smpl_pose, plot_images_and_actions_full_body, unnormalize_data_smpl_pose, unnormalize_data_smpl_pose_gaussian
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

def _compute_metrics_nomad(
    model_output_dict: Dict[str, torch.Tensor],
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :3], batch_action_label[:, :, :3], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :3], start_dim=1),
        torch.flatten(batch_action_label[:, :, :3], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :3], batch_action_label[:, :, :3], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :3], start_dim=1),
        torch.flatten(batch_action_label[:, :, :3], start_dim=1),
        dim=-1,
    ))
    
    # compute per-segment losses
    segment_results = {}
    
    uc_actions = uc_actions.flatten(0, 1)
    gc_actions = gc_actions.flatten(0, 1)
    batch_action_label = batch_action_label.flatten(0, 1)
    
    for i, segment in enumerate(XSensConstants.part_names[:XSensConstants.upper_body_num_parts]):
        if segment == "Pelvis":
            segment_results[f"segments/uc_{segment}_xyz_loss"] = F.mse_loss(uc_actions[:, 3*i:3*(i+1)], batch_action_label[:, 3*i:3*(i+1)])
            segment_results[f"segments/gc_{segment}_xyz_loss"] = F.mse_loss(gc_actions[:, 3*i:3*(i+1)], batch_action_label[:, 3*i:3*(i+1)], reduction="mean")
        
        uc_R = R.from_euler('xyz', to_numpy(uc_actions[:, 3*i+3:3*(i+1)+3]), degrees=False)
        gc_R = R.from_euler('xyz', to_numpy(gc_actions[:, 3*i+3:3*(i+1)+3]), degrees=False)
        batch_R = R.from_euler('xyz', to_numpy(batch_action_label[:, 3*i+3:3*(i+1)+3]), degrees=False)
        
        # Compute angular distance (in radians) between uc_R and batch_R
        ang_dist = uc_R.inv() * batch_R
        ang_dist = ang_dist.magnitude()  # Returns angle in radians as numpy array
        segment_results[f"segments/uc_{segment}_angular_distance"] = torch.from_numpy(ang_dist).to(uc_actions.device).float().mean()
        
        # Compute angular distance (in radians) between gc_R and batch_R
        ang_dist = gc_R.inv() * batch_R
        ang_dist = ang_dist.magnitude()
        segment_results[f"segments/gc_{segment}_angular_distance"] = torch.from_numpy(ang_dist).to(gc_actions.device).float().mean()

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
        **segment_results,
    }

    return results


def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    proprioception: bool,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
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
        model: model to train
        ema_model: exponential moving average model
        proprioception: whether to use proprioception
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
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
        (
            obs_image, # obs_image shape: torch.Size([256, 12, 96, 96])
            goal_image, # goal_image shape: torch.Size([256, 3, 96, 96])
            deltas, #  actions shape: torch.Size([256, 8, 48]) # 8 actions, each with 48 dimensions
            distance, # distance shape: torch.Size([256])
            goal_pos, # goal_pos shape: torch.Size([256, 1, 48]) # single position
            dataset_idx, # dataset_idx shape: torch.Size([256]) # which dataset?
            action_mask, # action_mask shape: torch.Size([256]) # if valid action, I guess
            first_pose, # first_pose shape: torch.Size([256, 1, 48])
        ) = data
                    
        obs_images = torch.split(obs_image, 3, dim=1)
        batch_obs_images = [transform(obs) for obs in obs_images]
        batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device, non_blocking=True)
        batch_goal_images = transform(goal_image).to(device, non_blocking=True)
        action_mask = action_mask.to(device, non_blocking=True)
        distance = distance.float().to(device, non_blocking=True)
        naction = deltas.to(device, non_blocking=True).float()
        joint_angles = first_pose[:, 0, 3:].to(device, non_blocking=True)

        B = deltas.shape[0]

        # Generate random goal mask
        goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
        obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
        # Predict distance
        dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
        dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

        if proprioception:
            obsgoal_cond = torch.cat([obsgoal_cond, joint_angles], dim=1)

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
        noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

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
        optimizer.step()
        lr_scheduler.step()

        # Update Exponential Moving Average of the model weights
        ema_model.step(model)
        
        # In-step evaluations
        if i % print_log_freq == 0 or (image_log_freq != 0 and i % image_log_freq == 0):
            model_output_dict = model_output(
                ema_model.averaged_model, proprioception, noise_scheduler,
                batch_obs_images, batch_goal_images, first_pose,
                pred_horizon=deltas.shape[1], action_dim=deltas.shape[2],
                num_samples=1,device=device,
            )
            model_output_dict["gc_actions"] = unnormalize_data_smpl_pose_gaussian(
                model_output_dict["gc_actions"].flatten(0, 1)
            ).unflatten(0, (B, -1))
            model_output_dict["uc_actions"] = unnormalize_data_smpl_pose_gaussian(
                model_output_dict["uc_actions"].flatten(0, 1)
            ).unflatten(0, (B, -1))
            
            # unnormalize from gaussian for loss metrics and visualizations
            first_pose = unnormalize_data_smpl_pose_gaussian(first_pose.flatten(0, 1)).unflatten(0, (B, -1))
            deltas = unnormalize_data_smpl_pose_gaussian(deltas.flatten(0, 1)).unflatten(0, (B, -1))
        
            # Compute metrics
            if i % print_log_freq == 0:
                metrics = _compute_metrics_nomad(model_output_dict, distance.to(device), deltas.to(device), action_mask.to(device))
                
                if torch.distributed.is_initialized():
                    # Reduce all metrics across ranks by averaging
                    for key, value in metrics.items():
                        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                        metrics[key] = value / torch.distributed.get_world_size()
                
                data_log = {}
                for key, value in metrics.items():
                    if key not in loggers:
                        loggers[key] = Logger(key, "train", window_size=print_log_freq)
                    loggers[key].log_data(value.item())
                    data_log[key] = value.item()
            
                for key, logger in loggers.items():
                    if "segments/" in key:
                        continue
                    if i % print_log_freq == 0 and print_log_freq != 0 and rank == 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and rank == 0:
                    wandb.log(data_log, commit=False)

            if image_log_freq != 0 and i % image_log_freq == 0 and rank == 0:
                batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
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
        reduced_loss = loss.clone()
        reduced_dist_loss = dist_loss.clone()
        reduced_diffusion_loss = diffusion_loss.clone()
        
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reduced_dist_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reduced_diffusion_loss, op=torch.distributed.ReduceOp.SUM)
            reduced_loss = reduced_loss / torch.distributed.get_world_size()
            reduced_dist_loss = reduced_dist_loss / torch.distributed.get_world_size()
            reduced_diffusion_loss = reduced_diffusion_loss / torch.distributed.get_world_size()
            
        if use_wandb and i % wandb_log_freq == 0 and rank == 0:
            wandb.log({
                "total_loss": reduced_loss,
                "dist_loss": reduced_dist_loss,
                "diffusion_loss": reduced_diffusion_loss,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
        if isinstance(tepoch, tqdm.tqdm):   
            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                        
        if use_wandb and (i % wandb_log_freq == 0 or i % image_log_freq == 0 or i % print_log_freq == 0) and rank == 0:
            wandb.log({}, commit=True)  # Commit the batch log to wandb

@torch.no_grad()
def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    proprioception: bool,
    dataloader: DataLoader,
    transform: transforms,
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
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        proprioception: whether to use proprioception
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
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
        (
            obs_image, # obs_image shape: torch.Size([256, 12, 96, 96])
            goal_image, # goal_image shape: torch.Size([256, 3, 96, 96])
            deltas, #  actions shape: torch.Size([256, 8, 48]) # 8 actions, each with 48 dimensions
            distance, # distance shape: torch.Size([256])
            goal_pos, # goal_pos shape: torch.Size([256, 1, 48]) # single position
            dataset_idx, # dataset_idx shape: torch.Size([256]) # which dataset?
            action_mask, # action_mask shape: torch.Size([256]) # if valid action, I guess
            first_pose, # first_pose shape: torch.Size([256, 1, 48])
        ) = data
        
        obs_images = torch.split(obs_image, 3, dim=1)
        batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
        batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
        batch_obs_images = [transform(obs) for obs in obs_images]
        batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
        batch_goal_images = transform(goal_image).to(device)
        action_mask = action_mask.to(device)

        B = deltas.shape[0]

        # Generate random goal mask
        rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
        goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
        no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

        rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

        obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
        obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
        goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

        if proprioception:
            joint_angles = first_pose[:, 0, 3:].to(device)
            obsgoal_cond = torch.cat([obsgoal_cond, joint_angles], dim=1)
            goal_mask_cond = torch.cat([goal_mask_cond, joint_angles], dim=1)
            rand_mask_cond = torch.cat([rand_mask_cond, joint_angles], dim=1)

        distance = distance.to(device)

        naction = deltas.to(device).float()

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
        rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
        
        # L2 loss
        rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
        
        ### NO MASK ERROR ###
        # Predict the noise residual
        no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
        
        # L2 loss
        no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

        ### GOAL MASK ERROR ###
        # predict the noise residual
        goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
        
        # L2 loss
        goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
        
        # Accumulate losses
        reduced_rand_mask_loss = rand_mask_loss.clone()
        reduced_no_mask_loss = no_mask_loss.clone()
        reduced_goal_mask_loss = goal_mask_loss.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reduced_rand_mask_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reduced_no_mask_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reduced_goal_mask_loss, op=torch.distributed.ReduceOp.SUM)
            reduced_rand_mask_loss = reduced_rand_mask_loss / torch.distributed.get_world_size()
            reduced_no_mask_loss = reduced_no_mask_loss / torch.distributed.get_world_size()
            reduced_goal_mask_loss = reduced_goal_mask_loss / torch.distributed.get_world_size()
            
        rand_mask_loss_list.append(reduced_rand_mask_loss.item())
        no_mask_loss_list.append(reduced_no_mask_loss.item())
        goal_mask_loss_list.append(reduced_goal_mask_loss.item())

        if isinstance(tepoch, tqdm.tqdm):
            tepoch.set_postfix(loss=rand_mask_loss.item())

        # Accumulate metrics for averaging at the end
        model_output_dict = model_output(
            ema_model,
            proprioception,
            noise_scheduler,
            batch_obs_images,
            batch_goal_images,
            first_pose,
            pred_horizon=deltas.shape[1],
            action_dim=deltas.shape[2],
            num_samples=1,
            device=device,
        )
        model_output_dict["gc_actions"] = unnormalize_data_smpl_pose_gaussian(
            model_output_dict["gc_actions"].flatten(0, 1)
        ).unflatten(0, (B, -1))
        model_output_dict["uc_actions"] = unnormalize_data_smpl_pose_gaussian(
            model_output_dict["uc_actions"].flatten(0, 1)
        ).unflatten(0, (B, -1))
        
        # unnormalize from gaussian for loss metrics and visualizations
        first_pose = unnormalize_data_smpl_pose_gaussian(first_pose.flatten(0, 1)).unflatten(0, (B, -1))
        deltas = unnormalize_data_smpl_pose_gaussian(deltas.flatten(0, 1)).unflatten(0, (B, -1))

        metrics = _compute_metrics_nomad(
                    model_output_dict,
                    distance.to(device),
                    deltas.to(device),
                    action_mask.to(device),
                )
        
        if torch.distributed.is_initialized():
            # Reduce all metrics across ranks by averaging
            for key, value in metrics.items():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                metrics[key] = value / torch.distributed.get_world_size()
        
        data_log = {}
        for key, value in metrics.items():
            if key not in loggers:
                loggers[key] = Logger(key, "eval", window_size=print_log_freq)
            loggers[key].log_data(value.item())
            data_log[f"eval/{key}"] = value.item()
        all_data_logs.append(data_log)
        
        if i % print_log_freq == 0:
            for key, logger in loggers.items():
                if "segments/" in key:
                    continue
                if i % print_log_freq == 0 and print_log_freq != 0 and rank == 0:
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

        if i == 0 and rank == 0:
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
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
                    avg_data_log[key] = float(np.mean(vals))

        avg_rand_mask_loss = np.mean(rand_mask_loss_list) if rand_mask_loss_list else 0.0
        avg_no_mask_loss = np.mean(no_mask_loss_list) if no_mask_loss_list else 0.0
        avg_goal_mask_loss = np.mean(goal_mask_loss_list) if goal_mask_loss_list else 0.0

        # Add the diffusion losses
        avg_data_log["eval/diffusion_loss (random masking)"] = avg_rand_mask_loss
        avg_data_log["eval/diffusion_loss (no masking)"] = avg_no_mask_loss
        avg_data_log["eval/diffusion_loss (goal masking)"] = avg_goal_mask_loss

        wandb.log(avg_data_log)


# normalize data
# def get_data_stats(data):
#     data = data.reshape(-1,data.shape[-1])
#     stats = {
#         'min': np.min(data, axis=0),
#         'max': np.max(data, axis=0)
#     }
#     return stats

# def normalize_data(data, stats):
#     # nomalize to [0,1]
#     ndata = (data - stats['min']) / (stats['max'] - stats['min'])
#     # normalize to [-1, 1]
#     ndata = ndata * 2 - 1
#     return ndata

# def unnormalize_data(ndata, stats):
#     ndata = (ndata + 1) / 2
#     data = ndata * (stats['max'] - stats['min']) + stats['min']
#     return data

# def get_delta(actions):
#     # append zeros to first action
#     ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
#     delta = ex_actions[:,1:] - ex_actions[:,:-1]
#     return delta

# def get_action(diffusion_output, action_stats=ACTION_STATS):
#     # diffusion_output: (B, 2*T+1, 1)
#     # return: (B, T-1)
#     device = diffusion_output.device
#     ndeltas = diffusion_output
#     ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
#     ndeltas = to_numpy(ndeltas)
#     ndeltas = unnormalize_data(ndeltas, action_stats)
#     actions = np.cumsum(ndeltas, axis=1)
#     return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    proprioception: bool,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    first_pose: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    """
    Generate model output (conditioned, unconditioned, distance) for the given batch of images.
    Outputs are DELTAS and are NOT unnormalized or scaled.
    """
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    if proprioception:
        joint_angles = first_pose[:, 0, 3:].to(device)
        obsgoal_cond = torch.cat([obsgoal_cond, joint_angles], dim=1)
        obs_cond = torch.cat([obs_cond, joint_angles], dim=1)

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
            global_cond=obs_cond
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
            global_cond=obsgoal_cond
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
