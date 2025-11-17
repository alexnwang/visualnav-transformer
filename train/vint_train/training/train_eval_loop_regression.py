from collections import defaultdict
import itertools
from diffusers.training_utils import EMAModel
import tqdm
import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import forward_kinematics_wrapper, get_action_smpl_torch, unnormalize_data_smpl_pose_gaussian
from vint_train.training.train_utils import reduce_metrics

def train_eval_loop_regression(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    current_epoch: int = 0,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    rank: int = 0,
):
    """
    Train and evaluate the model for several epochs (regression model)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    for epoch in range(current_epoch, epochs):
        if train_model:
            print(
            f"Start Regression Training Epoch {epoch}/{epochs - 1}"
            )
            train_regression(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                dataloader=train_loader,
                epoch=epoch,
                device=device,
                print_log_freq=print_log_freq,
                use_wandb=use_wandb,
                wandb_log_freq=wandb_log_freq,
            )
        if rank == 0:
            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)

            # save optimizer
            numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} Regression Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_regression(
                    model=model,
                    dataloader=loader,
                    device=device,
                    eval_fraction=eval_fraction,
                    use_wandb=use_wandb,
                )

def compute_metrics(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
):
    skeleton = XsensSkeleton()
    pred_xyz = forward_kinematics_wrapper(pred_action, skeleton, XSensConstants.upper_body_num_parts)
    gt_xyz = forward_kinematics_wrapper(gt_action, skeleton, XSensConstants.upper_body_num_parts)
    pred_rpy = pred_action[:, 3:].reshape(-1, XSensConstants.upper_body_num_parts, 3)
    gt_rpy = gt_action[:, 3:].reshape(-1, XSensConstants.upper_body_num_parts, 3)
    res = {}
    for i, body_part_name in enumerate(XSensConstants.part_names[:XSensConstants.upper_body_num_parts]):
        R_gt = R.from_euler('xyz', gt_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
        R_pred = R.from_euler('xyz', pred_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
        ang_dist = torch.from_numpy((R_gt.inv() * R_pred).magnitude() / np.pi * 180).to(pred_action.device).float() # B
        xyz_dist = torch.norm(gt_xyz[:, i, :] - pred_xyz[:, i, :], dim=-1) # B
        res[f"{body_part_name}-angular_distance"] = ang_dist.mean()
        res[f"{body_part_name}-xyz_distance"] = xyz_dist.mean()
    return res

def train_regression(
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler,
    dataloader: DataLoader,
    epoch: int,
    device: torch.device,
    print_log_freq: int = 50,
    use_wandb: bool = True,
    wandb_log_freq: int = 100,
):
    rank = torch.distributed.get_rank()
    model.train()
    
    if rank == 0:
        tepoch = tqdm.tqdm(
            dataloader,total=len(dataloader),dynamic_ncols=True,desc=f"Training Regression",leave=False
        )
    else:   
        tepoch = dataloader
    for i, data in enumerate(tepoch):
        (
            batch_obs_images, # obs_image shape: torch.Size([256, (context_size+1) * 3, *image_size])
            batch_goal_images, # goal_image shape: torch.Size([256, 3, *image_size])
            deltas, #  actions shape: torch.Size([256, 8, 48]) # 8 actions, each with 48 dimensions
            context_poses, # context poses shape: torch.Size([256, (context_size+1), 48]) # 3 context poses + current, each with 48 dimensions
            distance, # distance shape: torch.Size([256])
            goal_pos, # goal_pos shape: torch.Size([256, 1, 48]) # single position
            action_mask, # action_mask shape: torch.Size([256]) # if valid action, I guess
            first_pose, # first_pose shape: torch.Size([256, 1, 48]),
            gt_actions_with_initial, # gt_actions_with_initial shape: torch.Size([256, 1, 48]),
            obs_images, # batch_obs_images_transformed shape: torch.Size([256, (context_size+1) * 3, *image_size])
            goal_image, # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
        ) = data

        batch_obs_images = batch_obs_images.to(device, non_blocking=True)
        batch_goal_images = batch_goal_images.to(device, non_blocking=True)
        context_poses = context_poses.to(device, non_blocking=True)
        gt_action = gt_actions_with_initial.to(device, non_blocking=True)[:, 0]
        first_pose = first_pose.to(device, non_blocking=True)[:, 0]

        pred_action = model(batch_obs_images, batch_goal_images, context_poses)
        loss = nn.functional.mse_loss(pred_action, gt_action)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if rank == 0:
            tepoch.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])
            
        if i % print_log_freq == 0:
            if rank == 0: print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")
            metrics = compute_metrics(pred_action.detach(), gt_action.detach())
            init_metrics = compute_metrics(first_pose.detach(), gt_action.detach())
            if torch.distributed.is_initialized():
                metrics = reduce_metrics(metrics)
                init_metrics = reduce_metrics(init_metrics)
                
            data_log = {} 
            for key, value in init_metrics.items(): data_log[f"segments_init/{key}"] = value.item()
            for key, value in metrics.items():
                if rank == 0: print("Metrics:", key, value.item())
                if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                    data_log[f"segments_leaf/{key}"] = value.item()
                else:
                    data_log[f"segments/{key}"] = value.item()
            
            if use_wandb and i % wandb_log_freq == 0 and rank == 0:
                wandb.log(data_log, commit=False)
        if use_wandb and (i % wandb_log_freq == 0 or i % print_log_freq == 0) and rank == 0:
            wandb.log({
                "total_loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }, commit=True)  # Commit the batch log to wandb

        
def evaluate_regression(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    model.eval()
    
    metric_accumulator = defaultdict(float)
    rank = torch.distributed.get_rank()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)
    
    if rank == 0:
        tepoch = tqdm.tqdm(
            itertools.islice(dataloader, num_batches), 
            total=num_batches, 
            dynamic_ncols=True, 
            desc=f"Evaluating Regression", 
            leave=False)
    else:
        tepoch = itertools.islice(dataloader, num_batches)
    
    for i, data in enumerate(tepoch):
        (
            batch_obs_images, # batch_obs_images_transformed shape: torch.Size([256, (context_size+1) * 3, *image_size])
            batch_goal_images, # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
            deltas, #  actions shape: torch.Size([256, 8, 48]) # 8 actions, each with 48 dimensions
            context_poses, # context_poses shape: torch.Size([256, context_size+1, 45]) # context poses
            distance, # distance shape: torch.Size([256])
            goal_pos, # goal_pos shape: torch.Size([256, 1, 48]) # single position
            action_mask, # action_mask shape: torch.Size([256]) # if valid action, I guess
            first_pose, # first_pose shape: torch.Size([256, 1, 48]),
            gt_actions_with_initial, # gt_actions_with_initial shape: torch.Size([256, 1, 48]),
            obs_images, # batch_obs_images_transformed shape: torch.Size([256, (context_size+1) * 3, *image_size])
            goal_image, # batch_goal_images_transformed shape: torch.Size([256, 3, *image_size])
        ) = data
        
        batch_obs_images = batch_obs_images.to(device, non_blocking=True)
        batch_goal_images = batch_goal_images.to(device, non_blocking=True)
        context_poses = context_poses.to(device, non_blocking=True)
        gt_action = gt_actions_with_initial.to(device, non_blocking=True)[:, 0]
        first_pose = first_pose.to(device, non_blocking=True)[:, 0]

        pred_action = model(batch_obs_images, batch_goal_images, context_poses)
        loss = nn.functional.mse_loss(pred_action, gt_action)
        
        init_metrics = compute_metrics(first_pose.detach(), gt_action.detach())
        metrics = compute_metrics(pred_action.detach(), gt_action.detach())
        if torch.distributed.is_initialized():
            init_metrics = reduce_metrics(init_metrics)
            metrics = reduce_metrics(metrics)
            loss = reduce_metrics({"total_loss": loss})["total_loss"]
            
        for key, value in init_metrics.items(): metric_accumulator[f"eval_segments_init/{key}"] += value.mean().item()
        for key, value in metrics.items():
            if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                metric_accumulator[f"eval_segments_leaf/{key}"] += value.mean().item()
            else:
                metric_accumulator[f"eval_segments/{key}"] += value.mean().item()
        metric_accumulator["total_loss"] += loss.mean().item()
    
    for key, value in metric_accumulator.items():
        metric_accumulator[key] /= num_batches
    
    if rank == 0 and use_wandb:
        wandb.log(metric_accumulator, commit=False)
        
    if rank == 0: print("Eval Completed, Metrics:")
    for key, value in metric_accumulator.items():
        if rank == 0: print("Metrics:", key, value)
    return metric_accumulator