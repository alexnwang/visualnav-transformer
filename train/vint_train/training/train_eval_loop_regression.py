from collections import defaultdict
import itertools
import heapq
from diffusers.training_utils import EMAModel
import tqdm
import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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
    target_type: str = "goal_pose", # "goal_pose" or "current_pose"
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
                target_type=target_type,
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
                    target_type=target_type,
                    model=model,
                    dataloader=loader,
                    device=device,
                    eval_fraction=eval_fraction,
                    use_wandb=use_wandb,
                )
        if epoch + 1 == epochs:
            evaluate_regression_distribution(
                target_type=target_type,
                model=model,
                dataloader=loader,
                device=device,
                eval_fraction=eval_fraction,
                output_folder=os.path.join(project_folder, "distribution_evaluation"),
                use_wandb=use_wandb,
            )

def compute_metrics(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    reduce_mean: bool = True
):
    skeleton = XsensSkeleton()
    pred_xyz, pred_rpy = forward_kinematics_wrapper(pred_action, skeleton, XSensConstants.upper_body_num_parts, return_euler=True)
    gt_xyz, gt_rpy = forward_kinematics_wrapper(gt_action, skeleton, XSensConstants.upper_body_num_parts, return_euler=True)
    res = {}
    for i, body_part_name in enumerate(XSensConstants.part_names[:XSensConstants.upper_body_num_parts]):
        R_gt = R.from_euler('xyz', gt_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
        R_pred = R.from_euler('xyz', pred_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
        ang_dist = torch.from_numpy((R_gt.inv() * R_pred).magnitude() / np.pi * 180).to(pred_action.device).float() # B
        xyz_dist = torch.norm(gt_xyz[:, i, :] - pred_xyz[:, i, :], dim=-1) # B
        res[f"{body_part_name}-angular_distance"] = ang_dist.mean() if reduce_mean else ang_dist
        res[f"{body_part_name}-xyz_distance"] = xyz_dist.mean() if reduce_mean else xyz_dist
    return res

def train_regression(
    target_type: str,
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
        
        if target_type == "goal_pose":
            target = gt_action
        elif target_type == "current_pose":
            target = context_poses[:, -1]
        else:
            raise ValueError(f"Invalid target type: {target_type}")

        pred_action = model(batch_obs_images, batch_goal_images, context_poses)
        # loss = nn.functional.mse_loss(pred_action, gt_action)
        loss = nn.functional.mse_loss(pred_action, target)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if rank == 0:
            tepoch.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])
            
        if i % print_log_freq == 0:
            if rank == 0: print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")
            metrics = compute_metrics(pred_action.detach(), target.detach())
            init_metrics = compute_metrics(first_pose.detach(), target.detach())
            if torch.distributed.is_initialized():
                metrics = reduce_metrics(metrics)
                init_metrics = reduce_metrics(init_metrics)
                
            data_log = {} 
            for key, value in init_metrics.items(): data_log[f"segments_init/{key}"] = value.item()
            for key, value in metrics.items():
                # if rank == 0: print("Metrics:", key, value.item())
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

        
@torch.no_grad()
def evaluate_regression(
    target_type: str,
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
        if target_type == "goal_pose":
            target = gt_action
        elif target_type == "current_pose":
            target = context_poses[:, -1]
        else:
            raise ValueError(f"Invalid target type: {target_type}")
        loss = nn.functional.mse_loss(pred_action, target)
        
        init_metrics = compute_metrics(first_pose.detach(), target.detach())
        metrics = compute_metrics(pred_action.detach(), target.detach())
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
    # for key, value in metric_accumulator.items():
        # if rank == 0: print("Metrics:", key, value)
    return metric_accumulator

@torch.no_grad()
def evaluate_regression_distribution(
    target_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_folder: str,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
    save_top_k_images: int = 10,
):
    assert target_type == "goal_pose", "Only goal pose is supported for distribution evaluation"
    os.makedirs(output_folder, exist_ok=True)  
    model.eval()
    
    rank = torch.distributed.get_rank()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)
    
    if rank == 0:
        tepoch = tqdm.tqdm(
            itertools.islice(dataloader, num_batches), 
            total=num_batches, 
            dynamic_ncols=True, 
            desc=f"Evaluating Regression Distribution", 
            leave=False)
    else:
        tepoch = itertools.islice(dataloader, num_batches)
        
    metric_accumulator = defaultdict(list)
    top_k_heap_dict = defaultdict(list) # key: metric name, value: list of (metric value, counter, current_image, goal_image)
    
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
        gt_action = gt_actions_with_initial.to(device, non_blocking=True)[:, 0] # B, 48
        first_pose = first_pose.to(device, non_blocking=True)[:, 0]
        
        pred_action = model(batch_obs_images, batch_goal_images, context_poses)
        target = gt_action
        
        loss = nn.functional.mse_loss(pred_action, target, reduction="none").mean(dim=1) # B,

        metrics = compute_metrics(pred_action.detach(), target.detach(), reduce_mean=False)
        metrics["loss"] = loss.detach()
            
        if torch.distributed.is_initialized():
            all_metrics_list = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_metrics_list, metrics)

            combined_metrics = defaultdict(list)
            for key in metrics.keys():
                for proc_metric in all_metrics_list:
                    combined_metrics[key].append(proc_metric[key].detach().cpu())
                combined_metrics[key] = torch.cat(combined_metrics[key], dim=0)
            metrics = combined_metrics
            
        metric_accumulator["loss"].append(metrics["loss"].detach().cpu().numpy())
        for key, value in metrics.items():
            if any(part in key for part in ["Pelvis", "Head", "Hand"]):
                metric_accumulator[key].append(value.detach().cpu().numpy())
                heap = top_k_heap_dict[key]
                # Update top_k_heap_dict for this metric
                for sample_idx in range(len(metrics[key])):
                    metric_value = float(metrics[key][sample_idx].item())
                    entry = (-metric_value, obs_images[sample_idx, -1].clone(), goal_image[sample_idx].clone())
                    if len(heap) < save_top_k_images:
                        heapq.heappush(heap, entry)
                    else:
                        if metric_value < -heap[0][0]:
                            heapq.heapreplace(heap, entry)
    # Concatenate all metric values and plot histograms
    if rank == 0:
        for metric_name, metric_values in metric_accumulator.items():
            # Concatenate all batches into a single array
            all_values = np.concatenate(metric_values, axis=0)

            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(all_values, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Value'); plt.ylabel('Frequency'); plt.title(f'Histogram of {metric_name}')
            plt.grid(True, alpha=0.3)
            
            # Log to wandb if available
            if use_wandb: wandb.log({f"histogram/{metric_name}": wandb.Image(plt)}, commit=False)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{metric_name.replace('/', '_')}_histogram.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plotted histogram for {metric_name} with {len(all_values)} values")
        
        for key, heap in top_k_heap_dict.items():
            metric_folder = os.path.join(output_folder, key.replace('/', '_'))
            os.makedirs(metric_folder, exist_ok=True)
            for idx, (neg_metric_value, obs_image, goal_image) in enumerate(heap):
                metric_value = -neg_metric_value
                plt.figure(figsize=(10, 6))
                plt.imshow(obs_image.permute(1, 2, 0))
                plt.title(f"Obs Image")
                plt.savefig(f"{metric_folder}/{idx}-obs_image-metric{metric_value:.4f}.png", dpi=150, bbox_inches='tight')
                plt.close()
                plt.figure(figsize=(10, 6))
                plt.imshow(goal_image.permute(1, 2, 0))
                plt.title(f"Goal Image")
                plt.savefig(f"{metric_folder}/{idx}-goal_image-metric{metric_value:.4f}.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        