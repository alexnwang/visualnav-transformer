import sys
sys.path.append("/home/anw2067/visualnav-transformer/train")

import argparse
from datetime import datetime
import os
import torch
import yaml
import copy
import wandb
import json
import random

from nymeria.download_utils import DownloadManager
from nymeria.definitions import DataGroups
from nymeria.data_provider import SequencePathProvider, NymeriaDataProvider
from nymeria.definitions import Subpaths, VrsFiles
from nymeria.recording_data_provider import create_recording_data_provider

import numpy as np
from torchvision import transforms
from dreamsim import dreamsim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader
from diffusers.models import AutoencoderKL

from peva.models import CDiT_models
from peva.diffusion import create_diffusion

from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from vint_train.data.misc import XSensConstants, XsensSkeleton
from planning.utils import _compute_pose_and_loss, _compute_part_distance_matrices
from planning.cem import CEMPlanner
from planning.utils import get_nymeria_dataset, load_peva, load_policy
from planning.wrappers import EvaluatorPeva, EvaluatorWaypoint, ObjectiveDreamSIM, PevaWM, Preprocessor, WaypointWM
from planning.sampling import waypoint_sample
from planning.vis_utils import *

from torchvision.utils import save_image

OUTPUT_DIR = "/home/anw2067/visualnav-transformer/train/logs/figures/paper/paper_vis/teaser"
DATA_SAVE_DIR = "/home/anw2067/scratch/nymeria_camera_dir"
DATA_JSON="/home/anw2067/visualnav-transformer/data_jsons/visibility_no_data.json"

os.makedirs(DATA_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, policy_diffusion, nomad_stats, nomad_config = load_policy(args.nomad_config, args.nomad_checkpoint, device=device)
    peva_model, _, peva_diffusion, peva_vae, peva_stats, peva_config = load_peva(args.peva_config, args.peva_checkpoint, device=device,
                                                                    inference_context_size=args.peva_context_size,
                                                                    diffusion_steps=args.peva_diffusion_steps)
    
    dataset = get_nymeria_dataset(nomad_config, context_size=max(args.peva_context_size-1, nomad_config["context_size"]), goal_timestep_offset=args.goal_timestep_offset)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=args.shuffle, seed=seed)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1)
    
    prev_track_name = None
    for idx, batch in enumerate(dataloader):
        if idx > args.num_samples_to_vis: break
        obs_images = batch["obs_images"] # 1, context_size, 3, H, W
        goal_image = batch["goal_image"] # 1, 3, H, W
        context_poses = batch["context_poses"] # 1, context_size, 48

        deltas = batch["deltas"] # 1, horizon, action_dim
        first_pose = batch["first_pose"] # 1, 1, 48
        xsens_offsets = batch["xsens_offsets"][0] # 1, 15, 3
        goal_obs = batch["goal_obs"] # 1, 3, H, W
        goal_image_coords = batch["goal_image_coords"] # 1, 23, 2
        
        dataset_index = batch["dataset_index"].item()
        track_name, track_index = batch["dataset_track"][0], batch["dataset_track_index"].item()
        track_idx_name = f"{track_name}-{track_index}"
        
        skel = XsensSkeleton(xsens_offsets)
        gt_actions = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        xyz_dist_matrix, _, init_xyz, _ = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        visible_plus_head = (goal_image_coords != -1).all(dim=-1)[:, :XSensConstants.upper_body_num_parts] # B, num_parts
        visible_plus_head[:, XSensConstants.part_names.index("Head")] = True
        init_visible_plus_head = xyz_dist_matrix[:, XSensConstants.leaf_indices] * visible_plus_head[:, XSensConstants.leaf_indices]
        init_visible_plus_head = (init_visible_plus_head.sum() / visible_plus_head.sum()).item()
        
        if not args.keep_nonvisible_goal:
            visible = False
            find_count = 0
            for part in ["Pelvis", "Head", "R_Hand", "L_Hand"]:
                index = XSensConstants.part_names.index(part)
                if all(goal_image_coords[0, index] != -1):
                    find_count += 1
                    if find_count > 0:
                        visible = True
                        break
            if not visible:
                print(f"No visible parts in {track_idx_name}")
                continue
            
        if init_visible_plus_head < args.min_dist_threshold:
            print(f"Initial distance of visible + head joints is less than {args.min_dist_threshold} in {track_idx_name}")
            continue
        print(f"Visualizing {track_idx_name}")
        
        curr_save_dir = f"{OUTPUT_DIR}/{track_idx_name}"
        os.makedirs(curr_save_dir, exist_ok=True)
        save_img = torch.cat([obs_images, goal_obs[None],torch.zeros_like(obs_images[:, :-2]), goal_image[None]], dim=1)[0]
        save_image(save_img, f"{curr_save_dir}/context_and_goal.png", nrow=obs_images.shape[1])
        
        obs_images = obs_images.to(device)
        context_poses = context_poses.to(device)
        goal_obs = goal_obs.to(device)
        goal_image_coords = goal_image_coords.to(device)
        first_pose = first_pose.to(device)
        deltas = deltas.to(device)
        
        # config details
        policy_pred_horizon = nomad_config["len_traj_pred"]
        policy_action_dim = nomad_config["input_dims"]
        image_size = nomad_config["image_size"][0]
        policy_context_size = nomad_config["context_size"] + 1
        peva_context_size = peva_config["context_size"]
        peva_latent_size = image_size // 8
        
        # Duplicate all batches by 64
        batch_size_dup = args.num_batch_repeats

        obs_images = obs_images.repeat(batch_size_dup, 1, 1, 1, 1)
        context_poses = context_poses.repeat(batch_size_dup, 1, 1)
        goal_obs = goal_obs.repeat(batch_size_dup, 1, 1, 1)
        deltas = deltas.repeat(batch_size_dup, 1, 1)
        first_pose = first_pose.repeat(batch_size_dup, 1, 1)
        goal_image_coords = goal_image_coords.repeat(batch_size_dup, 1, 1)
        
        waypoints = goal_image_coords[:, XSensConstants.leaf_indices].flatten(1, 2)[:, None] # 1, 4, 2
        policy_context_poses = context_poses[:, -policy_context_size:]
        
        (
            pred_frames, # B, W=1, T=8, 3, H, W
            pred_delta, # B, W=1, T=8, 48
            goal_obs_accum # B, W=1, C, H, W
        ) = waypoint_sample(policy, policy_diffusion,
                            peva_model, peva_diffusion, peva_vae, peva_stats,
                            waypoints, policy_context_poses, obs_images, goal_obs,
                            policy_pred_horizon, policy_action_dim,
                            image_size, 
                            policy_context_size, peva_context_size, peva_latent_size,
                            device,
                            skip_last_peva=False,
                            gt_deltas=None,
                            first_pose=None)
        pred_frames = pred_frames.flatten(0, 1) # B*W=1, T, 3, H, W
        pred_delta = pred_delta.flatten(0, 1) # B*W=1, T, 48
        
        pred_actions = get_action_smpl_torch(first_pose, pred_delta, XSensConstants.upper_body_num_parts)
        gt_actions = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts)
        
        _, _, leaf_xyz, _ = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        argmin_idx = leaf_xyz.argmin(dim=-1)
        best_leaf_xyz = leaf_xyz[argmin_idx]
        
        obs_images = obs_images[argmin_idx]
        context_poses = context_poses[argmin_idx]
        goal_obs = goal_obs[argmin_idx]
        deltas = deltas[argmin_idx]
        first_pose = first_pose[argmin_idx]
        goal_image_coords = goal_image_coords[argmin_idx]
        skel = XsensSkeleton(xsens_offsets)
        pred_frames = pred_frames[argmin_idx]
        pred_delta = pred_delta[argmin_idx]
        
        if not os.path.exists(os.path.join(DATA_SAVE_DIR, track_name)):
            os.makedirs(os.path.join(DATA_SAVE_DIR, track_name))
            print(f"downloading episode {track_name}")
            download_episode(DATA_JSON, DATA_SAVE_DIR, track_name)
        
        if track_name != prev_track_name:
            nymeria_dp = NymeriaDataProvider(sequence_rootdir=Path(os.path.join(DATA_SAVE_DIR, track_name)), load_wrist=False, load_observer=False)
            cam_model = load_camera_model(DATA_SAVE_DIR, track_name)
        prev_track_name = track_name
        
        T_C_Pelvis = get_T_C_pelvis(nymeria_dp, track_index)
        
        # vis_image = obs_images[0, -1].detach().cpu()# B, 3, H, W
        vis_image = goal_image[0]
        drawn_images = []
        for t in range(pred_actions.shape[1]):
            color_multiplier = t / pred_actions.shape[1]
            image = Image.fromarray((255.*vis_image.permute(1, 2, 0)).to(torch.uint8).numpy())
            draw = ImageDraw.Draw(image)
            
            gc_image_coords = pose_to_image_coords(pred_actions[:, t], cam_model, xsens_offsets, T_C_Pelvis) # B, 15, 2
            gc_vis = draw_image_coords(draw, gc_image_coords, color=(int(255), int(255), int(255)), show_text=True)
            image.save(os.path.join(curr_save_dir, f"action-t{t}-gc{gc_vis}.png"))
            drawn_images.append(transforms.ToTensor()(image))
        
        for t in range(pred_frames.shape[1]):
            save_image(pred_frames[0, t].detach().cpu(), f"{curr_save_dir}/pred_frame-t{t}.png")
            
        save_image(goal_image[0].detach().cpu(), f"{curr_save_dir}/goal_image.png")
        save_image(goal_obs.detach().cpu(), f"{curr_save_dir}/goal_obs.png")
        
        image_list = [
            goal_image[0],
            *drawn_images,
            torch.ones_like(goal_image[0]),
            torch.ones_like(goal_image[0]),
        ]
        image_tensor = torch.stack(image_list, dim=0)
        image_tensor = torch.cat([image_tensor, pred_frames.detach().cpu(), goal_obs[None].detach().cpu()], dim=0)
        save_image(image_tensor, f"{curr_save_dir}/stacked_images.png", nrow=image_tensor.shape[0] // 2)
        save_image(image_tensor, f"{OUTPUT_DIR}/{track_idx_name}.png", nrow=image_tensor.shape[0] // 2)
                
        

MODEL_DIRECTORY={
    "draw": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/ema_9.pth"
    ),
    "gravity": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_18_11_47:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-preserveUpDown/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_18_11_47:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-preserveUpDown/ema_9.pth"
    ),
    "draw_mask": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_21_06_54:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth"
    )
}
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--algo", type=str, choices=["peva", "waypoint"], default="waypoint", help="Planning algorithm")
    parser.add_argument("--use_leafxyz_as_cost", action='store_true', help="Uses the metric(leaf-xyz) instead of a normal cost_fn")
    parser.add_argument("--goal_timestep_offset", type=int, default=None, help="Goal timestep offset")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    
    # # CEM parameters
    # parser.add_argument("-n", "--num_samples", type=int, default=32, help="Number of samples")
    # parser.add_argument("-t", "--topk", type=int, default=4, help="Top k samples")
    # parser.add_argument("-v", "--var_scale", type=float, default=0.5, help="Variance scale")
    # parser.add_argument("-o", "--opt_steps", type=int, default=8, help="Optimization steps")
    # parser.add_argument("-e", "--eval_every", type=int, default=1, help="Evaluation frequency")
    # parser.add_argument("-H", "--horizon", type=int, default=1, help="Time horizon")
    # parser.add_argument("-N", "--num_eval_samples", type=int, default=1, help="Number of evaluation samples")
    
    parser.add_argument("--num_batch_repeats", type=int, default=8, help="Number of batch repeats")
    parser.add_argument("--keep_nonvisible_goal", action="store_true", help="Keep non-visible goal in the dataset")
    parser.add_argument("--min_index_goal", type=int, default=0, help="Minimum index of the goal to plan")
    parser.add_argument("--min_dist_threshold", type=float, default=0.1, help="Minimum distance threshold")
    parser.add_argument("--num_samples_to_vis", type=int, default=64, help="Number of samples to plan")
    parser.add_argument("--no_wandb", action="store_true", help="Don't use wandb")
    parser.add_argument("--test", action="store_true", help="Test run")
    
    parser.add_argument("--peva_config", type=str, default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str, default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15, help="PEVA context size")
    parser.add_argument("--peva_diffusion_steps", type=int, default=250, help="PEVA diffusion steps")
    
    parser.add_argument("--nomad_model", type=str, default="draw", choices=["draw", "gravity", "draw_mask"])
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)
    
    parser.add_argument("--world_size", type=int, default=1, help="World size")
    parser.add_argument("--rank", type=int, default=0, help="Rank")
    
    # In Jupyter notebooks, pass arguments as a list to parse_args() instead of using sys.argv
    # Pass an empty list [] to use all defaults, or specify arguments like: ['--shuffle', '--nomad_model', 'draw']
    # args = parser.parse_args(["--shuffle", "--nomad_model", "draw_mask", "--peva_context_size", "7"])
    args = parser.parse_args()
    
    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    else:
        assert args.nomad_config is not None and args.nomad_checkpoint is not None
    
    main(args)