import argparse
from datetime import datetime
import os
import torch
import yaml
import copy
import wandb
import json
import random

import numpy as np
from torchvision import transforms
from dreamsim import dreamsim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader
from diffusers.models import AutoencoderKL

from peva.models import CDiT_models
from peva.diffusion import create_diffusion

from vint_train.data.misc import XSensConstants
from planning.cem import CEMPlanner
from planning.utils import get_nymeria_dataset, load_peva, load_policy
from planning.wrappers import EvaluatorPeva, EvaluatorWaypoint, ObjectiveDreamSIM, PevaWM, Preprocessor, WaypointWM

from torchvision.utils import save_image

from train_ddp import init_distributed

def build_peva_cem(args, wandb_run, log_dir, device):
    policy, policy_diffusion, nomad_stats, nomad_config = load_policy(args.nomad_config, args.nomad_checkpoint, device=device)
    model, _, peva_diffusion, vae, peva_stats, peva_config = load_peva(args.peva_config, args.peva_checkpoint, device=device,
                                                                inference_context_size=args.peva_context_size,
                                                                diffusion_steps=args.peva_diffusion_steps)
    
    # construct wrappers and CEM planner
    wm_wrapper = PevaWM(model, peva_diffusion, vae, peva_stats,
                        nomad_config["image_size"][0], peva_config["context_size"])   
    evaluator = EvaluatorPeva(model, peva_diffusion, vae, peva_stats,
                    nomad_config["image_size"][0], peva_config["context_size"],
                    num_eval_samples=args.num_eval_samples)
    objective_fn = ObjectiveDreamSIM(pred_horizon=args.horizon, device=device, return_metric=args.use_leafxyz_as_cost)
    preprocessor = Preprocessor()
    cem_planner = CEMPlanner(
        horizon=args.horizon,
        topk=args.topk,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        opt_steps=args.opt_steps,
        eval_every=args.eval_every,
        wm=wm_wrapper,
        action_dim=48,
        objective_fn=objective_fn,
        preprocessor=preprocessor,
        evaluator=evaluator,
        wandb_run=wandb_run,
        log_dir=log_dir
    )
    return cem_planner, nomad_config, peva_config

def build_waypoint_cem(args, wandb_run, log_dir, device):
    policy, policy_diffusion, nomad_stats, nomad_config = load_policy(args.nomad_config, args.nomad_checkpoint, device=device)
    model, _, peva_diffusion, vae, peva_stats, peva_config = load_peva(args.peva_config, args.peva_checkpoint, device=device,
                                                                inference_context_size=args.peva_context_size,
                                                                diffusion_steps=args.peva_diffusion_steps)
    
    # construct wrappers and CEM planner
    wm_wrapper = WaypointWM(model, peva_diffusion, vae, peva_stats, policy, policy_diffusion,
                nomad_config["image_size"][0], peva_config["context_size"], nomad_config["context_size"]+1,
                nomad_config["len_traj_pred"], nomad_config["input_dims"])   
    evaluator = EvaluatorWaypoint(model, peva_diffusion, vae, peva_stats, policy, policy_diffusion,
                    nomad_config["image_size"][0], peva_config["context_size"], nomad_config["context_size"]+1,
                    nomad_config["len_traj_pred"], nomad_config["input_dims"],
                    num_eval_samples=args.num_eval_samples)
    objective_fn = ObjectiveDreamSIM(pred_horizon=nomad_config["len_traj_pred"], device=device, return_metric=args.use_leafxyz_as_cost)
    preprocessor = Preprocessor()
    cem_planner = CEMPlanner(
        horizon=args.horizon,
        topk=args.topk,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        opt_steps=args.opt_steps,
        eval_every=args.eval_every,
        wm=wm_wrapper,
        action_dim=8,
        objective_fn=objective_fn,
        preprocessor=preprocessor,
        evaluator=evaluator,
        wandb_run=wandb_run,
        log_dir=log_dir
    )
    return cem_planner, nomad_config, peva_config

def main(args):
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    
    world_size, rank, gpu, is_distributed = init_distributed()
    print(f"Rank: {rank}, World size: {world_size}, GPU: {gpu}, Is distributed: {is_distributed}")
    torch.cuda.set_device(gpu)
    
    algo = args.algo
    
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    track_idx_name = f"{algo}_cem-h{args.horizon}-n{args.num_samples}-t{args.topk}-v{args.var_scale}-o{args.opt_steps}-N{args.num_eval_samples}-ds{args.peva_diffusion_steps}"
    if args.use_leafxyz_as_cost:
        track_idx_name = "CHEATMETRIC_leafxyz_as_cost" + track_idx_name
    if args.goal_timestep_offset is not None:
        track_idx_name = track_idx_name + f"-gt{args.goal_timestep_offset}"
    if args.test:
        track_idx_name = "test" + track_idx_name
    if args.no_wandb or args.test:
        wandb_run = None
    else:
        # Only initialize wandb from rank 0 in distributed setting
        if is_distributed and rank != 0:
            wandb_run = None
        else:
            wandb_run = wandb.init(project="peva-planning", name=track_idx_name)
    
    log_dir = f"logs/cem/{datetime_str}:{track_idx_name}"
    # Only create log directory from rank 0 to avoid race conditions
    if not is_distributed or rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    # Synchronize all processes before proceeding
    if is_distributed:
        torch.distributed.barrier()
    
    # load models
    device = f'cuda:{gpu}' if is_distributed else 'cuda'
    if algo == "waypoint": 
        action_init = torch.ones(1, args.horizon, 8) * 0.5
        cem_planner, nomad_config, peva_config = build_waypoint_cem(args, wandb_run, log_dir, device)
    elif algo == "peva":
        action_init = None
        cem_planner, nomad_config, peva_config = build_peva_cem(args, wandb_run, log_dir, device)
        
    # prepare dataset
    shuffle = False
    dataset = get_nymeria_dataset(nomad_config, context_size=max(args.peva_context_size-1, nomad_config["context_size"]), goal_timestep_offset=args.goal_timestep_offset)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1)
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=1, generator=generator)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    count = 0
    curr_track, curr_index = None, None
    for idx, batch in enumerate(dataloader):
        obs_images = batch["obs_images"] # 1, context_size, 3, H, W
        goal_image = batch["goal_image"] # 1, 3, H, W
        context_poses = batch["context_poses"] # 1, context_size, 48

        deltas = batch["deltas"] # 1, horizon, action_dim
        first_pose = batch["first_pose"] # 1, 1, 48
        xsens_offsets = batch["xsens_offsets"] # 15, 3
        goal_obs = batch["goal_obs"] # 1, 3, H, W
        goal_image_coords = batch["goal_image_coords"] # 1, 23, 2
        
        if not args.keep_nonvisible_goal:
            visible = False
            find_count = 0
            for part in ["Pelvis", "Head", "R_Hand", "L_Hand"]:
                index = XSensConstants.part_names.index(part)
                if all(goal_image_coords[0, index] != -1):
                    find_count += 1
                    if find_count >=3:
                        visible = True
                        break
            if not visible:
                continue
        assert shuffle == False, "shuffle must be False for dataloader"
        track, track_index, _ = dataloader.dataset.index_to_data[idx]
        track_idx_name = f"{track}-{track_index}"
        
        if curr_track == track and idx - curr_index < args.min_index_goal:
            continue
        curr_track = track
        curr_index = idx
        
        print("="*50)
        print(f"Planning {track_idx_name}")
        
        # visualize the context and goal images
        os.makedirs(f"{log_dir}/{track_idx_name}")
        save_img = torch.cat([obs_images, torch.zeros_like(obs_images[:, :-1]), goal_image[None]], dim=1)[0]
        save_image(save_img, f"{log_dir}/{track_idx_name}/context_and_goal.png", nrow=obs_images.shape[1])
        
            
        obs_0 = {"images": obs_images,
                 "goal_image": goal_image,
                 "context_poses": context_poses}
        obs_g = {"images": goal_obs,
                 "deltas": deltas,
                 "first_pose": first_pose,
                 "xsens_offsets": xsens_offsets,
                 "goal_image_coords": goal_image_coords}

        cem_planner.plan(obs_0, obs_g, track_idx_name, actions=action_init)
        count += 1
        if args.num_samples_to_plan > 0 and count > args.num_samples_to_plan: break
        
        
MODEL_DIRECTORY={
    "draw": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/ema_9.pth"
    ),
    "gravity": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_18_11_47:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-preserveUpDown/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_18_11_47:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-preserveUpDown/ema_9.pth"
    )
}
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--algo", type=str, choices=["peva", "waypoint"], default="waypoint", help="Planning algorithm")
    parser.add_argument("--use_leafxyz_as_cost", action='store_true', help="Uses the metric(leaf-xyz) instead of a normal cost_fn")
    parser.add_argument("--goal_timestep_offset", type=int, default=None, help="Goal timestep offset")
    
    parser.add_argument("-n", "--num_samples", type=int, default=32, help="Number of samples")
    parser.add_argument("-t", "--topk", type=int, default=4, help="Top k samples")
    parser.add_argument("-v", "--var_scale", type=float, default=0.5, help="Variance scale")
    parser.add_argument("-o", "--opt_steps", type=int, default=8, help="Optimization steps")
    parser.add_argument("-e", "--eval_every", type=int, default=1, help="Evaluation frequency")
    parser.add_argument("-H", "--horizon", type=int, default=1, help="Time horizon")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=1, help="Number of evaluation samples")
    
    parser.add_argument("--keep_nonvisible_goal", action="store_true", help="Keep non-visible goal in the dataset")
    parser.add_argument("--min_index_goal", type=int, default=80, help="Minimum index of the goal to plan")
    parser.add_argument("--num_samples_to_plan", type=int, default=32, help="Number of samples to plan")
    parser.add_argument("--no_wandb", action="store_true", help="Don't use wandb")
    parser.add_argument("--test", action="store_true", help="Test run")
    
    parser.add_argument("--peva_config", type=str, default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str, default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15, help="PEVA context size")
    parser.add_argument("--peva_diffusion_steps", type=int, default=250, help="PEVA diffusion steps")
    
    parser.add_argument("--nomad_model", type=str, default="draw", choices=["draw", "gravity"])
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    else:
        assert args.nomad_config is not None and args.nomad_checkpoint is not None
    
    main(args)