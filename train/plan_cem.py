import argparse
from datetime import datetime
import os
import torch
import yaml
import copy
import wandb
import json
import random
from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from dreamsim import dreamsim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw

from peva.models import CDiT_models
from peva.diffusion import create_diffusion

from planning.cem import CEMPlanner
from planning.utils import load_peva, load_policy
from planning.nymeria_dataset import NymeriaPlanningDataset, build_planning_split
from planning.wrappers import EvaluatorPeva, EvaluatorWaypoint, ObjectiveDreamSIM, PevaWM, Preprocessor, WaypointWM
from planning.vis_utils import load_camera_model, get_T_C_pelvis, draw_image_coords, disable_logging

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

    rank = args.rank
    world_size = args.world_size
    gpu = rank % torch.cuda.device_count()
    # world_size, rank, gpu, is_distributed = init_distributed()
    print(f"NOT TRUE DISTRIBUTED THIS IS USED FOR DATALOADING ONLY -- Rank: {rank}, World size: {world_size}")
    torch.cuda.set_device(gpu)
    
    algo = args.algo
    
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f"{algo}_cem-h{args.horizon}-n{args.num_samples}-t{args.topk}-v{args.var_scale}-o{args.opt_steps}-N{args.num_eval_samples}-ds{args.peva_diffusion_steps}-dist{args.min_dist_cat}-{args.max_dist_cat}"
    if args.use_leafxyz_as_cost:
        run_name = "CHEATMETRIC_leafxyz_as_cost" + run_name
    if world_size > 1:
        run_name = run_name + f"-rank:ws-{rank}:{world_size}"
    if args.test:
        run_name = "test" + run_name
    if args.no_wandb or args.test:
        wandb_run = None
    else:
        wandb_run = wandb.init(project="peva-planning", name=run_name)

    if args.nymeria_raw_dir:
        disable_logging()
    nymeria_dp_cache = {}  # track_name -> (NymeriaDataProvider, cam_model)
    
    log_dir = f"logs/cem/{datetime_str}:{run_name}"
    # Only create log directory from rank 0 to avoid race conditions
    os.makedirs(log_dir, exist_ok=True)
    
    # load models
    device = 'cuda'
    if algo in "waypoint": 
        action_init = torch.ones(1, args.horizon, 8) * 0.5
        cem_planner, nomad_config, peva_config = build_waypoint_cem(args, wandb_run, log_dir, device)
    elif algo == "peva":
        action_init = None
        cem_planner, nomad_config, peva_config = build_peva_cem(args, wandb_run, log_dir, device)
        
    # prepare dataset
    data_config = nomad_config["datasets"]["nymeria"]
    context_size = max(args.peva_context_size - 1, nomad_config["context_size"])
    tasks_file = build_planning_split(
        data_folder=data_config["data_folder"],
        traj_names_file=os.path.join(data_config["test"], "traj_names.txt"),
        split_save_path=os.path.join(
            data_config["test"],
            f"planning_split_dist{args.min_dist_cat}-{args.max_dist_cat}"
            f"_ctx{context_size}_thresh{args.min_dist_threshold}"
            f"{'_keepnonvis' if args.keep_nonvisible_goal else ''}.pkl"
        ),
        context_size=context_size,
        min_dist_cat=args.min_dist_cat,
        max_dist_cat=args.max_dist_cat,
        min_dist_threshold=args.min_dist_threshold,
        keep_nonvisible_goal=args.keep_nonvisible_goal,
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        end_slack=data_config.get("end_slack", 0),
        curr_time_stride=args.curr_time_stride,
    )
    dataset = NymeriaPlanningDataset(
        tasks_file=tasks_file,
        data_folder=data_config["data_folder"],
        image_size=nomad_config["image_size"],
        context_size=context_size,
        goal_type=nomad_config.get("goal_type", None),
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
    )
    shuffle = args.shuffle
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1)

    count = 0
    for idx, batch in enumerate(dataloader):
        obs_images = batch["obs_images"]               # 1, context_size+1, 3, H, W
        goal_image = batch["goal_image"]               # 1, 3, H, W
        context_poses = batch["context_poses"]         # 1, context_size+1, 48
        deltas = batch["deltas"]                       # 1, n_steps, 48
        first_pose = batch["first_pose"]               # 1, 1, 48
        goal_pose = batch["goal_pose"]                 # 1, 1, 48
        xsens_offsets = batch["xsens_offsets"]         # 1, 15, 3
        goal_obs = batch["goal_obs"]                   # 1, 3, H, W
        goal_image_coords = batch["goal_image_coords"] # 1, 23, 2
        gt_frames = batch["gt_frames"]                 # 1, n_steps, 3, H, W
        gt_image_coords_seq = batch["gt_image_coords_seq"]  # 1, n_steps, 23, 2

        dataset_index = batch["dataset_index"].item()
        track = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index = batch["goal_index"].item()
        track_idx_name = f"{track}-s{start_index}-g{goal_index}"

        print("="*50)
        print(f"Planning {track_idx_name}")

        task_dir = f"{log_dir}/{track_idx_name}"
        os.makedirs(task_dir, exist_ok=True)

        # --- Per-task setup save ---
        # Context frames + current frame + goal frame
        save_image(obs_images[0], f"{task_dir}/context_frames.png", nrow=obs_images.shape[1])
        save_image(goal_obs[0], f"{task_dir}/goal_obs.png")
        save_image(goal_image[0], f"{task_dir}/goal_image.png")
        # GT trajectory frames
        save_image(gt_frames[0], f"{task_dir}/gt_frames.png", nrow=gt_frames.shape[1])
        # GT skeleton overlays on current observation
        curr_image = obs_images[0, -1]  # (3, H, W)
        n_steps = gt_frames.shape[1]
        gt_skel_imgs = []
        for t in range(n_steps):
            img_pil = Image.fromarray((255.0 * curr_image.permute(1, 2, 0)).to(torch.uint8).numpy())
            draw = ImageDraw.Draw(img_pil)
            coords = gt_image_coords_seq[0, t]  # (23, 2)
            draw_image_coords(draw, coords[None])
            gt_skel_imgs.append(transforms.ToTensor()(img_pil))
        save_image(torch.stack(gt_skel_imgs), f"{task_dir}/gt_skeleton_overlays.png", nrow=n_steps)
        # Numerical data
        torch.save({
            "start_index": start_index,
            "goal_index": goal_index,
            "first_pose": first_pose[0],
            "goal_pose": goal_pose[0],
            "context_poses": context_poses[0],
            "deltas": deltas[0],
        }, f"{task_dir}/task_data.pth")

        # --- NymeriaDataProvider for skeleton projection ---
        if args.nymeria_raw_dir:
            if track not in nymeria_dp_cache:
                from nymeria.data_provider import NymeriaDataProvider
                nymeria_dp_cache[track] = (
                    NymeriaDataProvider(
                        sequence_rootdir=Path(os.path.join(args.nymeria_raw_dir, track)),
                        load_wrist=False, load_observer=False,
                    ),
                    load_camera_model(args.nymeria_raw_dir, track),
                )
            nymeria_dp, cam_model = nymeria_dp_cache[track]
            T_C_pelvis = get_T_C_pelvis(nymeria_dp, start_index)
        else:
            cam_model, T_C_pelvis = None, None

        obs_0 = {"images": obs_images,
                 "goal_image": goal_image,
                 "context_poses": context_poses}
        obs_g = {"images": goal_obs,
                 "deltas": deltas,
                 "first_pose": first_pose,
                 "xsens_offsets": xsens_offsets,
                 "goal_image_coords": goal_image_coords}

        cem_planner.plan(obs_0, obs_g, track_idx_name, actions=action_init,
                         cam_model=cam_model, T_C_pelvis=T_C_pelvis)
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
    ),
    "draw_mask": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_21_06_54:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth"
    ),
    "heldout": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask-heldoutEnvs.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_24_06_25:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask-heldoutEnvs/ema_9.pth"
    ),
    "draw_mask_heldout": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask-heldoutEnvs.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_21_06_54:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth"
    )
}
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--algo", type=str, choices=["peva", "waypoint"], default="waypoint", help="Planning algorithm")
    parser.add_argument("--use_leafxyz_as_cost", action='store_true', help="Uses the metric(leaf-xyz) instead of a normal cost_fn")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")

    parser.add_argument("-n", "--num_samples", type=int, default=32, help="Number of samples")
    parser.add_argument("-t", "--topk", type=int, default=4, help="Top k samples")
    parser.add_argument("-v", "--var_scale", type=float, default=0.5, help="Variance scale")
    parser.add_argument("-o", "--opt_steps", type=int, default=8, help="Optimization steps")
    parser.add_argument("-H", "--horizon", type=int, default=1, help="Time horizon")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=1, help="Number of evaluation samples")

    parser.add_argument("--keep_nonvisible_goal", action="store_true", help="Keep non-visible goal in the dataset")
    parser.add_argument("--min_dist_cat", type=int, default=8, help="Minimum goal distance in frames")
    parser.add_argument("--max_dist_cat", type=int, default=8, help="Maximum goal distance in frames (max 32 given projection window)")
    parser.add_argument("--curr_time_stride", type=int, default=1, help="Stride when iterating start times during split building")
    parser.add_argument("--min_dist_threshold", type=float, default=0.1, help="Minimum distance threshold")
    parser.add_argument("--num_samples_to_plan", type=int, default=32, help="Number of samples to plan")
    parser.add_argument("--no_wandb", action="store_true", help="Don't use wandb")
    parser.add_argument("--test", action="store_true", help="Test run")
    parser.add_argument("--nymeria_raw_dir", type=str, default=None,
                        help="Path to raw Nymeria data (for skeleton overlay projection). "
                             "If not provided, skeleton overlays are skipped.")
    
    parser.add_argument("--peva_config", type=str, default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str, default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15, help="PEVA context size")
    parser.add_argument("--peva_diffusion_steps", type=int, default=250, help="PEVA diffusion steps")
    
    parser.add_argument("--nomad_model", type=str, default="draw", choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)
    
    parser.add_argument("--world_size", type=int, default=1, help="World size")
    parser.add_argument("--rank", type=int, default=0, help="Rank")
    
    args = parser.parse_args()
    
    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    else:
        assert args.nomad_config is not None and args.nomad_checkpoint is not None
    
    main(args)