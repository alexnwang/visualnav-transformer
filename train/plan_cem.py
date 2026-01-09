import argparse
from datetime import datetime
import torch
import yaml
import copy
import wandb
import json

import numpy as np
from torchvision import transforms
from dreamsim import dreamsim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import RandomSampler, DataLoader
from diffusers.models import AutoencoderKL

from peva.models import CDiT_models
from peva.diffusion import create_diffusion

from vint_train.data.misc import XSensConstants
from planning.cem import CEMPlanner
from planning.utils import get_nymeria_dataset, load_peva, load_policy
from planning.wrappers import Evaluator, ObjectiveDreamSIM, Preprocessor, WaypointWM

from torchvision.utils import save_image

def main(args):
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f"wapoint_cem-h{args.horizon}-n{args.num_samples}-t{args.topk}-v{args.var_scale}-o{args.opt_steps}-N{args.num_eval_samples}"
    if args.no_wandb:
        wandb_run = None
    else:
        wandb_run = wandb.init(project="peva-planning", name=run_name)
    
    # load models
    policy, policy_diffusion, nomad_stats, nomad_config = load_policy(args.nomad_config, args.nomad_checkpoint, device='cuda')
    model, _, peva_diffusion, vae, peva_stats, peva_config = load_peva(args.peva_config, args.peva_checkpoint, device='cuda',
                                                                  inference_context_size=args.peva_context_size,
                                                                  diffusion_steps=args.peva_diffusion_steps)
    
    # construct wrappers and CEM planner
    wm_wrapper = WaypointWM(model, peva_diffusion, vae, peva_stats, policy, policy_diffusion,
                nomad_config["image_size"][0], peva_config["context_size"], nomad_config["context_size"]+1,
                nomad_config["len_traj_pred"], nomad_config["input_dims"])   
    evaluator = Evaluator(model, peva_diffusion, vae, peva_stats, policy, policy_diffusion,
                    nomad_config["image_size"][0], peva_config["context_size"], nomad_config["context_size"]+1,
                    nomad_config["len_traj_pred"], nomad_config["input_dims"],
                    num_eval_samples=args.num_eval_samples)
    objective_fn = ObjectiveDreamSIM(device="cuda")
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
        logging_prefix=run_name,
        log_dir=f"logs/cem/{datetime_str}:{run_name}"
    )
    
    dataset = get_nymeria_dataset(nomad_config, context_size=args.peva_context_size-1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    count = 0
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
        obs_0 = {"images": obs_images,
                 "goal_image": goal_image,
                 "context_poses": context_poses}
        obs_g = {"images": goal_obs,
                 "deltas": deltas,
                 "first_pose": first_pose,
                 "xsens_offsets": xsens_offsets,
                 "goal_image_coords": goal_image_coords}

        cem_planner.plan(obs_0, obs_g, actions=torch.ones(1, args.horizon, 8) * 0.5)
        count += 1
        if args.num_samples_to_plan > 0 and count > args.num_samples_to_plan: break
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n", "--num_samples", type=int, default=32, help="Number of samples")
    parser.add_argument("-t", "--topk", type=int, default=4, help="Top k samples")
    parser.add_argument("-v", "--var_scale", type=float, default=0.5, help="Variance scale")
    parser.add_argument("-o", "--opt_steps", type=int, default=8, help="Optimization steps")
    parser.add_argument("-e", "--eval_every", type=int, default=1, help="Evaluation frequency")
    parser.add_argument("-H", "--horizon", type=int, default=1, help="Time horizon")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=1, help="Number of evaluation samples")
    
    parser.add_argument("--keep_nonvisible_goal", action="store_true", help="Keep non-visible goal in the dataset")
    parser.add_argument("--num_samples_to_plan", type=int, default=32, help="Number of samples to plan")
    parser.add_argument("--no_wandb", action="store_true", help="Don't use wandb")
    
    parser.add_argument("--peva_config", type=str, default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str, default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15, help="PEVA context size")
    parser.add_argument("--peva_diffusion_steps", type=int, default=250, help="PEVA diffusion steps")
    
    parser.add_argument("--nomad_config", type=str, default="/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/config.yaml")
    parser.add_argument("--nomad_checkpoint", type=str, default="/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/ema_9.pth")
    
    args = parser.parse_args()
    
    main(args)