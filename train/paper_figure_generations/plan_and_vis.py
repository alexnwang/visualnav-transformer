import sys
sys.path.append("/home/anw2067/visualnav-transformer/train")

import argparse
from datetime import datetime
import os
import random
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DistributedSampler, DataLoader
from PIL import Image, ImageDraw

from nymeria.data_provider import NymeriaDataProvider

from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from vint_train.data.misc import XSensConstants, XsensSkeleton

from planning.utils import _compute_part_distance_matrices, get_nymeria_dataset, load_peva, load_policy
from planning.wrappers import EvaluatorWaypoint, ObjectiveDreamSIM, Preprocessor, WaypointWM
from planning.cem import CEMPlanner
from planning.sampling import waypoint_sample
from planning.vis_utils import *

OUTPUT_DIR = "/home/anw2067/visualnav-transformer/train/logs/figures/paper/paper_vis/plan_and_vis"
DATA_SAVE_DIR = "/home/anw2067/scratch/nymeria_camera_dir"
DATA_JSON = "/home/anw2067/visualnav-transformer/data_jsons/visibility_no_data.json"

os.makedirs(DATA_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIRECTORY = {
    "draw": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/ema_9.pth",
    ),
    "gravity": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_18_11_47:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-preserveUpDown/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_18_11_47:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-preserveUpDown/ema_9.pth",
    ),
    "draw_mask": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_21_06_54:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth",
    ),
    "heldout": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask-heldoutEnvs.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_24_06_25:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask-heldoutEnvs/ema_9.pth",
    ),
    "draw_mask_heldout": (
        "/home/anw2067/visualnav-transformer/train/config/torch/minimal-nomad-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask-heldoutEnvs.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_01_21_06_54:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth",
    ),
}


def visualize_best_sample(
    sel_idx,
    pred_actions,   # B, T, 48
    pred_frames,    # B, T, 3, H, W
    obs_images_rep, # B, ctx, 3, H, W
    goal_image,     # 1, 3, H, W
    goal_obs,       # 1, 3, H, W
    goal_image_coords,  # 1, 23, 2
    xsens_offsets,      # 15, 3
    cam_model,
    T_C_Pelvis,
    curr_save_dir,
    metric_tag,
    log_dir,
    track_idx_name,
):
    obs_sel = obs_images_rep[sel_idx]             # ctx, 3, H, W
    pred_frames_sel = pred_frames[sel_idx]        # T, 3, H, W
    pred_actions_sel = pred_actions[sel_idx:sel_idx+1]  # 1, T, 48  (batch dim for pose_to_image_coords)
    goal_coords_sel = goal_image_coords[0]        # 23, 2

    # Waypoint-annotated image: goal image with leaf joints drawn at larger radius
    wp_image = Image.fromarray((255. * goal_image[0].permute(1, 2, 0)).to(torch.uint8).numpy())
    wp_draw = ImageDraw.Draw(wp_image)
    draw_image_coords(wp_draw, goal_coords_sel[None], color=(255, 200, 50), show_text=False, radius=7)
    wp_image.save(os.path.join(curr_save_dir, "waypoint_annotated.png"))
    waypoint_annotated_tensor = transforms.ToTensor()(wp_image)

    # Per-timestep action images on goal image
    drawn_images = []
    for t in range(pred_actions_sel.shape[1]):
        img = Image.fromarray((255. * goal_image[0].permute(1, 2, 0)).to(torch.uint8).numpy())
        draw = ImageDraw.Draw(img)
        gc_coords = pose_to_image_coords(pred_actions_sel[:, t], cam_model, xsens_offsets, T_C_Pelvis)
        gc_vis = draw_image_coords(draw, gc_coords, color=(255, 255, 255), show_text=True)
        img.save(os.path.join(curr_save_dir, f"action-t{t}-gc{gc_vis}.png"))
        drawn_images.append(transforms.ToTensor()(img))

    # All actions accumulated on the final context frame
    final_ctx_image = Image.fromarray((255. * obs_sel[-1].permute(1, 2, 0)).to(torch.uint8).numpy())
    all_actions_draw = ImageDraw.Draw(final_ctx_image)
    n_t = pred_actions_sel.shape[1]
    for t in range(n_t):
        alpha = int(80 + 175 * (t / max(n_t - 1, 1)))
        gc_coords_t = pose_to_image_coords(pred_actions_sel[:, t], cam_model, xsens_offsets, T_C_Pelvis)
        draw_image_coords(all_actions_draw, gc_coords_t, color=(alpha, alpha, alpha), show_text=False)
    final_ctx_image.save(os.path.join(curr_save_dir, "all_actions.png"))
    all_actions_tensor = transforms.ToTensor()(final_ctx_image)

    # Save all waypoint generation frames
    pred_frames_tensors = []
    for t in range(pred_frames_sel.shape[0]):
        frame_tensor = pred_frames_sel[t].detach().cpu()
        save_image(frame_tensor, f"{curr_save_dir}/waypoint_gen-t{t}.png")
        pred_frames_tensors.append(frame_tensor)

    save_image(goal_image[0].detach().cpu(), f"{curr_save_dir}/goal_image.png")
    save_image(goal_obs[0].detach().cpu(), f"{curr_save_dir}/goal_obs.png")
    save_image(obs_sel[-1].detach().cpu(), f"{curr_save_dir}/final_context.png")

    # Stacked image with everything
    image_list = [goal_image[0], *drawn_images, torch.ones_like(goal_image[0]), torch.ones_like(goal_image[0])]
    image_tensor = torch.stack(image_list, dim=0)
    image_tensor = torch.cat([image_tensor, torch.stack(pred_frames_tensors), goal_obs[0:1].detach().cpu()], dim=0)
    save_image(image_tensor, f"{curr_save_dir}/stacked_images.png", nrow=image_tensor.shape[0] // 2)

    # Teaser: 2 rows — top: [waypoint_annotated, all_actions, goal_obs, pad...], bottom: pred_frames
    n_frames = len(pred_frames_tensors)
    pad = torch.ones_like(goal_image[0])
    top_row = [waypoint_annotated_tensor, all_actions_tensor, goal_obs[0].detach().cpu()]
    while len(top_row) < n_frames:
        top_row.append(pad)
    top_row = top_row[:n_frames]
    teaser_tensor = torch.stack(top_row + pred_frames_tensors, dim=0)
    save_image(teaser_tensor, f"{curr_save_dir}/teaser_{metric_tag}.png", nrow=n_frames)
    save_image(teaser_tensor, f"{log_dir}/{track_idx_name}_{metric_tag}_teaser.png", nrow=n_frames)


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_name = f"plan_and_vis-h{args.horizon}-n{args.num_samples}-t{args.topk}-o{args.opt_steps}-ds{args.peva_diffusion_steps}"
    log_dir = f"{OUTPUT_DIR}/{datetime_str}:{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    policy, policy_diffusion, _, nomad_config = load_policy(
        args.nomad_config, args.nomad_checkpoint, device=device)
    peva_model, _, peva_diffusion, peva_vae, peva_stats, peva_config = load_peva(
        args.peva_config, args.peva_checkpoint, device=device,
        inference_context_size=args.peva_context_size,
        diffusion_steps=args.peva_diffusion_steps)

    policy_pred_horizon = nomad_config["len_traj_pred"]
    policy_action_dim = nomad_config["input_dims"]
    image_size = nomad_config["image_size"][0]
    policy_context_size = nomad_config["context_size"] + 1
    peva_context_size = peva_config["context_size"]
    peva_latent_size = image_size // 8

    # Build CEM planner; reuse DreamSIM from objective_fn to avoid double-loading
    objective_fn = ObjectiveDreamSIM(pred_horizon=policy_pred_horizon, device=device)
    dreamsim_model = objective_fn.model
    dreamsim_preprocess = objective_fn.preprocess

    wm_wrapper = WaypointWM(
        peva_model, peva_diffusion, peva_vae, peva_stats,
        policy, policy_diffusion,
        image_size, peva_context_size, policy_context_size,
        policy_pred_horizon, policy_action_dim)
    evaluator = EvaluatorWaypoint(
        peva_model, peva_diffusion, peva_vae, peva_stats,
        policy, policy_diffusion,
        image_size, peva_context_size, policy_context_size,
        policy_pred_horizon, policy_action_dim,
        num_eval_samples=args.num_eval_samples)
    cem_planner = CEMPlanner(
        horizon=args.horizon, topk=args.topk,
        num_samples=args.num_samples, var_scale=args.var_scale,
        opt_steps=args.opt_steps, eval_every=args.eval_every,
        wm=wm_wrapper, action_dim=8,
        objective_fn=objective_fn, preprocessor=Preprocessor(),
        evaluator=evaluator, wandb_run=None,
        log_dir=log_dir)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = get_nymeria_dataset(
        nomad_config,
        context_size=max(args.peva_context_size - 1, nomad_config["context_size"]),
        goal_timestep_offset=args.goal_timestep_offset)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=args.shuffle, seed=seed)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1)

    prev_track_name = None
    for idx, batch in enumerate(dataloader):
        if idx > args.num_samples_to_vis:
            break

        obs_images = batch["obs_images"]            # 1, ctx, 3, H, W
        goal_image = batch["goal_image"]            # 1, 3, H, W
        context_poses = batch["context_poses"]      # 1, ctx, 48
        deltas = batch["deltas"]                    # 1, T, 48
        first_pose = batch["first_pose"]            # 1, 1, 48
        xsens_offsets = batch["xsens_offsets"]      # 1, 15, 3
        goal_obs = batch["goal_obs"]                # 1, 3, H, W
        goal_image_coords = batch["goal_image_coords"]  # 1, 23, 2

        dataset_index = batch["dataset_index"].item()
        track_name = batch["dataset_track"][0]
        track_index = batch["dataset_track_index"].item()
        track_idx_name = f"{track_name}-{track_index}"

        skel = XsensSkeleton(xsens_offsets[0])
        gt_actions = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts)
        xyz_dist_matrix, _, _, _ = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        visible_plus_head = (goal_image_coords != -1).all(dim=-1)[:, :XSensConstants.upper_body_num_parts]
        visible_plus_head[:, XSensConstants.part_names.index("Head")] = True
        init_visible_plus_head = xyz_dist_matrix[:, XSensConstants.leaf_indices] * visible_plus_head[:, XSensConstants.leaf_indices]
        init_visible_plus_head = (init_visible_plus_head.sum() / visible_plus_head.sum()).item()

        # Visibility filter
        if not args.keep_nonvisible_goal:
            visible = False
            for part in ["Pelvis", "Head", "R_Hand", "L_Hand"]:
                index = XSensConstants.part_names.index(part)
                if all(goal_image_coords[0, index] != -1):
                    visible = True
                    break
            if not visible:
                print(f"Skipping {track_idx_name}: no visible key parts")
                continue

        if init_visible_plus_head < args.min_dist_threshold:
            print(f"Skipping {track_idx_name}: init distance {init_visible_plus_head:.3f} < threshold")
            continue

        print("=" * 60)
        print(f"Planning + visualizing {track_idx_name}")

        curr_save_dir = os.path.join(log_dir, track_idx_name)
        os.makedirs(curr_save_dir, exist_ok=True)

        # Save context + goal overview
        save_img = torch.cat(
            [obs_images, goal_obs[None], torch.zeros_like(obs_images[:, :-2]), goal_image[None]], dim=1)[0]
        save_image(save_img, f"{curr_save_dir}/context_and_goal.png", nrow=obs_images.shape[1])

        # ── 1. CEM PLANNING ──────────────────────────────────────────────────
        action_init = torch.ones(1, args.horizon, 8) * 0.5
        obs_0 = {"images": obs_images, "goal_image": goal_image, "context_poses": context_poses}
        obs_g = {
            "images": goal_obs, "deltas": deltas, "first_pose": first_pose,
            "xsens_offsets": xsens_offsets, "goal_image_coords": goal_image_coords,
        }
        mu = cem_planner.plan(obs_0, obs_g, track_idx_name, actions=action_init)
        # mu: (1, H, 8) — optimized waypoints in [0, 1] normalized space

        # ── 2. FINAL STOCHASTIC ROLLOUT with optimized waypoints ─────────────
        B = args.num_batch_repeats
        waypoints_px = (mu * image_size).repeat(B, 1, 1)  # (B, H, 8) pixel coords
        obs_images_rep = obs_images.to(device).repeat(B, 1, 1, 1, 1)
        goal_obs_rep = goal_obs.to(device).repeat(B, 1, 1, 1)
        first_pose_rep = first_pose.to(device).repeat(B, 1, 1)
        deltas_rep = deltas.to(device).repeat(B, 1, 1)
        policy_context_poses = context_poses.to(device).repeat(B, 1, 1)[:, -policy_context_size:]

        with torch.no_grad():
            pred_frames, pred_delta, _ = waypoint_sample(
                policy, policy_diffusion,
                peva_model, peva_diffusion, peva_vae, peva_stats,
                waypoints_px, policy_context_poses,
                obs_images_rep, goal_obs_rep,
                policy_pred_horizon, policy_action_dim,
                image_size, policy_context_size, peva_context_size, peva_latent_size,
                device, skip_last_peva=False)

        pred_frames = pred_frames.flatten(0, 1)   # B, T, 3, H, W
        pred_delta = pred_delta.flatten(0, 1)      # B, T, 48

        # ── 3. COMPUTE BOTH METRICS FOR ALL SAMPLES ───────────────────────────
        pred_actions = get_action_smpl_torch(first_pose_rep, pred_delta, XSensConstants.upper_body_num_parts)
        gt_actions_rep = get_action_smpl_torch(first_pose_rep, deltas_rep, XSensConstants.upper_body_num_parts)
        _, _, leaf_xyz, _ = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions_rep[:, -1], skel)

        goal_obs_pil = transforms.ToPILImage()(goal_obs[0].detach().cpu().clamp(0, 1))
        goal_obs_ds = dreamsim_preprocess(goal_obs_pil).to(device)
        perceptual_dists = torch.zeros(B, device=device)
        with torch.no_grad():
            for i in range(B):
                pred_last_pil = transforms.ToPILImage()(pred_frames[i, -1].detach().cpu().clamp(0, 1))
                perceptual_dists[i] = dreamsim_model(
                    dreamsim_preprocess(pred_last_pil).to(device), goal_obs_ds).squeeze()

        # ── 4. AGGRESSIVE PRUNING ─────────────────────────────────────────────
        # Normalize each metric to [0, 1] and sum for combined score
        leaf_xyz_norm = (leaf_xyz - leaf_xyz.min()) / (leaf_xyz.max() - leaf_xyz.min() + 1e-8)
        dsim_norm = (perceptual_dists - perceptual_dists.min()) / (perceptual_dists.max() - perceptual_dists.min() + 1e-8)
        combined_score = leaf_xyz_norm + dsim_norm

        best_joint_idx = leaf_xyz.argmin().item()
        best_dsim_idx = perceptual_dists.argmin().item()
        best_combined_idx = combined_score.argmin().item()

        # Log all-sample metrics
        with open(os.path.join(curr_save_dir, "metrics.txt"), "w") as f:
            f.write(f"{'sample':>8}  {'leaf_xyz':>10}  {'dsim':>10}  {'combined':>10}\n")
            f.write("-" * 44 + "\n")
            for i in range(B):
                marker = ""
                if i == best_joint_idx: marker += " <best_joint"
                if i == best_dsim_idx: marker += " <best_dsim"
                if i == best_combined_idx: marker += " <best_combined"
                f.write(f"{i:>8}  {leaf_xyz[i].item():>10.4f}  {perceptual_dists[i].item():>10.4f}  {combined_score[i].item():>10.4f}{marker}\n")
            f.write(f"\nbest_joint:    sample {best_joint_idx}  leaf_xyz={leaf_xyz[best_joint_idx].item():.4f}\n")
            f.write(f"best_dsim:     sample {best_dsim_idx}  dsim={perceptual_dists[best_dsim_idx].item():.4f}\n")
            f.write(f"best_combined: sample {best_combined_idx}  leaf_xyz={leaf_xyz[best_combined_idx].item():.4f}  dsim={perceptual_dists[best_combined_idx].item():.4f}\n")

        print(f"  best_joint={best_joint_idx} (lxyz={leaf_xyz[best_joint_idx].item():.3f}), "
              f"best_dsim={best_dsim_idx} (dsim={perceptual_dists[best_dsim_idx].item():.3f}), "
              f"best_combined={best_combined_idx}")

        # ── 5. LOAD NYMERIA CAMERA DATA (for pose → image coordinate projection) ─
        if not os.path.exists(os.path.join(DATA_SAVE_DIR, track_name)):
            os.makedirs(os.path.join(DATA_SAVE_DIR, track_name))
            print(f"Downloading episode {track_name}")
            download_episode(DATA_JSON, DATA_SAVE_DIR, track_name)

        if track_name != prev_track_name:
            nymeria_dp = NymeriaDataProvider(
                sequence_rootdir=Path(os.path.join(DATA_SAVE_DIR, track_name)),
                load_wrist=False, load_observer=False)
            cam_model = load_camera_model(DATA_SAVE_DIR, track_name)
        prev_track_name = track_name
        T_C_Pelvis = get_T_C_pelvis(nymeria_dp, track_index)

        # ── 6. VISUALIZE BEST COMBINED (primary output) ───────────────────────
        metric_tag = f"lxyz{leaf_xyz[best_combined_idx].item():.3f}_dsim{perceptual_dists[best_combined_idx].item():.3f}"
        visualize_best_sample(
            sel_idx=best_combined_idx,
            pred_actions=pred_actions,
            pred_frames=pred_frames,
            obs_images_rep=obs_images_rep,
            goal_image=goal_image,
            goal_obs=goal_obs,
            goal_image_coords=goal_image_coords,
            xsens_offsets=xsens_offsets[0],
            cam_model=cam_model,
            T_C_Pelvis=T_C_Pelvis,
            curr_save_dir=curr_save_dir,
            metric_tag=metric_tag,
            log_dir=log_dir,
            track_idx_name=track_idx_name,
        )

        # Also save the best-by-joint and best-by-dsim teasers when they differ
        if best_joint_idx != best_combined_idx:
            sub_dir = os.path.join(curr_save_dir, "best_joint")
            os.makedirs(sub_dir, exist_ok=True)
            tag = f"lxyz{leaf_xyz[best_joint_idx].item():.3f}_dsim{perceptual_dists[best_joint_idx].item():.3f}"
            visualize_best_sample(
                sel_idx=best_joint_idx,
                pred_actions=pred_actions,
                pred_frames=pred_frames,
                obs_images_rep=obs_images_rep,
                goal_image=goal_image,
                goal_obs=goal_obs,
                goal_image_coords=goal_image_coords,
                xsens_offsets=xsens_offsets[0],
                cam_model=cam_model,
                T_C_Pelvis=T_C_Pelvis,
                curr_save_dir=sub_dir,
                metric_tag=tag,
                log_dir=log_dir,
                track_idx_name=f"{track_idx_name}_best_joint",
            )

        if best_dsim_idx != best_combined_idx and best_dsim_idx != best_joint_idx:
            sub_dir = os.path.join(curr_save_dir, "best_dsim")
            os.makedirs(sub_dir, exist_ok=True)
            tag = f"lxyz{leaf_xyz[best_dsim_idx].item():.3f}_dsim{perceptual_dists[best_dsim_idx].item():.3f}"
            visualize_best_sample(
                sel_idx=best_dsim_idx,
                pred_actions=pred_actions,
                pred_frames=pred_frames,
                obs_images_rep=obs_images_rep,
                goal_image=goal_image,
                goal_obs=goal_obs,
                goal_image_coords=goal_image_coords,
                xsens_offsets=xsens_offsets[0],
                cam_model=cam_model,
                T_C_Pelvis=T_C_Pelvis,
                curr_save_dir=sub_dir,
                metric_tag=tag,
                log_dir=log_dir,
                track_idx_name=f"{track_idx_name}_best_dsim",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--goal_timestep_offset", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--num_samples_to_vis", type=int, default=32)

    # CEM parameters
    parser.add_argument("-n", "--num_samples", type=int, default=32, help="CEM samples per step")
    parser.add_argument("-t", "--topk", type=int, default=4, help="CEM top-k kept per step")
    parser.add_argument("-v", "--var_scale", type=float, default=0.5, help="CEM initial variance")
    parser.add_argument("-o", "--opt_steps", type=int, default=8, help="CEM optimization steps")
    parser.add_argument("-e", "--eval_every", type=int, default=1, help="CEM eval frequency")
    parser.add_argument("-H", "--horizon", type=int, default=1, help="Waypoint horizon (num waypoints)")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=1, help="CEM eval samples")

    # Final rollout stochastic samples for pruning
    parser.add_argument("--num_batch_repeats", type=int, default=8,
                        help="Stochastic rollout samples from optimized waypoints for final pruning")

    parser.add_argument("--peva_config", type=str,
        default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str,
        default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15)
    parser.add_argument("--peva_diffusion_steps", type=int, default=250)

    parser.add_argument("--nomad_model", type=str, default="draw_mask", choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    else:
        assert args.nomad_config is not None and args.nomad_checkpoint is not None

    main(args)
