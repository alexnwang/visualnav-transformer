"""
Cherry-pick best CEM planning results for visualization.

For each task:
  1. Run CEM planning with high-fidelity settings to find the best waypoints (mu).
  2. Score mu by how well the waypoints match ground-truth (MJE after CEM).
  3. Sample N stochastic policy rollouts with the fixed mu.
  4. Pick the rollout with the lowest MJE.
  5. Save the visualization for that best rollout.

After all tasks, print a summary sorted by MJE so you can cherry-pick the best tasks.
"""
import argparse
from datetime import datetime
import os
import torch
import copy
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw
from einops import repeat
from torch.utils.data import DistributedSampler, DataLoader, Subset

from peva.diffusion import create_diffusion
from plan_cem import build_waypoint_cem, build_peva_cem, MODEL_DIRECTORY
from planning.cem import CEMPlanner, move_to_device
from planning.nymeria_dataset import NymeriaPlanningDataset, build_planning_split, _LEAF_IDX
from planning.wrappers import (
    Preprocessor, build_skeleton_top_seq, save_mu_step_results,
    ObjectiveDreamSIM,
)
from planning.utils import _compute_part_distance_matrices
from planning.vis_utils import draw_image_coords, disable_logging
from planning.plotting_fns import save_action_obs_sequence_viz
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch


def multi_rollout_topk(wm, state_0, state_g, mu, num_rollouts, device, algo,
                       top_k_save=3, topk_wm=8):
    """
    For waypoint algos the WM has a cheap policy stage and an expensive PEVA stage,
    so we do a two-stage filter: policy-only for all `num_rollouts`, then PEVA on
    the top `topk_wm`, returning the top `top_k_save` by MJE plus stats.

    For `peva`, mu is a single deterministic action sequence — we roll it out once
    and return a single-element result. `num_rollouts` / `topk_wm` are ignored.
    """
    xsens_offsets = state_g["xsens_offsets"][0]
    skel = XsensSkeleton(xsens_offsets)
    first_pose = state_g["first_pose"]   # 1, 1, 48
    deltas_gt = state_g["deltas"]        # 1, T, 48

    def score_mje(pred_deltas, fp, gt):
        pred_actions = get_action_smpl_torch(fp, pred_deltas, XSensConstants.upper_body_num_parts)
        gt_actions = get_action_smpl_torch(fp, gt, XSensConstants.upper_body_num_parts)
        xyz_dist_matrix, _, leaf_xyz, _ = _compute_part_distance_matrices(
            pred_actions[:, -1], gt_actions[:, -1], skel
        )
        return xyz_dist_matrix.mean(dim=-1), leaf_xyz

    if algo == "peva":
        # PEVA has a single action sequence (mu) — just roll it out once.
        with torch.no_grad():
            wm_state = wm.rollout(state_0=state_0, act=mu)

        all_xyz, leaf_xyz = score_mje(wm_state["deltas"], first_pose, deltas_gt)  # length-1 tensors
        saved_states = [{
            "state": {
                "generated_obs": wm_state["generated_obs"],
                "deltas": wm_state["deltas"],
                "goal_images": wm_state["goal_images"],
            },
            "mje": all_xyz[0].item(),
            "leaf_xyz": leaf_xyz[0].item(),
            "orig_idx": 0,
        }]
    else:
        batched_state_0 = {
            key: repeat(arr, "1 ... -> n ...", n=num_rollouts)
            for key, arr in state_0.items()
        }
        batched_mu = repeat(mu, "1 ... -> n ...", n=num_rollouts)
        first_pose_rep = repeat(first_pose, "1 ... -> n ...", n=num_rollouts)
        deltas_gt_rep = repeat(deltas_gt, "1 ... -> n ...", n=num_rollouts)

        # --- Stage 1: policy-only rollouts (cheap) ---
        with torch.no_grad():
            policy_state = wm.policy_only_rollout(state_0=batched_state_0, act=batched_mu)

        pred_deltas = policy_state["deltas"]  # N, T, 48
        all_xyz, leaf_xyz = score_mje(pred_deltas, first_pose_rep, deltas_gt_rep)

        # --- Stage 2: WM rollout on top-k only (expensive) ---
        k = min(max(topk_wm, top_k_save), num_rollouts)
        topk_indices = torch.topk(all_xyz, k, largest=False).indices  # k best, ascending

        topk_state_0 = {key: arr[topk_indices] for key, arr in batched_state_0.items()}
        topk_deltas = pred_deltas[topk_indices]  # k, T, 48
        topk_goal_images = policy_state["goal_images"][topk_indices] if policy_state["goal_images"] is not None else None

        with torch.no_grad():
            wm_state = wm.wm_only_rollout(topk_state_0, topk_deltas, goal_images=topk_goal_images)

        n_save = min(top_k_save, k)
        saved_states = []
        for i in range(n_save):
            orig_idx = topk_indices[i].item()
            saved_states.append({
                "state": {
                    "generated_obs": wm_state["generated_obs"][i:i+1],
                    "deltas": wm_state["deltas"][i:i+1],
                    "goal_images": wm_state["goal_images"][i:i+1] if wm_state["goal_images"] is not None else None,
                },
                "mje": all_xyz[orig_idx].item(),
                "leaf_xyz": leaf_xyz[orig_idx].item(),
                "orig_idx": orig_idx,
            })

    stats = {
        "all_mje": all_xyz.cpu().tolist(),
        "all_leaf_xyz": leaf_xyz.cpu().tolist(),
        "mean_mje": all_xyz.mean().item(),
        "median_mje": all_xyz.median().item(),
        "std_mje": all_xyz.std().item(),
        "mean_leaf_xyz": leaf_xyz.mean().item(),
        "median_leaf_xyz": leaf_xyz.median().item(),
    }
    return saved_states, stats


def compute_waypoint_err(mu, gt_leaf_coords, image_size_px):
    """Per-joint waypoint error in [0,1] normalized image space (H=1 only).

    A waypoint is considered "present" in the searched mu iff both its (x,y)
    coords lie strictly inside (0,1) — matching the training-time masking
    convention (policy treats out-of-image waypoints as absent).

    Per-joint error:
      - present & GT visible:    L2 distance between mu and GT in [0,1].
      - present & GT invisible:  shortest distance to push mu outside [0,1]
                                 in any axis (penalizes spurious waypoints).
      - absent in mu:            excluded from the mean (policy can infer).

    Returns nan if no waypoints are present in mu.
    """
    mu_wp = mu.reshape(-1, 4, 2)[0]  # (4, 2), [0,1]
    mu_present = ((mu_wp > 0) & (mu_wp < 1)).all(dim=-1)  # (4,)
    if not mu_present.any():
        return float("nan")

    device = mu_wp.device
    gt_visible = (gt_leaf_coords != -1).all(dim=-1).to(device)         # (4,)
    gt_norm = gt_leaf_coords.float().to(device) / image_size_px         # (4, 2)

    l2 = (mu_wp - gt_norm).norm(dim=-1)                                 # (4,)
    border_dist = torch.minimum(
        torch.minimum(mu_wp[:, 0], 1 - mu_wp[:, 0]),
        torch.minimum(mu_wp[:, 1], 1 - mu_wp[:, 1]),
    )                                                                   # (4,)
    err_per_joint = torch.where(gt_visible, l2, border_dist)            # (4,)
    return err_per_joint[mu_present].mean().item()


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    gpu = 0
    torch.cuda.set_device(gpu)
    device = "cuda"

    algo = args.algo
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = (
        f"viz_{algo}_cem-h{args.horizon}-n{args.num_samples}-t{args.topk}"
        f"-v{args.var_scale}-o{args.opt_steps}-R{args.num_vis_rollouts}"
        f"-ds{args.peva_diffusion_steps}-visds{args.peva_vis_diffusion_steps}"
        f"-dist{args.min_dist_cat}-{args.max_dist_cat}"
    )

    disable_logging()
    camera_data_cache = {}
    log_dir = f"logs/cem_viz/{datetime_str}:{run_name}:ws{args.world_size}-r{args.rank}"
    os.makedirs(log_dir, exist_ok=True)

    # Build CEM planner (no wandb -- this is offline viz)
    if algo == "waypoint":
        action_init = torch.ones(1, args.horizon, 8) * 0.5
        cem_planner, nomad_config, peva_config = build_waypoint_cem(args, None, log_dir, device)
    elif algo == "waypoint_point3d":
        action_init = torch.cat([
            torch.ones(1, args.horizon, 4, 2) * 0.5,
            torch.ones(1, args.horizon, 4, 1) * 0.5
        ], dim=-1).flatten(2, 3)
        cem_planner, nomad_config, peva_config = build_waypoint_cem(args, None, log_dir, device)
    elif algo == "peva":
        action_init = None
        cem_planner, nomad_config, peva_config = build_peva_cem(args, None, log_dir, device)

    # Prepare dataset
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
        gaussian_normalization_stats_path=data_config.get("gaussian_normalization_stats_path", None),
    )
    if args.target_tracks:
        lookup = {f"{t['track']}-s{t['curr_time']}-g{t['goal_time']}": i
                  for i, t in enumerate(dataset.tasks)}
        indices, missing = [], []
        for k in args.target_tracks:
            if k in lookup:
                indices.append(lookup[k])
            else:
                missing.append(k)
        if missing:
            print(f"WARNING: target tracks not found in split: {missing}")
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False, num_workers=0)
    else:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle, seed=seed)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    # ---------- Main loop ----------
    task_results = []

    for idx, batch in enumerate(dataloader):
        if args.num_samples_to_plan > 0 and idx >= args.num_samples_to_plan:
            break

        obs_images = batch["obs_images"]
        goal_image = batch["goal_image"]
        context_poses = batch["context_poses"]
        deltas = batch["deltas"]
        first_pose = batch["first_pose"]
        goal_pose = batch["goal_pose"]
        xsens_offsets = batch["xsens_offsets"]
        goal_obs = batch["goal_obs"]
        goal_image_coords = batch["goal_image_coords"]
        gt_frames = batch["gt_frames"]
        gt_image_coords_seq = batch["gt_image_coords_seq"]

        track = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index = batch["goal_index"].item()
        track_idx_name = f"{track}-s{start_index}-g{goal_index}"

        print("=" * 60)
        print(f"[{idx+1}/{args.num_samples_to_plan}] Planning {track_idx_name}")

        task_dir = f"{log_dir}/{track_idx_name}"
        os.makedirs(task_dir, exist_ok=True)

        # Camera params
        if args.camera_data_folder:
            if track not in camera_data_cache:
                cam_path = os.path.join(args.camera_data_folder, track, "camera_data.pt")
                camera_data_cache[track] = torch.load(cam_path, weights_only=False)
            cam_data = camera_data_cache[track]
            T_mat = cam_data["T_C_pelvis"][start_index]
            fisheye_params = cam_data["fisheye_params"]
            R_C_pelvis = T_mat[:3, :3]
            t_C_pelvis = T_mat[:3, 3]
        else:
            fisheye_params, R_C_pelvis, t_C_pelvis = None, None, None

        obs_0 = {"images": obs_images, "goal_image": goal_image, "context_poses": context_poses}
        obs_g = {"images": goal_obs, "deltas": deltas, "first_pose": first_pose,
                 "xsens_offsets": xsens_offsets, "goal_image_coords": goal_image_coords}

        # ------ Step 1: Run CEM planning ------
        mu = cem_planner.plan(
            obs_0, obs_g, track_idx_name, actions=action_init,
            fisheye_params=fisheye_params, R_C_pelvis=R_C_pelvis, t_C_pelvis=t_C_pelvis,
            render_skin=True,
        )

        # ------ Step 1b: Pick the best mu across all CEM iterations ------
        trans_obs_0 = move_to_device(Preprocessor().transform_obs(obs_0), device)
        trans_obs_g = move_to_device(Preprocessor().transform_obs(obs_g), device)

        cem_eval = cem_planner.accum_eval_metric_dicts.get(track_idx_name, {})
        all_xyz_per_step = cem_eval.get("all_xyz", [])

        # Load mu_history from the results file we just saved
        results_path = f"{log_dir}/{track_idx_name}/results.pth"
        results = torch.load(results_path, weights_only=False)
        mu_history = results["mu_history"]  # (opt_steps, 1, H, action_dim)

        gt_leaf_coords = goal_image_coords[0, _LEAF_IDX]  # (4, 2), pixel space
        image_size_px = nomad_config["image_size"][0]

        # Per-step waypoint error
        if mu_history is not None:
            wp_err_per_step = [
                compute_waypoint_err(mu_history[t], gt_leaf_coords, image_size_px)
                for t in range(len(mu_history))
            ]
        else:
            wp_err_per_step = []

        # Pick two bests: one by per-step MJE, one by per-step wp_err
        if mu_history is not None and len(all_xyz_per_step) > 0:
            best_step_mje_idx = int(np.argmin(all_xyz_per_step))
            mu_best_mje = mu_history[best_step_mje_idx].to(device)
        else:
            best_step_mje_idx = len(mu_history) - 1 if mu_history is not None else 0
            mu_best_mje = mu_history[best_step_mje_idx].to(device) if mu_history is not None else mu

        if len(wp_err_per_step) > 0 and not all(np.isnan(wp_err_per_step)):
            wp_arr = np.array(wp_err_per_step, dtype=float)
            wp_arr_filled = np.where(np.isnan(wp_arr), np.inf, wp_arr)
            best_step_wp_idx = int(np.argmin(wp_arr_filled))
            mu_best_wp = mu_history[best_step_wp_idx].to(device)
        else:
            best_step_wp_idx = best_step_mje_idx
            mu_best_wp = mu_best_mje

        wp_err_at_best_mje = wp_err_per_step[best_step_mje_idx] if wp_err_per_step else float("nan")
        wp_err_at_best_wp = wp_err_per_step[best_step_wp_idx] if wp_err_per_step else float("nan")
        mje_at_best_mje = all_xyz_per_step[best_step_mje_idx] if all_xyz_per_step else float("nan")
        mje_at_best_wp = all_xyz_per_step[best_step_wp_idx] if all_xyz_per_step else float("nan")

        print(f"  Best by MJE: step {best_step_mje_idx} "
              f"(mje={mje_at_best_mje:.4f}, wp_err={wp_err_at_best_mje:.4f})")
        print(f"  Best by WP : step {best_step_wp_idx} "
              f"(mje={mje_at_best_wp:.4f}, wp_err={wp_err_at_best_wp:.4f})")
        print(f"  Final step : mje={all_xyz_per_step[-1] if all_xyz_per_step else float('nan'):.4f}, "
              f"wp_err={wp_err_per_step[-1] if wp_err_per_step else float('nan'):.4f}")

        # ------ Step 2: Multi-rollout for each best mu, save top-K ------
        if args.peva_vis_diffusion_steps != args.peva_diffusion_steps:
            hifi_diffusion = create_diffusion(str(args.peva_vis_diffusion_steps))
            cem_planner.wm.peva_diffusion = hifi_diffusion

        best_variants = [("best_mje", mu_best_mje, best_step_mje_idx)]
        if best_step_wp_idx != best_step_mje_idx:
            best_variants.append(("best_wp", mu_best_wp, best_step_wp_idx))

        variant_results = {}
        for vname, vmu, vstep in best_variants:
            saved_states, stats = multi_rollout_topk(
                cem_planner.wm, trans_obs_0, trans_obs_g, vmu,
                num_rollouts=args.num_vis_rollouts, device=device,
                algo=algo,
                top_k_save=args.top_k_save,
            )
            print(f"  [{vname}] top1 mje={saved_states[0]['mje']:.4f} leaf={saved_states[0]['leaf_xyz']:.4f}  "
                  f"mean mje={stats['mean_mje']:.4f} ± {stats['std_mje']:.4f}  "
                  f"median={stats['median_mje']:.4f}")
            variant_results[vname] = {
                "mu_step": vstep,
                "saved_states": saved_states,
                "stats": stats,
            }

        if args.peva_vis_diffusion_steps != args.peva_diffusion_steps:
            cem_planner.wm.peva_diffusion = create_diffusion(str(args.peva_diffusion_steps))

        # ------ Step 3: Compute fraction solved (from best_mje top1) and rename folder ------
        task_dicts = cem_planner.accum_task_dicts.get(track_idx_name, {})
        init_mje = task_dicts.get("task", {}).get("all_xyz_init", None)
        best_mje_top1 = variant_results["best_mje"]["saved_states"][0]["mje"]
        mean_mje_best = variant_results["best_mje"]["stats"]["mean_mje"]
        if init_mje is not None and init_mje > 0:
            frac_solved = max(0.0, 1.0 - best_mje_top1 / init_mje)
            frac_solved_mean = max(0.0, 1.0 - mean_mje_best / init_mje)
        else:
            frac_solved = 0.0
            frac_solved_mean = 0.0
        print(f"  Fraction solved: {frac_solved:.2%} (top1)  {frac_solved_mean:.2%} (mean)  "
              f"(init_mje={init_mje:.4f})")

        scored_dir = f"{log_dir}/{frac_solved:.3f}_{track_idx_name}"
        os.rename(task_dir, scored_dir)
        task_dir = scored_dir

        # ------ Step 4: Save visualizations for best rollout ------
        # Save GT visualization
        curr_image = obs_images[0, -1]
        n_steps = gt_frames.shape[1]
        gt_skel_imgs = build_skeleton_top_seq(
            curr_image, deltas, first_pose, xsens_offsets[0],
            fisheye_params, R_C_pelvis, t_C_pelvis,
            curr_image.shape[-1], n_steps,
            overlay='skeleton',
        )
        save_action_obs_sequence_viz(
            save_path=f"{task_dir}/gt_action_obs_seq.png",
            goal_image=goal_image[0],
            curr_obs=curr_image,
            goal_obs=goal_obs[0],
            top_seq=gt_skel_imgs,
            bot_seq=gt_frames[0],
        )
        if fisheye_params is not None:
            gt_skel_imgs_skin = build_skeleton_top_seq(
                curr_image, deltas, first_pose, xsens_offsets[0],
                fisheye_params, R_C_pelvis, t_C_pelvis,
                curr_image.shape[-1], n_steps,
                overlay='both', smpl_alpha=0.9,
            )
            save_action_obs_sequence_viz(
                save_path=f"{task_dir}/gt_action_obs_seq_skin.png",
                goal_image=goal_image[0],
                curr_obs=curr_image,
                goal_obs=goal_obs[0],
                top_seq=gt_skel_imgs_skin,
                bot_seq=gt_frames[0],
            )

        # Save top-K rollout visualizations for each variant (with skin)
        per_variant_save = {}
        for vname, vdata in variant_results.items():
            for ri, sstate in enumerate(vdata["saved_states"]):
                save_mu_step_results(
                    sstate["state"], trans_obs_0, trans_obs_g,
                    fisheye_params, R_C_pelvis, t_C_pelvis,
                    save_path=f"{task_dir}/{vname}/rollout_{ri}_mje{sstate['mje']:.3f}",
                    render_skin=(fisheye_params is not None),
                )
            per_variant_save[vname] = {
                "mu_step": vdata["mu_step"],
                "wp_err": wp_err_per_step[vdata["mu_step"]] if wp_err_per_step else float("nan"),
                "step_mje": all_xyz_per_step[vdata["mu_step"]] if all_xyz_per_step else float("nan"),
                "rollout_top1_mje": vdata["saved_states"][0]["mje"],
                "rollout_top1_leaf_xyz": vdata["saved_states"][0]["leaf_xyz"],
                "rollout_topk_mjes": [s["mje"] for s in vdata["saved_states"]],
                "rollout_topk_leaf_xyz": [s["leaf_xyz"] for s in vdata["saved_states"]],
                "rollout_mean_mje": vdata["stats"]["mean_mje"],
                "rollout_median_mje": vdata["stats"]["median_mje"],
                "rollout_std_mje": vdata["stats"]["std_mje"],
                "rollout_mean_leaf_xyz": vdata["stats"]["mean_leaf_xyz"],
                "rollout_stats": vdata["stats"],
            }

        # Save context and goal frames
        save_image(obs_images[0], f"{task_dir}/context_frames.png", nrow=obs_images.shape[1])
        save_image(goal_obs[0], f"{task_dir}/goal_obs.png")

        # Save numerical results
        torch.save({
            "track": track,
            "start_index": start_index,
            "goal_index": goal_index,
            "mu_best_mje": mu_best_mje.cpu(),
            "mu_best_wp": mu_best_wp.cpu(),
            "best_step_mje_idx": best_step_mje_idx,
            "best_step_wp_idx": best_step_wp_idx,
            "all_xyz_per_step": all_xyz_per_step,
            "wp_err_per_step": wp_err_per_step,
            "init_mje": init_mje,
            "frac_solved": frac_solved,
            "frac_solved_mean": frac_solved_mean,
            "variants": per_variant_save,
        }, f"{task_dir}/viz_results.pth")

        bm = per_variant_save["best_mje"]
        bw = per_variant_save.get("best_wp", bm)
        task_results.append({
            "task_name": track_idx_name,
            "frac_solved": frac_solved,
            "frac_solved_mean": frac_solved_mean,
            "init_mje": init_mje,
            "best_mje_step": bm["mu_step"],
            "best_mje_top1": bm["rollout_top1_mje"],
            "best_mje_mean": bm["rollout_mean_mje"],
            "best_mje_wp_err": bm["wp_err"],
            "best_wp_step": bw["mu_step"],
            "best_wp_top1": bw["rollout_top1_mje"],
            "best_wp_mean": bw["rollout_mean_mje"],
            "best_wp_wp_err": bw["wp_err"],
        })

    # ------ Summary ------
    print("\n" + "=" * 60)
    print("TASK RANKING (sorted by frac_solved descending)")
    print("=" * 60)
    print(f"  {'rk':<3} {'frac':>6} {'fracμ':>6} {'init':>7} | "
          f"{'mje_top1':>8} {'mje_mean':>8} {'mje_wp':>7} | "
          f"{'wp_top1':>7} {'wp_mean':>7} {'wp_wp':>6} | task")
    task_results.sort(key=lambda x: x["frac_solved"], reverse=True)
    for rank, tr in enumerate(task_results, 1):
        print(f"  {rank:<3} {tr['frac_solved']:>6.3f} {tr['frac_solved_mean']:>6.3f} "
              f"{tr['init_mje']:>7.4f} | "
              f"{tr['best_mje_top1']:>8.4f} {tr['best_mje_mean']:>8.4f} {tr['best_mje_wp_err']:>7.4f} | "
              f"{tr['best_wp_top1']:>7.4f} {tr['best_wp_mean']:>7.4f} {tr['best_wp_wp_err']:>6.4f} | "
              f"{tr['task_name']}")

    torch.save(task_results, f"{log_dir}/task_ranking.pth")
    print(f"\nResults saved to {log_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cherry-pick best CEM results for visualization")

    parser.add_argument("-a", "--algo", type=str, choices=["peva", "waypoint", "waypoint_point3d"], default="waypoint")
    parser.add_argument("--use_leafxyz_as_cost", action="store_true")
    parser.add_argument("--shuffle", action="store_true")

    # CEM parameters
    parser.add_argument("-n", "--num_samples", type=int, default=128, help="CEM samples per iteration")
    parser.add_argument("-t", "--topk", type=int, default=8, help="CEM top-k elites")
    parser.add_argument("-v", "--var_scale", type=float, default=0.5, help="CEM initial variance")
    parser.add_argument("-o", "--opt_steps", type=int, default=16, help="CEM optimization steps")
    parser.add_argument("-H", "--horizon", type=int, default=1, help="Waypoint horizon")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=1, help="Eval samples per CEM step")

    # Visualization-specific
    parser.add_argument("-R", "--num_vis_rollouts", type=int, default=32,
                        help="Number of stochastic rollouts per task to cherry-pick from")
    parser.add_argument("--top_k_save", type=int, default=3,
                        help="Number of top rollouts (per best-mu variant) to save with visualizations")

    # Dataset
    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--min_dist_cat", type=int, default=8)
    parser.add_argument("--max_dist_cat", type=int, default=8)
    parser.add_argument("--curr_time_stride", type=int, default=1)
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--num_samples_to_plan", type=int, default=64,
                        help="Number of tasks to process")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--no_skin", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--camera_data_folder", type=str,
                        default="/home/anw2067/scratch/nymeria_visibility_matrix")
    parser.add_argument("--target_tracks", type=str, nargs="+", default=None,
                        help="Specific tasks to run, formatted as '{track}-s{curr_time}-g{goal_time}'. "
                             "When set, --rank/--world_size/--shuffle are ignored.")

    # Models
    parser.add_argument("--peva_config", type=str,
                        default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str,
                        default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15)
    parser.add_argument("--peva_diffusion_steps", type=int, default=64,
                        help="Diffusion steps during CEM search (fast)")
    parser.add_argument("--peva_vis_diffusion_steps", type=int, default=250,
                        help="Diffusion steps for final visualization rollouts (high fidelity)")

    parser.add_argument("--nomad_model", type=str, default="draw", choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)

    # Unused but needed by build_*_cem
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()

    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    else:
        assert args.nomad_config is not None and args.nomad_checkpoint is not None

    main(args)
