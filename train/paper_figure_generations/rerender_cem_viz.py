"""
Re-render visualizations for an existing CEM viz log directory.

Reads `mu_history` from each task's saved `results.pth` and re-runs only the
WM rollouts + visualization functions (no CEM optimization). Useful when
rendering functions in `planning/wrappers.py`, `planning/vis_utils.py`, or
`planning/plotting_fns.py` have changed but you don't want to redo planning.

For each task:
  1. Reload the dataset batch via target_tracks lookup.
  2. Reload `mu_history` (per-step mean) and best_step indices from
     `results.pth` / `viz_results.pth`.
  3. Re-render GT + static frames (always cheap).
  4. For each step in mu_history (`--steps all`), re-roll WM with that mu and
     save an `action_obs_seq.png` under `step{NNN}/`.
  5. For best_mje and best_wp variants, re-roll WM with top-k saving and
     write under `best_mje/` and `best_wp/`.

Output: logs/cem_viz/{datetime}:Rerender_from_{source_basename}/{frac}_{task}/...
"""
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("/home/anw2067/visualnav-transformer/train")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from peva.diffusion import create_diffusion
from plan_cem import build_waypoint_cem, build_peva_cem, MODEL_DIRECTORY
from plan_cem_viz import multi_rollout_topk, compute_waypoint_err
from planning.cem import move_to_device
from planning.nymeria_dataset import NymeriaPlanningDataset, build_planning_split, _LEAF_IDX
from planning.utils import draw_waypoints
from planning.wrappers import (
    Preprocessor, build_skeleton_top_seq, save_mu_step_results,
)
from planning.plotting_fns import save_action_obs_sequence_viz
from planning.vis_utils import disable_logging
from vint_train.data.misc import XSensConstants


RUN_NAME_RE = re.compile(
    r"viz_(?P<algo>waypoint_point3d|waypoint|peva)_cem"
    r"-h(?P<horizon>\d+)"
    r"-n(?P<num_samples>\d+)"
    r"-t(?P<topk>\d+)"
    r"-v(?P<var_scale>[\d.]+)"
    r"-o(?P<opt_steps>\d+)"
    r"-R(?P<num_vis_rollouts>\d+)"
    r"-ds(?P<peva_diffusion_steps>\d+)"
    r"-visds(?P<peva_vis_diffusion_steps>\d+)"
    r"-dist(?P<min_dist_cat>\d+)-(?P<max_dist_cat>\d+)"
)


def parse_run_name(source_log_dir):
    """Parse defaults out of the source dir's run name. Returns dict (may be empty)."""
    base = os.path.basename(source_log_dir.rstrip("/"))
    # base format: "{datetime}:{run_name}[:ws{n}-r{r}]"
    parts = base.split(":")
    if len(parts) < 2:
        return {}
    run_name = parts[1]
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "algo": d["algo"],
        "horizon": int(d["horizon"]),
        "num_samples": int(d["num_samples"]),
        "topk": int(d["topk"]),
        "var_scale": float(d["var_scale"]),
        "opt_steps": int(d["opt_steps"]),
        "num_vis_rollouts": int(d["num_vis_rollouts"]),
        "peva_diffusion_steps": int(d["peva_diffusion_steps"]),
        "peva_vis_diffusion_steps": int(d["peva_vis_diffusion_steps"]),
        "min_dist_cat": int(d["min_dist_cat"]),
        "max_dist_cat": int(d["max_dist_cat"]),
    }


def list_task_subdirs(source_log_dir):
    """Return list of (task_subdir_name, track_idx_name) for tasks present in source."""
    out = []
    for name in sorted(os.listdir(source_log_dir)):
        full = os.path.join(source_log_dir, name)
        if not os.path.isdir(full):
            continue
        # Task subdirs are either "{frac}_{track-sX-gY}" or just "{track-sX-gY}"
        if not os.path.exists(os.path.join(full, "results.pth")):
            continue
        m = re.match(r"^([\d.]+)_(.+-s\d+-g\d+)$", name)
        if m:
            track_idx_name = m.group(2)
        else:
            m2 = re.match(r"^(.+-s\d+-g\d+)$", name)
            if not m2:
                continue
            track_idx_name = m2.group(1)
        out.append((name, track_idx_name))
    return out


def render_gt_and_static(task_dir, batch, fisheye_params, R_C_pelvis, t_C_pelvis):
    """Re-render gt_action_obs_seq{,_skin}.png + context_frames.png + goal_obs.png."""
    obs_images = batch["obs_images"]
    goal_image = batch["goal_image"]
    deltas = batch["deltas"]
    first_pose = batch["first_pose"]
    xsens_offsets = batch["xsens_offsets"]
    goal_obs = batch["goal_obs"]
    gt_frames = batch["gt_frames"]

    curr_image = obs_images[0, -1]
    n_steps = gt_frames.shape[1]

    gt_skel_imgs = build_skeleton_top_seq(
        curr_image, deltas, first_pose, xsens_offsets[0],
        fisheye_params, R_C_pelvis, t_C_pelvis,
        curr_image.shape[-1], n_steps,
        overlay='skeleton', show_text=False,
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
            overlay='both', smpl_alpha=0.9, show_text=False,
        )
        save_action_obs_sequence_viz(
            save_path=f"{task_dir}/gt_action_obs_seq_skin.png",
            goal_image=goal_image[0],
            curr_obs=curr_image,
            goal_obs=goal_obs[0],
            top_seq=gt_skel_imgs_skin,
            bot_seq=gt_frames[0],
        )

    save_image(obs_images[0], f"{task_dir}/context_frames.png", nrow=obs_images.shape[1])
    save_image(goal_obs[0], f"{task_dir}/goal_obs.png")

    # curr_obs + GT goal-pose mesh (no skeleton) + leaf waypoints
    if fisheye_params is not None:
        leaf_wp = batch["goal_image_coords"][:, XSensConstants.leaf_indices]  # 1, 4, 2
        gt_top_mesh_only = build_skeleton_top_seq(
            curr_image, deltas, first_pose, xsens_offsets[0],
            fisheye_params, R_C_pelvis, t_C_pelvis,
            curr_image.shape[-1], n_steps,
            overlay='skin', smpl_alpha=0.9, show_text=False,
        )
        goal_image_wp_mesh = draw_waypoints(gt_top_mesh_only[-1:], leaf_wp)  # 1, 3, H, W
        save_image(goal_image_wp_mesh[0], f"{task_dir}/goal_image_waypoints_mesh.png")


def main(args):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    device = "cuda"

    disable_logging()

    # Output dir
    source_basename = os.path.basename(args.source_log_dir.rstrip("/"))
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = f"logs/cem_viz/{datetime_str}:Rerender_from_{source_basename}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}")

    # Determine which tasks to re-render
    all_tasks = list_task_subdirs(args.source_log_dir)
    if args.tasks:
        wanted = set(args.tasks)
        # Match against either the full subdir name or just the track_idx_name
        filtered = [(sd, tin) for (sd, tin) in all_tasks if sd in wanted or tin in wanted]
        missing = wanted - {sd for sd, _ in filtered} - {tin for _, tin in filtered}
        if missing:
            print(f"WARNING: requested tasks not found: {sorted(missing)}")
        all_tasks = filtered
    if not all_tasks:
        raise RuntimeError(f"No tasks to re-render in {args.source_log_dir}")
    print(f"Re-rendering {len(all_tasks)} tasks")

    # Build CEM planner — we only need .wm; planner is convenient for nomad/peva configs
    algo = args.algo
    if algo in ("waypoint", "waypoint_point3d"):
        cem_planner, nomad_config, _ = build_waypoint_cem(args, None, log_dir, device)
    elif algo == "peva":
        cem_planner, nomad_config, _ = build_peva_cem(args, None, log_dir, device)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # Build dataset
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
    full_dataset = NymeriaPlanningDataset(
        tasks_file=tasks_file,
        data_folder=data_config["data_folder"],
        image_size=nomad_config["image_size"],
        context_size=context_size,
        goal_type=nomad_config.get("goal_type", None),
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        gaussian_normalization_stats_path=data_config.get("gaussian_normalization_stats_path", None),
    )
    lookup = {f"{t['track']}-s{t['curr_time']}-g{t['goal_time']}": i
              for i, t in enumerate(full_dataset.tasks)}
    indices, present_tasks = [], []
    for sd, tin in all_tasks:
        if tin in lookup:
            indices.append(lookup[tin])
            present_tasks.append((sd, tin))
        else:
            print(f"WARNING: {tin} not in dataset split — skipping")
    dataset = Subset(full_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False, num_workers=0)

    camera_data_cache = {}
    task_results = []

    for (source_subdir, track_idx_name), batch in zip(present_tasks, dataloader):
        print("=" * 60)
        print(f"Re-rendering {track_idx_name} (from {source_subdir})")

        source_task_dir = os.path.join(args.source_log_dir, source_subdir)
        results = torch.load(f"{source_task_dir}/results.pth", weights_only=False)
        prior_viz = torch.load(f"{source_task_dir}/viz_results.pth", weights_only=False) \
            if os.path.exists(f"{source_task_dir}/viz_results.pth") else {}

        mu_history = results["mu_history"]  # (opt_steps, 1, H, action_dim)
        if mu_history is None:
            print("  no mu_history; skipping")
            continue

        # Camera params
        track = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
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

        # Build obs dicts (mirrors plan_cem_viz)
        obs_0 = {"images": batch["obs_images"], "goal_image": batch["goal_image"],
                 "context_poses": batch["context_poses"]}
        obs_g = {"images": batch["goal_obs"], "deltas": batch["deltas"],
                 "first_pose": batch["first_pose"], "xsens_offsets": batch["xsens_offsets"],
                 "goal_image_coords": batch["goal_image_coords"]}
        trans_obs_0 = move_to_device(Preprocessor().transform_obs(obs_0), device)
        trans_obs_g = move_to_device(Preprocessor().transform_obs(obs_g), device)

        # Output task dir (with frac prefix from prior viz_results if present)
        frac_solved_prev = prior_viz.get("frac_solved", None)
        prefix = f"{frac_solved_prev:.3f}_" if frac_solved_prev is not None else ""
        task_dir = os.path.join(log_dir, f"{prefix}{track_idx_name}")
        os.makedirs(task_dir, exist_ok=True)

        # ---- Tier 1: GT + static ----
        if args.render_gt:
            render_gt_and_static(task_dir, batch, fisheye_params, R_C_pelvis, t_C_pelvis)

        # ---- Per-step rollouts (Tier 2 lite) ----
        # Use vis-fidelity diffusion steps for all WM rollouts
        if args.peva_vis_diffusion_steps != args.peva_diffusion_steps:
            cem_planner.wm.peva_diffusion = create_diffusion(str(args.peva_vis_diffusion_steps))

        # All-step waypoint err / step-mje (re-derive; prefer prior values if available)
        gt_leaf_coords = batch["goal_image_coords"][0, _LEAF_IDX]
        image_size_px = nomad_config["image_size"][0]
        wp_err_per_step = [
            compute_waypoint_err(mu_history[t], gt_leaf_coords, image_size_px)
            for t in range(len(mu_history))
        ]
        all_xyz_per_step = prior_viz.get("all_xyz_per_step", [])

        if args.steps == "all":
            for step_idx in range(len(mu_history)):
                step_mu = mu_history[step_idx].to(device)
                saved_states, _ = multi_rollout_topk(
                    cem_planner.wm, trans_obs_0, trans_obs_g, step_mu,
                    num_rollouts=args.num_vis_rollouts, device=device,
                    algo=algo, top_k_save=1, topk_wm=1,
                )
                step_dir = os.path.join(task_dir, f"step{step_idx:03d}")
                save_mu_step_results(
                    saved_states[0]["state"], trans_obs_0, trans_obs_g,
                    fisheye_params, R_C_pelvis, t_C_pelvis,
                    save_path=step_dir,
                    render_skin=(fisheye_params is not None),
                )
                print(f"  step{step_idx:03d}: top1 mje={saved_states[0]['mje']:.4f}")

        # ---- Best variants (full top-k) ----
        # Pick best by re-derived per-step metrics (fall back to prior_viz if needed)
        if all_xyz_per_step:
            best_step_mje_idx = int(np.argmin(all_xyz_per_step))
        else:
            best_step_mje_idx = prior_viz.get("best_step_mje_idx", len(mu_history) - 1)

        if any(not np.isnan(x) for x in wp_err_per_step):
            wp_arr = np.where(np.isnan(wp_err_per_step), np.inf, wp_err_per_step)
            best_step_wp_idx = int(np.argmin(wp_arr))
        else:
            best_step_wp_idx = prior_viz.get("best_step_wp_idx", best_step_mje_idx)

        mu_best_mje = mu_history[best_step_mje_idx].to(device)
        mu_best_wp = mu_history[best_step_wp_idx].to(device)

        best_variants = [("best_mje", mu_best_mje, best_step_mje_idx)]
        if best_step_wp_idx != best_step_mje_idx:
            best_variants.append(("best_wp", mu_best_wp, best_step_wp_idx))

        variant_results = {}
        for vname, vmu, vstep in best_variants:
            saved_states, stats = multi_rollout_topk(
                cem_planner.wm, trans_obs_0, trans_obs_g, vmu,
                num_rollouts=args.num_vis_rollouts, device=device,
                algo=algo, top_k_save=args.top_k_save, topk_wm=args.top_k_save,
            )
            print(f"  [{vname}] top1 mje={saved_states[0]['mje']:.4f}  "
                  f"mean={stats['mean_mje']:.4f} ± {stats['std_mje']:.4f}")
            for ri, sstate in enumerate(saved_states):
                save_mu_step_results(
                    sstate["state"], trans_obs_0, trans_obs_g,
                    fisheye_params, R_C_pelvis, t_C_pelvis,
                    save_path=f"{task_dir}/{vname}/rollout_{ri}_mje{sstate['mje']:.3f}",
                    render_skin=(fisheye_params is not None),
                )
            variant_results[vname] = {
                "mu_step": vstep,
                "wp_err": wp_err_per_step[vstep] if wp_err_per_step else float("nan"),
                "step_mje": all_xyz_per_step[vstep] if all_xyz_per_step else float("nan"),
                "rollout_top1_mje": saved_states[0]["mje"],
                "rollout_top1_leaf_xyz": saved_states[0]["leaf_xyz"],
                "rollout_topk_mjes": [s["mje"] for s in saved_states],
                "rollout_topk_leaf_xyz": [s["leaf_xyz"] for s in saved_states],
                "rollout_mean_mje": stats["mean_mje"],
                "rollout_median_mje": stats["median_mje"],
                "rollout_std_mje": stats["std_mje"],
                "rollout_mean_leaf_xyz": stats["mean_leaf_xyz"],
                "rollout_stats": stats,
            }

        # frac_solved (refreshed using new top1 vs prior init_mje)
        init_mje = prior_viz.get("init_mje", None)
        best_top1 = variant_results["best_mje"]["rollout_top1_mje"]
        mean_mje_best = variant_results["best_mje"]["rollout_mean_mje"]
        if init_mje is not None and init_mje > 0:
            frac_solved = max(0.0, 1.0 - best_top1 / init_mje)
            frac_solved_mean = max(0.0, 1.0 - mean_mje_best / init_mje)
        else:
            frac_solved = 0.0
            frac_solved_mean = 0.0
        print(f"  frac_solved: {frac_solved:.2%} (top1)  {frac_solved_mean:.2%} (mean)")

        # Rename task_dir to use the freshly-computed frac_solved
        new_task_dir = os.path.join(log_dir, f"{frac_solved:.3f}_{track_idx_name}")
        if new_task_dir != task_dir:
            os.rename(task_dir, new_task_dir)
            task_dir = new_task_dir

        torch.save({
            "track": track,
            "start_index": start_index,
            "goal_index": batch["goal_index"].item(),
            "mu_best_mje": mu_best_mje.cpu(),
            "mu_best_wp": mu_best_wp.cpu(),
            "best_step_mje_idx": best_step_mje_idx,
            "best_step_wp_idx": best_step_wp_idx,
            "all_xyz_per_step": all_xyz_per_step,
            "wp_err_per_step": wp_err_per_step,
            "init_mje": init_mje,
            "frac_solved": frac_solved,
            "frac_solved_mean": frac_solved_mean,
            "variants": variant_results,
            "source_log_dir": args.source_log_dir,
            "source_subdir": source_subdir,
        }, f"{task_dir}/viz_results.pth")

        bm = variant_results["best_mje"]
        bw = variant_results.get("best_wp", bm)
        task_results.append({
            "task_name": track_idx_name,
            "frac_solved": frac_solved,
            "frac_solved_mean": frac_solved_mean,
            "init_mje": init_mje,
            "best_mje_step": bm["mu_step"],
            "best_mje_top1": bm["rollout_top1_mje"],
            "best_mje_mean": bm["rollout_mean_mje"],
            "best_wp_step": bw["mu_step"],
            "best_wp_top1": bw["rollout_top1_mje"],
            "best_wp_mean": bw["rollout_mean_mje"],
        })

        # Reset diffusion steps (in case any fast-step path is reused)
        if args.peva_vis_diffusion_steps != args.peva_diffusion_steps:
            cem_planner.wm.peva_diffusion = create_diffusion(str(args.peva_diffusion_steps))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RE-RENDER RANKING (sorted by frac_solved descending)")
    print("=" * 60)
    task_results.sort(key=lambda x: x["frac_solved"], reverse=True)
    for rank, tr in enumerate(task_results, 1):
        print(f"  {rank:<3} frac={tr['frac_solved']:.3f} init={tr['init_mje']:.4f} "
              f"top1={tr['best_mje_top1']:.4f} mean={tr['best_mje_mean']:.4f}  "
              f"{tr['task_name']}")

    torch.save(task_results, f"{log_dir}/task_ranking.pth")
    print(f"\nResults saved to {log_dir}/")


def build_argparser():
    parser = argparse.ArgumentParser(description="Re-render visualizations from an existing CEM viz log dir")

    # Required: source dir
    parser.add_argument("--source_log_dir", type=str, required=True,
                        help="Path to existing logs/cem_viz/{datetime}:viz_... directory")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific task subdir names (or just track-sX-gY) to re-render. "
                             "Default: all tasks in source_log_dir.")
    parser.add_argument("--steps", choices=["all", "best"], default="all",
                        help="'all' = render per-step rollouts + best variants; "
                             "'best' = only best_mje/best_wp variants")
    parser.add_argument("--render_gt", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-render GT skeleton + context/goal frames")

    # Algo / dataset (auto-derived from source dir name when possible)
    parser.add_argument("-a", "--algo", choices=["peva", "waypoint", "waypoint_point3d"], default=None)
    parser.add_argument("--use_leafxyz_as_cost", action="store_true")
    parser.add_argument("-n", "--num_samples", type=int, default=None)
    parser.add_argument("-t", "--topk", type=int, default=None)
    parser.add_argument("-v", "--var_scale", type=float, default=None)
    parser.add_argument("-o", "--opt_steps", type=int, default=None)
    parser.add_argument("-H", "--horizon", type=int, default=None)
    parser.add_argument("-N", "--num_eval_samples", type=int, default=1)
    parser.add_argument("-R", "--num_vis_rollouts", type=int, default=None)
    parser.add_argument("--top_k_save", type=int, default=3)

    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--min_dist_cat", type=int, default=None)
    parser.add_argument("--max_dist_cat", type=int, default=None)
    parser.add_argument("--curr_time_stride", type=int, default=1)
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--no_skin", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--camera_data_folder", type=str,
                        default="/home/anw2067/scratch/nymeria_visibility_matrix")

    # Models
    parser.add_argument("--peva_config", type=str,
                        default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str,
                        default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_context_size", type=int, default=15)
    parser.add_argument("--peva_diffusion_steps", type=int, default=None)
    parser.add_argument("--peva_vis_diffusion_steps", type=int, default=None)

    parser.add_argument("--nomad_model", type=str, default="draw", choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()

    # Auto-derive missing args from the source dir name
    derived = parse_run_name(args.source_log_dir)
    for k, v in derived.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    # Hard fallbacks for any args still unset
    defaults = {
        "algo": "waypoint", "horizon": 1, "num_samples": 8, "topk": 8, "var_scale": 0.5,
        "opt_steps": 12, "num_vis_rollouts": 128, "peva_diffusion_steps": 64,
        "peva_vis_diffusion_steps": 250, "min_dist_cat": 8, "max_dist_cat": 8,
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]

    print(f"Algo: {args.algo}  horizon={args.horizon}  num_vis_rollouts={args.num_vis_rollouts}  "
          f"vis_ds={args.peva_vis_diffusion_steps}  dist={args.min_dist_cat}-{args.max_dist_cat}")

    main(args)
