"""Visualize best-of-N policy rollouts vs ground truth for a fixed task list.

Mirrors plan_policy_only.py's policy forward pass (NoMaD pose-diffusion policy),
then renders the predicted and GT pose trajectories with the SAME machinery the
CEM viz uses: build_skeleton_top_seq (one skeleton-overlay frame per timestep)
+ save_action_obs_sequence_viz (2-row strip).

For each task:
  1. Draw N policy samples, reconstruct poses, score MJE vs GT, keep the best.
  2. Save gt_seq.png   (GT skeleton per timestep, top row; GT frames, bottom row).
  3. Save pred_seq.png (best predicted skeleton per timestep, top; GT frames, bottom).

Unlike plan_policy_only, the goal fed to the policy is the *drawn* goal image
(goal_type=draw) — matching training and planning/sampling.py — not the raw
future frame.
"""
import argparse
import os
import random
import sys
from collections import defaultdict

# Make imports work regardless of the launch directory.
_TRAIN_DIR = "/home/anw2067/visualnav-transformer/train"
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)
os.chdir(_TRAIN_DIR)

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from plan_cem import MODEL_DIRECTORY
from planning.nymeria_dataset import NymeriaPlanningDataset
from planning.sampling import policy_sample
from planning.utils import _compute_part_distance_matrices, load_policy
from planning.wrappers import build_skeleton_top_seq
from planning.vis_utils import disable_logging
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch

from PIL import Image, ImageDraw, ImageFont


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _to_img(x):  # (3, H, W) float[0,1] -> PIL
    return Image.fromarray((x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy())


def compose_labeled_grid(rows_cells, col_labels, row_labels, title="", cell=224):
    """Lay out rows_cells (list of rows; each row a list of (3,H,W) tensors) into a
    labeled grid: title bar on top, column headers, and row labels on the left."""
    LM, TITLE_H, HDR_H, PAD = 46, 18, 16, 2
    nrows, ncols = len(rows_cells), len(rows_cells[0])
    W = LM + ncols * (cell + PAD) + PAD
    H = TITLE_H + HDR_H + nrows * (cell + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (15, 15, 15))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((LM + 2, 3), title, fill=(255, 220, 60), font=_font(13))
    for c, lab in enumerate(col_labels):
        d.text((LM + c * (cell + PAD) + 3, TITLE_H + 1), str(lab), fill=(200, 200, 200), font=_font(12))
    for r, row in enumerate(rows_cells):
        y = TITLE_H + HDR_H + r * (cell + PAD)
        rc = (120, 255, 120) if str(row_labels[r]).upper() == "GT" else (120, 200, 255)
        d.text((4, y + cell // 2 - 8), str(row_labels[r]), fill=rc, font=_font(15))
        for c, cellimg in enumerate(row):
            canvas.paste(_to_img(cellimg).resize((cell, cell)), (LM + c * (cell + PAD), y))
    return canvas


def save_labeled_grid(save_path, rows_cells, col_labels, row_labels, title=""):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    compose_labeled_grid(rows_cells, col_labels, row_labels, title).save(save_path)


def save_compare_webp(save_path, input_img, gt_seq, pred_seq, goalobs_img, title="", fps=4):
    """Animated full-color WebP, 2 labeled rows (GT / Pred) x 3 cols
    (input | action@t | goal obs). Outer columns are static; the middle 'action'
    column animates over t. WebP keeps true RGB (no GIF 256-color palette)."""
    T = gt_seq.shape[0]
    frames = []
    for t in range(T):
        rows = [[input_img, gt_seq[t], goalobs_img],
                [input_img, pred_seq[t], goalobs_img]]
        frames.append(compose_labeled_grid(
            rows, ["input", "action", "goal obs"], ["GT", "Pred"],
            title=f"{title}  step {t + 1}/{T}"))
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    frames[0].save(save_path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, lossless=True, method=4)


def save_mje_hist(summary, save_path):
    """Overlaid per-set density histograms of best-of-N final-pose MJE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred = defaultdict(list)
    init = defaultdict(list)
    for label, mje, leaf, init_mje, name in summary:
        pred[label].append(mje)
        init[label].append(init_mje)
    allv = [m for _, m, _, _, _ in summary]
    if not allv:
        return
    # clip the x-range to the 99th pct so heavy tails don't crush the bulk
    hi = float(np.percentile(allv, 99))
    edges = np.linspace(0, hi, 31)
    colors = {"hand": "#4c9be8", "balanced-hand": "#1f4e8c",
              "lateral": "#e8794c", "balanced-lateral": "#a83232",
              "balanced": "#5fbf60"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals in pred.items():
        c = colors.get(label, None)
        med = float(np.median(vals))
        imed = float(np.median(init[label]))
        # best-of-N policy MJE (solid) + stay-put baseline (dashed, same color)
        ax.hist(vals, bins=edges, histtype="step", density=True, linewidth=2,
                color=c, label=f"{label} (n={len(vals)}, med={med:.3f}, init={imed:.3f})")
        ax.hist(init[label], bins=edges, histtype="step", density=True, linewidth=1.2,
                color=c, ls="--", alpha=0.7)
    ax.set_xlabel("final-pose MJE (m)  —  solid: best-of-N policy, dashed: stay-put baseline")
    ax.set_ylabel("density")
    ax.set_title("Policy MJE vs stay-put baseline over deduped task sets")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    print("wrote", save_path, flush=True)


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda"
    disable_logging()

    policy, noise_scheduler, _, nomad_config = load_policy(
        args.nomad_config, args.nomad_checkpoint, device=device
    )
    pred_horizon = nomad_config["len_traj_pred"]
    action_dim = nomad_config["input_dims"]
    context_size = nomad_config["context_size"]
    image_size = nomad_config["image_size"]
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_config = nomad_config["datasets"]["nymeria"]

    def build_loader(tasks_file):
        ds = NymeriaPlanningDataset(
            tasks_file=tasks_file,
            data_folder=data_config["data_folder"],
            image_size=image_size,
            context_size=context_size,
            goal_type=nomad_config.get("goal_type", None),
            waypoint_spacing=data_config.get("waypoint_spacing", 1),
            gaussian_normalization_stats_path=data_config.get(
                "gaussian_normalization_stats_path", None
            ),
        )
        return ds, DataLoader(ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers,
                              persistent_workers=(args.num_workers > 0))

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    cam_cache = {}
    N = args.num_eval_samples
    summary = []  # (set_label, mje, leaf, task_name)

    if args.task_sets:
        task_sets = [tuple(s.split("=", 1)) for s in args.task_sets]
    else:
        task_sets = [(os.path.splitext(os.path.basename(args.tasks_file))[0], args.tasks_file)]

    def iter_tasks():
        for set_label, tasks_file in task_sets:
            ds, dl = build_loader(tasks_file)
            for j, b in enumerate(dl):
                yield set_label, j, len(ds), b

    for set_label, idx, n_in_set, batch in iter_tasks():
        obs_images = batch["obs_images"].to(device)        # 1, ctx+1, 3, H, W
        goal_image = batch["goal_image"].to(device)        # 1, 3, H, W  (drawn goal)
        goal_obs = batch["goal_obs"].to(device)            # 1, 3, H, W  (raw goal frame)
        context_poses = batch["context_poses"].to(device)  # 1, ctx+1, 48
        deltas = batch["deltas"].to(device)                # 1, T, 48
        first_pose = batch["first_pose"].to(device)        # 1, 1, 48
        xsens_offsets = batch["xsens_offsets"].to(device)  # 1, 15, 3
        gt_frames = batch["gt_frames"][0].to(device)       # T, 3, H, W

        track = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index = batch["goal_index"].item()
        task_name = f"{track}-s{start_index}-g{goal_index}"
        n_steps = gt_frames.shape[0]

        skel = XsensSkeleton(xsens_offsets[0])
        gt_actions = get_action_smpl_torch(
            first_pose, deltas, XSensConstants.upper_body_num_parts
        )

        # camera params for projection
        if track not in cam_cache:
            cam_cache[track] = torch.load(
                os.path.join(args.camera_data_folder, track, "camera_data.pt"),
                weights_only=False,
            )
        cam = cam_cache[track]
        T_mat = cam["T_C_pelvis"][start_index]
        fisheye_params = cam["fisheye_params"].to(device).float()
        R_C_pelvis = T_mat[:3, :3].to(device).float()
        t_C_pelvis = T_mat[:3, 3].to(device).float()

        # --- policy: draw N samples, pick best by mean joint error ---
        policy_obs = imagenet_norm(
            obs_images[:, -context_size - 1:].flatten(0, 1)
        ).unflatten(0, (1, context_size + 1))
        goal_img = imagenet_norm(goal_image)

        policy_obs_N = policy_obs.expand(N, -1, -1, -1, -1)
        goal_img_N = goal_img.expand(N, -1, -1, -1)
        ctx_poses_N = context_poses[:, -context_size - 1:].expand(N, -1, -1)
        first_pose_N = first_pose.expand(N, -1, -1)
        gt_actions_N = gt_actions.expand(N, -1, -1)

        pred_deltas = policy_sample(
            policy, noise_scheduler, policy_obs_N, goal_img_N, ctx_poses_N,
            pred_horizon, action_dim, device,
        )
        pred_actions = get_action_smpl_torch(
            first_pose_N, pred_deltas, XSensConstants.upper_body_num_parts
        )
        xyz_dist, _, leaf_xyz, _ = _compute_part_distance_matrices(
            pred_actions[:, -1], gt_actions_N[:, -1], skel
        )
        mje_per = xyz_dist.mean(dim=-1)            # (N,)
        best = int(torch.argmin(mje_per).item())
        best_mje = float(mje_per[best])
        best_leaf = float(leaf_xyz[best].mean())
        pred_deltas_best = pred_deltas[best:best + 1]  # (1, T, 48)

        # stay-put baseline: hold the current pose (zero deltas) vs GT final pose
        init_actions = get_action_smpl_torch(
            first_pose, torch.zeros_like(deltas), XSensConstants.upper_body_num_parts
        )
        init_xyz, _, _, _ = _compute_part_distance_matrices(
            init_actions[:, -1], gt_actions[:, -1], skel
        )
        init_mje = float(init_xyz.mean())

        if not args.no_viz:
            # --- render with existing machinery ---
            # For each overlay style, build the GT and predicted top-row sequences
            # (one pose-overlay frame per timestep) and stack GT above pred.
            # Each row: [drawn goal | step1 ... stepT | raw goal obs].
            curr_image = obs_images[0, -1]
            task_dir = f"{args.log_dir}/{best_mje:.3f}_{task_name}"

            def top_seq(delt, overlay):
                return build_skeleton_top_seq(
                    curr_image, delt, first_pose, xsens_offsets[0],
                    fisheye_params, R_C_pelvis, t_C_pelvis,
                    curr_image.shape[-1], n_steps, overlay=overlay, show_text=False,
                )

            gt_skel, pred_skel = top_seq(deltas, "skeleton"), top_seq(pred_deltas_best, "skeleton")
            gt_skin, pred_skin = top_seq(deltas, "both"), top_seq(pred_deltas_best, "both")

            in_img, goal_img_r = goal_image[0], goal_obs[0]
            title = f"{task_name}  mje={best_mje:.3f}"
            step_cols = ["input"] + [f"t{i+1}" for i in range(n_steps)] + ["goal obs"]
            def cells(seq):
                return [in_img] + [seq[i] for i in range(n_steps)] + [goal_img_r]

            save_labeled_grid(f"{task_dir}/compare.png",
                              [cells(gt_skel), cells(pred_skel)], step_cols, ["GT", "Pred"], title)
            save_labeled_grid(f"{task_dir}/compare_skin.png",
                              [cells(gt_skin), cells(pred_skin)], step_cols, ["GT", "Pred"], title)

            # 8-frame animation (skinned): 2 rows (GT top, pred bottom) x 3 cols (input | action@t | goal obs)
            save_compare_webp(f"{task_dir}/compare.webp", in_img, gt_skin, pred_skin,
                              goal_img_r, title=title, fps=args.gif_fps)

        summary.append((set_label, best_mje, best_leaf, init_mje, task_name))
        print(f"[{set_label} {idx+1}/{n_in_set}] {task_name}  best_mje={best_mje:.4f}  "
              f"leaf={best_leaf:.4f}  init_mje={init_mje:.4f}", flush=True)

    # --- outputs ---
    if args.mje_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.mje_csv)), exist_ok=True)
        with open(args.mje_csv, "w") as f:
            f.write("set,task,mje,leaf,init_mje\n")
            for label, mje, leaf, init_mje, name in summary:
                f.write(f"{label},{name},{mje:.6f},{leaf:.6f},{init_mje:.6f}\n")
        print("wrote", args.mje_csv, flush=True)

    if args.hist_out:
        save_mje_hist(summary, args.hist_out)

    print("\n=== per-set MJE summary (best-of-N | stay-put baseline) ===")
    by = defaultdict(list)
    for label, mje, leaf, init_mje, name in summary:
        by[label].append((mje, init_mje))
    for label, vals in by.items():
        vs = sorted(m for m, _ in vals)
        iv = sorted(i for _, i in vals)
        print(f"  {label:10s} n={len(vs):4d}  mje[mean={sum(vs)/len(vs):.4f} "
              f"median={vs[len(vs)//2]:.4f} min={vs[0]:.4f} max={vs[-1]:.4f}]  "
              f"init[median={iv[len(iv)//2]:.4f}]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomad_model", type=str, default="draw_mask",
                        choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)
    parser.add_argument("--tasks_file", type=str, default=None)
    parser.add_argument("--task_sets", type=str, nargs="+", default=None,
                        help="one or more 'label=path.pkl' entries; runs all under one model load")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="dataloader workers; >0 prefetches upcoming tasks' images")
    parser.add_argument("--gif_fps", type=int, default=4)
    parser.add_argument("--no_viz", action="store_true",
                        help="skip per-task skeleton renders; only collect MJE")
    parser.add_argument("--mje_csv", type=str, default=None,
                        help="write per-task MJE table to this CSV")
    parser.add_argument("--hist_out", type=str, default=None,
                        help="save the per-set MJE distribution figure to this path")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--camera_data_folder", type=str,
                        default="/scratch/anw2067/nymeria_visibility_matrix")
    args = parser.parse_args()
    if not args.tasks_file and not args.task_sets:
        parser.error("provide --tasks_file or --task_sets")
    if not args.no_viz and not args.log_dir:
        parser.error("--log_dir is required unless --no_viz is set")
    if args.nomad_config is None or args.nomad_checkpoint is None:
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    main(args)
