"""High-res paper figures for one task: GT and best-of-N policy-predicted action
sequences rendered on the high-res (1408) current-obs frame.

Outputs under --log_dir/{task}/:
  noskin/{gt_actions.webp, pred_actions.webp, goal_pose.png}   overlay='skeleton'
  skin/  {gt_actions.webp, pred_actions.webp, goal_pose.png}   overlay='both' (mesh+skeleton)
  waypoints.png                                                 GT goal-pose leaf joints

Policy runs at the model's 224 input (drawn-goal, like plan_policy_viz); the resulting
deltas are resolution-independent and rendered on the 1408 frame. Best-of-N = argmin
final-pose LEAF MJE vs GT. The current-obs frame is the high-res mp4 frame produced by
extract_highres_frame.py (pass via --highres_frame).
"""
import argparse, os, pickle, random, sys, tempfile

_TRAIN_DIR = "/home/anw2067/visualnav-transformer/train"
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)
os.chdir(_TRAIN_DIR)

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from plan_cem import MODEL_DIRECTORY
from planning.nymeria_dataset import NymeriaPlanningDataset
from planning.sampling import policy_sample
from planning.utils import _compute_part_distance_matrices, load_policy
from planning.wrappers import build_skeleton_top_seq
from planning.vis_utils import disable_logging
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch

from paper_figure_generations.task_figure_viz import save_png, save_webp, waypoints_frame


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main(args):
    _seed(args.seed)
    device = "cuda"
    disable_logging()

    policy, noise_scheduler, _, cfg = load_policy(args.nomad_config, args.nomad_checkpoint, device=device)
    pred_horizon = cfg["len_traj_pred"]
    action_dim = cfg["input_dims"]
    context_size = cfg["context_size"]
    image_size = cfg["image_size"]            # [224,224] (policy input)
    data_cfg = cfg["datasets"]["nymeria"]
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # one-task dataset via a temp pkl
    with tempfile.NamedTemporaryFile("wb", suffix=".pkl", delete=False) as tf:
        pickle.dump([{"track": args.track, "curr_time": args.curr_time, "goal_time": args.goal_time}], tf)
        tasks_file = tf.name
    ds = NymeriaPlanningDataset(
        tasks_file=tasks_file, data_folder=data_cfg["data_folder"],
        image_size=image_size, context_size=context_size,
        goal_type=cfg.get("goal_type", None),
        waypoint_spacing=data_cfg.get("waypoint_spacing", 1),
        gaussian_normalization_stats_path=data_cfg.get("gaussian_normalization_stats_path", None),
    )
    b = ds[0]
    os.unlink(tasks_file)

    obs_images = b["obs_images"].to(device)            # (ctx+1,3,224,224)
    goal_image = b["goal_image"].to(device)            # (3,224,224) drawn goal
    context_poses = b["context_poses"].to(device)      # (ctx+1,48)
    deltas = b["deltas"].to(device)                    # (T,48)
    first_pose = b["first_pose"].to(device)            # (1,48)
    xsens_offsets = b["xsens_offsets"].to(device)      # (15,3)
    n_steps = deltas.shape[0]
    task_name = f"{args.track}-s{args.curr_time}-g{args.goal_time}"

    # high-res current-obs frame (1408)
    hr = Image.open(args.highres_frame).convert("RGB")
    curr_hr = transforms.ToTensor()(hr).to(device)     # (3,H,W) float[0,1]
    render_size = curr_hr.shape[-1]
    print(f"{task_name}: T={n_steps}  high-res curr_obs={tuple(curr_hr.shape)}", flush=True)

    # camera params (resolution-independent; build_skeleton_top_seq scales by render_size)
    cam = torch.load(os.path.join(args.camera_data_folder, args.track, "camera_data.pt"), weights_only=False)
    T_mat = cam["T_C_pelvis"][args.curr_time]
    fisheye_params = cam["fisheye_params"].to(device).float()
    R_C_pelvis = T_mat[:3, :3].to(device).float()
    t_C_pelvis = T_mat[:3, 3].to(device).float()

    skel = XsensSkeleton(xsens_offsets)
    deltas_b = deltas[None]; first_pose_b = first_pose[None]
    gt_actions = get_action_smpl_torch(first_pose_b, deltas_b, XSensConstants.upper_body_num_parts)

    def seq(delt, overlay):
        return build_skeleton_top_seq(
            curr_hr, delt, first_pose_b, xsens_offsets,
            fisheye_params, R_C_pelvis, t_C_pelvis,
            render_size, n_steps, overlay=overlay,
            smpl_alpha=args.smpl_alpha, show_text=False)

    step_ms = round(1000 / args.fps * args.slow)
    out = os.path.join(args.log_dir, task_name)

    # --- GT renders + waypoints: identical across runs, so rendered once ---
    for sub, overlay in [("noskin", "skeleton"), ("skin", "both")]:
        gt_seq = seq(deltas_b, overlay)
        save_webp(f"{out}/{sub}/gt_actions.webp", list(gt_seq), [step_ms] * n_steps)
        save_png(f"{out}/{sub}/goal_pose.png", gt_seq[-1])
    goal_pose = gt_actions[:, -1]
    wp = waypoints_frame(curr_hr, goal_pose, R_C_pelvis, t_C_pelvis, fisheye_params, xsens_offsets, render_size)
    save_png(f"{out}/waypoints.png", wp)

    # --- policy: each run independently draws N samples (seed+i) and keeps the
    #     best by final-pose LEAF MJE vs GT. Multiple runs give several distinct
    #     best-of-N predictions to pick a good-looking one for the figure. ---
    N = args.num_eval_samples
    policy_obs = imagenet_norm(obs_images[-context_size - 1:])[None]   # (1, ctx+1, 3, H, W)
    goal_img = imagenet_norm(goal_image[None])
    for run in range(args.num_runs):
        _seed(args.seed + run)
        pred_deltas = policy_sample(
            policy, noise_scheduler,
            policy_obs.expand(N, -1, -1, -1, -1), goal_img.expand(N, -1, -1, -1),
            context_poses[-context_size - 1:][None].expand(N, -1, -1),
            pred_horizon, action_dim, device,
        )
        pred_actions = get_action_smpl_torch(first_pose_b.expand(N, -1, -1), pred_deltas, XSensConstants.upper_body_num_parts)
        _, _, leaf_xyz, _ = _compute_part_distance_matrices(
            pred_actions[:, -1], gt_actions.expand(N, -1, -1)[:, -1], skel)
        best = int(torch.argmin(leaf_xyz).item())
        pred_deltas_best = pred_deltas[best:best + 1]
        run_out = out if args.num_runs == 1 else os.path.join(out, f"run{run:02d}")
        for sub, overlay in [("noskin", "skeleton"), ("skin", "both")]:
            pred_seq = seq(pred_deltas_best, overlay)
            save_webp(f"{run_out}/{sub}/pred_actions.webp", list(pred_seq), [step_ms] * n_steps)
        print(f"  run {run:02d} seed={args.seed + run} best-of-{N} leaf-MJE={float(leaf_xyz[best]):.4f}", flush=True)

    print(f"done: {task_name} -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--curr_time", type=int, required=True)
    p.add_argument("--goal_time", type=int, required=True)
    p.add_argument("--highres_frame", required=True, help="path to extracted 1408 curr_obs png")
    p.add_argument("--log_dir", default="/home/anw2067/visualnav-transformer/train/logs/figures/talk_highres/task_figure_viz_hr")
    p.add_argument("--nomad_model", default="draw_mask", choices=list(MODEL_DIRECTORY.keys()))
    p.add_argument("--nomad_config", default=None)
    p.add_argument("--nomad_checkpoint", default=None)
    p.add_argument("-N", "--num_eval_samples", type=int, default=64)
    p.add_argument("--num_runs", type=int, default=1,
                   help="independent best-of-N runs; >1 writes preds to run{i}/ subdirs (GT rendered once)")
    p.add_argument("--seed", type=int, default=42, help="base RNG seed; run i uses seed+i")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--slow", type=float, default=1.0,
                   help="multiply each webp frame's duration (1.5 = 1.5x slower playback)")
    p.add_argument("--smpl_alpha", type=float, default=0.7)
    p.add_argument("--camera_data_folder", default="/scratch/anw2067/nymeria_visibility_matrix")
    args = p.parse_args()
    if args.nomad_config is None or args.nomad_checkpoint is None:
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    main(args)
