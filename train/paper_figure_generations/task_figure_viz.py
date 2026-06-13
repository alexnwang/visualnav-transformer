"""Paper-figure generator: render the GROUND-TRUTH motion of a single task three ways,
projected onto the current observation frame.

No policy / PEVA / sampling — this only consumes the dataset's GT `deltas`, `first_pose`,
`xsens_offsets`, the current frame, and the per-track camera calibration, then reuses the
same skeleton/SMPL machinery the planners use (planning.wrappers.build_skeleton_top_seq).

For one task it writes, under --log_dir/{task_name}/:

  noskin/  (overlay='skeleton', stick figure)
    actions.webp   GT action skeleton, t=1..T  (movement)
    goal_pose.png  final goal pose skeleton  (== last action frame)
    waypoints.png  the 4 leaf joints (Pelvis/Head/R_Hand/L_Hand) of the goal pose
    combined.webp  [skel_1..skel_T] -> [goal skeleton] -> [waypoints]
  skin/    (overlay='both', SMPL mesh with skeleton drawn over it)
    actions.webp / goal_pose.png / combined.webp   (waypoints.png is unskinned, same as above)
  combined_mixed.webp  [skin_1..skin_T] -> [goal skin] -> [goal skeleton] -> [waypoints]

The combined webps play action frames at --fps, then HOLD each trailing
goal/waypoint frame for --hold_ms (default 400ms).
"""
import argparse
import os
import sys

# Make imports work regardless of the launch directory.
_TRAIN_DIR = "/home/anw2067/visualnav-transformer/train"
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)
os.chdir(_TRAIN_DIR)

import yaml
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from plan_cem import MODEL_DIRECTORY
from planning.nymeria_dataset import NymeriaPlanningDataset
from planning.wrappers import build_skeleton_top_seq
from planning.vis_utils import disable_logging, pose_to_image_coords_v2, draw_image_coords, add_dark_glow
from vint_train.data.misc import XSensConstants
from vint_train.training.nymeria_training_utils import get_action_smpl_torch


def _to_pil(x):  # (3, H, W) float[0,1] -> PIL
    return Image.fromarray((x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy())


def save_png(path, frame):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _to_pil(frame).save(path)
    print("wrote", path, flush=True)


def save_webp(path, frames, durations):
    """frames: list of (3,H,W) tensors. durations: per-frame ms (list, len == frames)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pil = [_to_pil(f) for f in frames]
    pil[0].save(path, format="WEBP", save_all=True, append_images=pil[1:],
                duration=durations, loop=0, lossless=True, method=4)
    print("wrote", path, flush=True)


def waypoints_frame(curr_image, goal_pose, R_C_pelvis, t_C_pelvis, fisheye_params,
                    xsens_offsets, image_size, marker_scale=1.8, glow=True,
                    glow_color=(0, 0, 0)):
    """Draw only the 4 leaf-joint waypoints of the goal pose on the current frame.

    Projects the goal pose, masks every non-leaf joint to (-1,-1) so draw_image_coords
    emits just the 4 colored dots (edges auto-skip when a parent is invisible). Leaf
    colors (red/green/blue/yellow) come from vis_utils._default_per_joint_colors.
    `marker_scale` multiplies the dot size; `glow=True` lays a soft dark halo behind
    each dot for contrast against busy backgrounds."""
    coords = pose_to_image_coords_v2(
        goal_pose, R_C_pelvis, t_C_pelvis, fisheye_params, xsens_offsets,
        image_size=image_size,
    ).clone()  # (1, 15, 2)
    leaf = set(XSensConstants.leaf_indices)
    for j in range(coords.shape[1]):
        if j not in leaf:
            coords[0, j] = -1
    img = _to_pil(curr_image)
    scale = image_size / 224.0 * marker_scale
    if glow:
        img = add_dark_glow(img, coords, scale=scale, glow_color=glow_color)
    draw_image_coords(ImageDraw.Draw(img), coords, show_text=False, scale=scale, dot_stroke=0.3)
    return transforms.ToTensor()(img).to(curr_image.device)


def main(args):
    device = "cuda"
    disable_logging()

    with open(args.nomad_config, "r") as f:
        config = yaml.safe_load(f)
    image_size = config["image_size"]
    context_size = config["context_size"]
    data_config = config["datasets"]["nymeria"]

    ds = NymeriaPlanningDataset(
        tasks_file=args.tasks_file,
        data_folder=data_config["data_folder"],
        image_size=image_size,
        context_size=context_size,
        goal_type=config.get("goal_type", None),
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        gaussian_normalization_stats_path=data_config.get(
            "gaussian_normalization_stats_path", None
        ),
    )
    batch = ds[args.task_index]

    curr_image = batch["obs_images"][-1].to(device)          # (3, H, W)
    render_size = curr_image.shape[-1]  # int spatial size; config image_size may be a list
    deltas = batch["deltas"].to(device)                      # (T, 48)
    first_pose = batch["first_pose"].to(device)              # (1, 48)
    xsens_offsets = batch["xsens_offsets"].to(device)        # (15, 3)
    track = batch["dataset_track"]
    start_index = int(batch["start_index"])
    goal_index = int(batch["goal_index"])
    task_name = f"{track}-s{start_index}-g{goal_index}"
    n_steps = deltas.shape[0]

    # camera params for projection
    cam = torch.load(
        os.path.join(args.camera_data_folder, track, "camera_data.pt"),
        weights_only=False,
    )
    T_mat = cam["T_C_pelvis"][start_index]
    fisheye_params = cam["fisheye_params"].to(device).float()
    R_C_pelvis = T_mat[:3, :3].to(device).float()
    t_C_pelvis = T_mat[:3, 3].to(device).float()

    deltas_b = deltas[None]            # (1, T, 48)
    first_pose_b = first_pose[None]    # (1, 1, 48)

    def top_seq(overlay):
        return build_skeleton_top_seq(
            curr_image, deltas_b, first_pose_b, xsens_offsets,
            fisheye_params, R_C_pelvis, t_C_pelvis,
            render_size, n_steps, overlay=overlay,
            smpl_alpha=args.smpl_alpha, show_text=False,
        )  # (T, 3, H, W)

    skel_seq = top_seq("skeleton")
    skin_seq = top_seq("both")  # SMPL mesh with the skeleton drawn over it

    goal_pose = get_action_smpl_torch(
        first_pose_b, deltas_b, XSensConstants.upper_body_num_parts
    )[:, -1]  # (1, 48)
    wp_frame = waypoints_frame(
        curr_image, goal_pose, R_C_pelvis, t_C_pelvis, fisheye_params,
        xsens_offsets, render_size,
    )

    step_ms = round(1000 / args.fps)
    out = os.path.join(args.log_dir, task_name)

    # --- noskin / skin sets ---
    for sub, seq in [("noskin", skel_seq), ("skin", skin_seq)]:
        d = os.path.join(out, sub)
        save_webp(f"{d}/actions.webp", list(seq), [step_ms] * n_steps)
        save_png(f"{d}/goal_pose.png", seq[-1])
        save_png(f"{d}/waypoints.png", wp_frame)
        combined = list(seq) + [seq[-1], wp_frame]
        save_webp(f"{d}/combined.webp", combined,
                  [step_ms] * n_steps + [args.hold_ms, args.hold_ms])

    # --- mixed combined: skin actions -> skin goal -> skeleton goal -> waypoints ---
    mixed = list(skin_seq) + [skin_seq[-1], skel_seq[-1], wp_frame]
    save_webp(f"{out}/combined_mixed.webp", mixed,
              [step_ms] * n_steps + [args.hold_ms] * 3)

    print(f"\ndone: {task_name}  (T={n_steps})  ->  {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomad_model", type=str, default="draw_mask",
                        choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None,
                        help="config YAML override; defaults to MODEL_DIRECTORY[nomad_model]")
    parser.add_argument("--tasks_file", type=str,
                        default="/home/anw2067/visualnav-transformer/train/data_splits/"
                                "nymeria/test/viz_shortlists/viz_selected.pkl")
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument("--log_dir", type=str,
                        default="/home/anw2067/visualnav-transformer/train/logs/task_figure_viz")
    parser.add_argument("--camera_data_folder", type=str,
                        default="/scratch/anw2067/nymeria_visibility_matrix")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--hold_ms", type=int, default=1200,
                        help="hold time (ms) for trailing goal/waypoint frames in combined webps")
    parser.add_argument("--smpl_alpha", type=float, default=0.7)
    args = parser.parse_args()
    if args.nomad_config is None:
        args.nomad_config = MODEL_DIRECTORY[args.nomad_model][0]
    main(args)
