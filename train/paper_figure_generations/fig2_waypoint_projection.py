"""Paper fig 2 recreation — improved waypoint-projection visualization.

Geometry
--------
  - 3D scene with two XSens upper-body skeletons: CURRENT pose and GOAL pose
  - Camera is at the current head. The camera's forward axis is the head's
    +X axis in world (from the head's full 3D orientation, NOT just yaw).
  - The goal_image (= current observation, optionally with waypoints) is
    pasted onto a 3D plane perpendicular to that forward axis, positioned
    between the current head and the goal body along fwd so that the rays
    from each goal leaf joint (Pelvis/Head/R_Hand/L_Hand) to the camera
    origin pass through the plane.
  - The plane is sized so that the ray intersections lie inside the image
    with configurable padding.

This fixes the older `custom_plot` in paper-fig2-waypoint_generation.ipynb,
which used only head YAW for the image orientation, hardcoded the image's
distance and size, and therefore didn't place the image "between" the poses
or make the connecting rays pass cleanly through it.

Usage
-----
  python fig2_waypoint_projection.py --shuffle --num_samples_to_plot 16
  # --use_drawn_goal to render the goal_image with waypoints drawn (if goal_type=="draw")
  # --target_tracks track1-curr_time track2-curr_time ...
"""
import argparse
import os
import random
import sys
from datetime import datetime
from io import BytesIO

import matplotlib.colors as mcolors
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageColor, ImageDraw
from scipy.spatial.transform import Rotation as R_scipy
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.utils import save_image

sys.path.append("/home/anw2067/visualnav-transformer/train")

from planning.nymeria_dataset import NymeriaPlanningDataset, build_planning_split
from planning.utils import draw_waypoints
from planning.vis_utils import disable_logging, draw_image_coords, render_smpl_on_image
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import (
    forward_kinematics_wrapper,
    get_action_smpl_torch,
)
from vint_train.visualizing.nymeria_utils import plot_trajs_and_points_full_body


MODEL_DIRECTORY = {
    "draw_mask": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/config.yaml",
        None,
    ),
    "3d_mask": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goal3d5050-waypointMask/config.yaml",
        None,
    ),
}

OUTPUT_ROOT = "/home/anw2067/visualnav-transformer/train/logs/figures/paper/paper_vis/fig2_waypoint_projection"

# Pelvis/Head/R_Hand/L_Hand — match draw_waypoints' color_order exactly
# (PIL named "red"/"green"/"blue"/"yellow") so 3D leaf markers and 2D
# waypoints on the image use identical colors.
_LEAF_LINE_COLORS = {
    "Pelvis": "#FF0000",   # PIL "red"
    "Head":   "#008000",   # PIL "green"
    "R_Hand": "#0000FF",   # PIL "blue"
    "L_Hand": "#FFFF00",   # PIL "yellow"
}


# ---------------------------------------------------------------------------
# Preview-collage helpers
# ---------------------------------------------------------------------------
def _tensor_to_pil(img_tensor):
    arr = (255.0 * img_tensor.detach().cpu().permute(1, 2, 0)).clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def _draw_goal_skeleton_on_obs(curr_obs_img, goal_image_coords, num_segments):
    """curr_obs_img: (3, H, W) [0,1].  goal_image_coords: (N, 2) or (1, N, 2).
    Layered render: vertices (white lines) → joint dots (white) → waypoint
    circles (red/green/blue/yellow at Pelvis/Head/R_Hand/L_Hand) on top."""
    img_pil = _tensor_to_pil(curr_obs_img)
    draw = ImageDraw.Draw(img_pil)
    coords = goal_image_coords if goal_image_coords.ndim == 3 else goal_image_coords[None]
    # Lines + dots, all white.
    draw_image_coords(draw, coords, color=(255, 255, 255),
                      num_segments=num_segments, show_text=False)
    # Waypoint circles on top — colors match draw_waypoints' color_order.
    waypoint_radius = 4
    for color_name, leaf_idx in zip(
        ["red", "green", "blue", "yellow"], XSensConstants.leaf_indices,
    ):
        point = coords[0, leaf_idx]
        if all(point == -1):
            continue
        x, y = float(point[0]), float(point[1])
        c = ImageColor.getrgb(color_name)
        draw.ellipse([x - waypoint_radius, y - waypoint_radius,
                      x + waypoint_radius, y + waypoint_radius], fill=c)
    return img_pil


def _build_preview_collage(panels, plot_pil, gap_px=4):
    """panels: list of PIL images (all same height H). plot_pil resized to H."""
    H = panels[0].height
    plot_w = max(1, int(round(plot_pil.width * H / plot_pil.height)))
    plot_thumb = plot_pil.resize((plot_w, H), Image.LANCZOS).convert("RGB")
    parts = [p.convert("RGB") for p in panels] + [plot_thumb]
    total_w = sum(p.width for p in parts) + gap_px * (len(parts) - 1)
    canvas = Image.new("RGB", (total_w, H), (255, 255, 255))
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0))
        x += p.width + gap_px
    return canvas


# ---------------------------------------------------------------------------
# Core visualization
# ---------------------------------------------------------------------------
def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _prepare_image(img):
    """Normalize to HWC float in [0, 1]."""
    img = _to_np(img)
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(np.float32) / 255.0
    elif img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0, 1).astype(np.float32)


def _compute_matplotlib_view(cur_body, goal_body, head_rpy,
                             azim_offset_deg, elev_deg,
                             shift_toward_current=True):
    """Return (elev, azim) for the matplotlib 3D camera.

    Camera setup:
      - Targets M = midpoint of (current_head, goal_head).
      - Horizontal position starts perpendicular to the H→G ray.
      - Rotated ``azim_offset_deg`` around world Z toward the current head
        (so H is in the foreground, G behind it) when shift_toward_current,
        else toward the goal head.
      - Raised ``elev_deg`` above horizontal (tilts the view down onto M).

    Falls back to head_yaw+π when H≈G (degenerate ray).
    """
    HEAD_IDX = XSensConstants.part_names.index("Head")
    H = cur_body[HEAD_IDX, :2].astype(np.float64)
    G = goal_body[HEAD_IDX, :2].astype(np.float64)

    ray_xy = G - H
    if float(np.linalg.norm(ray_xy)) < 1e-3:
        head_yaw = float(head_rpy[2])
        return elev_deg, np.rad2deg(head_yaw + np.pi) + azim_offset_deg

    ray_yaw = float(np.arctan2(ray_xy[1], ray_xy[0]))
    # Deterministic perpendicular side: ray_yaw + 90° (CCW of ray).
    perp_yaw = ray_yaw + np.pi / 2

    # Shifting "along the H→G line" = rotating the camera around world Z
    # while still looking at M. +azim_offset rotates toward H when we started
    # at ray_yaw+π/2; the sign flips for the toward-G direction.
    sign = +1.0 if shift_toward_current else -1.0
    mpl_azim = np.rad2deg(perp_yaw) + sign * azim_offset_deg
    return elev_deg, mpl_azim


def plot_waypoint_projection(
    current_body,
    goal_body,
    current_rpy,
    goal_image,
    R_C_pelvis,
    t_C_pelvis,
    fisheye_params,
    *,
    image_distance_frac=0.5,
    min_image_distance=0.12,
    view_azim_offset_deg=45.0,
    view_elev_deg=15.0,
    view_azim_override=None,
    view_elev_override=None,
    skeleton_size=0.7,
    line_width=1.5,
    line_alpha=0.9,
    show_camera_arrows=True,
    show_goal_joint_markers=True,
    fig_size=(6, 6),
    dpi=300,
    xsens_colors=True,
    orig_image_size=2880,
):
    """Render the 3D waypoint-projection figure.

    Args:
        current_body: (15, 3) — current pose joint xyz in world frame
        goal_body:    (15, 3) — goal pose joint xyz in world frame
        current_rpy:  (15, 3) — current pose per-joint xyz-Euler (radians)
        goal_image:   (3, H, W) or (H, W, 3), [0, 1] or [0, 255]

        image_distance_frac : fraction of the *min positive forward distance*
            from camera to goal leaf joint at which to place the image plane.
            0.5 puts it halfway between camera and the nearest leaf joint.
        image_pad_frac : extra margin factor for the image extent relative to
            the projected-joint extent (>=1; 1 just fits all joints inside).
        min_image_distance : safety floor for image plane distance.
        view_azim_offset_deg : offset (deg) between view direction and
            scene-axis perpendicular, rotating the view toward camera-forward.
            0 = fully perpendicular to scene axis (both skeletons visible but
            image edge-on). 90 = straight down the camera forward (image
            face-on but skeletons stacked). Default 40 balances both.
        view_elev_deg : elevation above horizontal for matplotlib view.
        view_azim_override, view_elev_override : if set, bypass adaptive
            computation and use these matplotlib values directly (useful when
            iterating over a panorama of angles).
        skeleton_size : passed through to `plot_trajs_and_points_full_body`.
        line_width, line_alpha, show_camera_arrows, show_goal_joint_markers,
        fig_size, dpi : aesthetic knobs.
        xsens_colors : if True both skeletons use XSensConstants.color_skeleton
            (gradient); otherwise current=blue, goal=orange.

    Returns:
        PIL.Image
    """
    cur_body = _to_np(current_body).astype(np.float32)
    goal_body = _to_np(goal_body).astype(np.float32)
    cur_rpy = _to_np(current_rpy).astype(np.float32)
    img = _prepare_image(goal_image)
    img_H, img_W = img.shape[:2]
    aspect = img_W / img_H  # pixel aspect ratio W/H

    HEAD_IDX = XSensConstants.part_names.index("Head")
    LEAF_INDICES = list(XSensConstants.leaf_indices)
    LEAF_NAMES = list(XSensConstants.leaf_parts)

    # --- Camera frame (in the pelvis frame that cur_body/goal_body live in) ---
    R_CP = _to_np(R_C_pelvis).astype(np.float32)  # pelvis → camera
    t_CP = _to_np(t_C_pelvis).astype(np.float32)
    R_PC = R_CP.T  # camera-axes-in-pelvis-frame as columns
    cam_pos   = (-R_PC @ t_CP).astype(np.float32)
    cam_right = R_PC[:, 0].astype(np.float32)   # camera +X = right
    cam_down  = R_PC[:, 1].astype(np.float32)   # camera +Y = down
    fwd       = R_PC[:, 2].astype(np.float32)   # camera +Z = forward
    cam_up    = -cam_down
    cam_left  = -cam_right
    head_rpy  = cur_rpy[HEAD_IDX]  # used for view computation only

    # Visual cheat: shift the current skeleton so its head joint overlaps the
    # camera. The XSens head joint and Aria camera are a few cm apart (head
    # anatomy + glasses mount); aligning them just looks cleaner. Doesn't
    # affect the image plane, goal body, or ray geometry.
    cur_body = cur_body + (cam_pos - cur_body[HEAD_IDX])[None, :]

    # --- Decide image plane distance -------------------------------------
    goal_leafs = goal_body[LEAF_INDICES]                     # (4, 3)
    rel = goal_leafs - cam_pos[None, :]                      # (4, 3)
    fwd_dists = rel @ fwd                                    # (4,)
    pos_mask = fwd_dists > 1e-3

    if pos_mask.any():
        ref_dist = float(fwd_dists[pos_mask].min())
    else:
        ref_dist = float(max(np.abs(fwd_dists).max(), 0.4))
    image_d = max(ref_dist * image_distance_frac, min_image_distance)
    image_center = cam_pos + fwd * image_d

    # --- Intersect each leaf-joint ray with the plane (for axis bounds) --
    safe = np.where(np.abs(fwd_dists) > 1e-3, fwd_dists, 1.0)
    t_vals = image_d / safe
    intersections = cam_pos[None, :] + t_vals[:, None] * rel  # (4, 3)

    # --- Figure ----------------------------------------------------------
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    # Disable matplotlib's automatic depth sort so artists render in call
    # order (far skeleton → image plane → near skeleton). Without this, the
    # plot_surface ends up drawn over things matplotlib *thinks* are behind.
    try:
        ax.set_computed_zorder(False)
    except AttributeError:
        ax.computed_zorder = False  # mpl < 3.5 fallback
    ax.grid(True)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)

    # Skeletons ---------------------------------------------------------------
    upper_n = XSensConstants.upper_body_num_parts
    if xsens_colors:
        color_cur = XSensConstants.color_skeleton[:upper_n]
        color_goal = XSensConstants.color_skeleton[:upper_n]
    else:
        cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cur = 255. * np.tile(np.array(mcolors.to_rgb(cc[0])), (upper_n, 1))
        color_goal = 255. * np.tile(np.array(mcolors.to_rgb(cc[1])), (upper_n, 1))

    # Z-order bands. computed_zorder=False above means matplotlib draws in
    # ascending zorder, ignoring depth — so we encode back-to-front here.
    # Far band (1-3), image plane (5), near band (10-12). The user-visible
    # invariant: lines under dots, dots under colored waypoint/leaf markers.
    Z_FAR_LINES, Z_FAR_DOTS, Z_FAR_MARKERS = 1, 2, 3
    Z_IMAGE = 5
    Z_NEAR_LINES, Z_NEAR_DOTS, Z_NEAR_MARKERS = 10, 11, 12

    def _plot_skeleton(body, color_skel, z_lines, z_dots):
        for i, parent in enumerate(XSensConstants.kintree_parents[:upper_n]):
            if parent == -1:
                continue
            c, p = body[i], body[parent]
            ax.plot([c[0], p[0]], [c[1], p[1]], [c[2], p[2]],
                    color=color_skel[i] / 255., linewidth=3, alpha=0.9,
                    zorder=z_lines)
        ax.scatter(body[:, 0], body[:, 1], body[:, 2],
                   c='k', s=8, depthshade=False, zorder=z_dots)

    # Compute mpl view direction now so we can order the draw calls
    # back-to-front: farther skeleton → image plane → closer skeleton.
    if view_azim_override is not None or view_elev_override is not None:
        _elev = view_elev_override if view_elev_override is not None else view_elev_deg
        _azim = view_azim_override if view_azim_override is not None else \
                _compute_matplotlib_view(cur_body, goal_body, head_rpy,
                                         view_azim_offset_deg, view_elev_deg)[1]
    else:
        _elev, _azim = _compute_matplotlib_view(
            cur_body, goal_body, head_rpy, view_azim_offset_deg, view_elev_deg
        )
    _e = np.deg2rad(_elev); _a = np.deg2rad(_azim)
    view_dir = np.array([np.cos(_e)*np.cos(_a), np.cos(_e)*np.sin(_a), np.sin(_e)])
    cur_to_viewer = float(cur_body[HEAD_IDX] @ view_dir)
    goal_to_viewer = float(goal_body[HEAD_IDX] @ view_dir)
    cur_is_closer = cur_to_viewer >= goal_to_viewer

    def _draw_cur_skeleton(z_lines, z_dots, z_markers):
        _plot_skeleton(cur_body, color_cur, z_lines, z_dots)

    def _draw_goal_skeleton(z_lines, z_dots, z_markers):
        _plot_skeleton(goal_body, color_goal, z_lines, z_dots)
        if show_goal_joint_markers:
            for leaf_name, leaf_idx in zip(LEAF_NAMES, LEAF_INDICES):
                P = goal_body[leaf_idx]
                col = _LEAF_LINE_COLORS.get(leaf_name, "#444444")
                ax.scatter(*P, color=col, s=36, depthshade=False, zorder=z_markers)

    # Pass 1: farther skeleton.
    if cur_is_closer:
        _draw_goal_skeleton(Z_FAR_LINES, Z_FAR_DOTS, Z_FAR_MARKERS)
    else:
        _draw_cur_skeleton(Z_FAR_LINES, Z_FAR_DOTS, Z_FAR_MARKERS)

    # Image plane ------------------------------------------------------------
    # For each displayed pixel (u_d, v_d) compute its 3D position in pelvis
    # frame by inverting the (rotate_aria_pixels ∘ pinhole) chain. This
    # guarantees the rendered image's pixel locations coincide with the rays
    # from the camera to the goal joints — i.e., waypoints drawn at
    # goal_image_coords[leaf] line up exactly with the goal joints.
    # goal_image_coords lives in `rotate_aria_pixels` frame (raw fisheye
    # pixels rotated 90° CW + scaled to image_size — see vint_dataset.py
    # and planning/nymeria_dataset.py). Inverse of that chain:
    #   X_cam = d * (v_d - cu_d) / f_d
    #   Y_cam = d * (h_max - u_d - cv_d) / f_d
    # with h_max = (orig-1)*scale, cu_d = cu*scale, cv_d = cv*scale,
    # f_d = f*scale. Verified against render_smpl_on_image's intrinsics.
    f_raw = float(fisheye_params[0]); cu_raw = float(fisheye_params[1]); cv_raw = float(fisheye_params[2])
    scale_p = img_W / orig_image_size
    f_d  = f_raw  * scale_p
    cu_d = cu_raw * scale_p
    cv_d = cv_raw * scale_p
    h_max = (orig_image_size - 1) * scale_p
    i_grid, j_grid = np.meshgrid(np.arange(img_H), np.arange(img_W), indexing='ij')
    v_d = i_grid.astype(np.float32)
    u_d = j_grid.astype(np.float32)
    X_cam = image_d * (v_d - cu_d) / f_d
    Y_cam = image_d * (h_max - u_d - cv_d) / f_d
    Z_cam = np.full_like(X_cam, image_d, dtype=np.float32)
    P_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)            # (H, W, 3)
    # camera frame → pelvis frame: P_pelvis = R_PC @ P_cam + cam_pos
    # broadcasted: P_cam @ R_CP + cam_pos   (since R_PC = R_CP.T)
    P_pelvis = P_cam @ R_CP + cam_pos[None, None, :]
    X = P_pelvis[..., 0]; Y = P_pelvis[..., 1]; Z = P_pelvis[..., 2]
    # plot_surface with facecolors can obscure 3D geometry that matplotlib
    # fails to depth-sort correctly. A small alpha keeps the image content
    # readable while letting skeleton parts that are geometrically *in front*
    # of the plane remain visible if mpl's sort happens to drop them behind.
    img_with_alpha = np.concatenate(
        [img, np.full((img.shape[0], img.shape[1], 1), 0.92, dtype=img.dtype)],
        axis=-1,
    )
    ax.plot_surface(
        X, Y, Z, facecolors=img_with_alpha, rstride=1, cstride=1,
        shade=False, antialiased=False, linewidth=0,
        zorder=Z_IMAGE,
    )

    # Pass 2: closer skeleton (drawn after image plane).
    if cur_is_closer:
        _draw_cur_skeleton(Z_NEAR_LINES, Z_NEAR_DOTS, Z_NEAR_MARKERS)
    else:
        _draw_goal_skeleton(Z_NEAR_LINES, Z_NEAR_DOTS, Z_NEAR_MARKERS)

    # Axis bounds — center on M (midpoint of the two heads) so mpl's view
    # actually aims at M. Use a separate span for each axis so the box is
    # tight around the content, then set the box_aspect to the span ratio
    # so mpl doesn't squash the scene into a cube.
    M = 0.5 * (cur_body[HEAD_IDX] + goal_body[HEAD_IDX])
    corners = np.stack([
        P_pelvis[0, 0], P_pelvis[0, -1], P_pelvis[-1, -1], P_pelvis[-1, 0],
    ], axis=0)
    all_pts = np.concatenate([cur_body, goal_body, corners, intersections], axis=0)
    span_xyz = np.max(np.abs(all_pts - M[None, :]), axis=0) * 1.02  # (3,)
    span_xyz = np.maximum(span_xyz, 0.15)
    ax.set_xlim(M[0] - span_xyz[0], M[0] + span_xyz[0])
    ax.set_ylim(M[1] - span_xyz[1], M[1] + span_xyz[1])
    ax.set_zlim(M[2] - span_xyz[2], M[2] + span_xyz[2])
    try:
        ax.set_box_aspect(tuple(span_xyz))
    except Exception:
        pass  # older matplotlib

    # View: scene-axis-perpendicular by default, rotated toward camera-fwd
    if view_azim_override is not None or view_elev_override is not None:
        ax.view_init(
            elev=view_elev_override if view_elev_override is not None else view_elev_deg,
            azim=view_azim_override if view_azim_override is not None else
                 _compute_matplotlib_view(cur_body, goal_body, head_rpy,
                                          view_azim_offset_deg, view_elev_deg)[1],
        )
    else:
        elev, azim = _compute_matplotlib_view(
            cur_body, goal_body, head_rpy, view_azim_offset_deg, view_elev_deg
        )
        ax.view_init(elev=elev, azim=azim)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0., dpi=dpi)
    buf.seek(0)
    pil = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(args):
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    disable_logging()

    with open(args.nomad_config, "r") as f:
        nomad_config = yaml.safe_load(f)

    data_config = nomad_config["datasets"]["nymeria"]
    image_size_hw = nomad_config["image_size"]
    policy_ctx = nomad_config["context_size"]
    context_size = max(args.peva_context_size - 1, policy_ctx)

    tasks_file = build_planning_split(
        data_folder=data_config["data_folder"],
        traj_names_file=os.path.join(data_config["test"], "traj_names.txt"),
        split_save_path=os.path.join(
            data_config["test"],
            f"planning_split_dist{args.min_dist_cat}-{args.max_dist_cat}"
            f"_ctx{context_size}_thresh{args.min_dist_threshold}"
            f"{'_keepnonvis' if args.keep_nonvisible_goal else ''}.pkl",
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
    ds_goal_type = nomad_config.get("goal_type", None) if args.use_drawn_goal else None
    dataset = NymeriaPlanningDataset(
        tasks_file=tasks_file,
        data_folder=data_config["data_folder"],
        image_size=image_size_hw,
        context_size=context_size,
        goal_type=ds_goal_type,
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        gaussian_normalization_stats_path=data_config.get("gaussian_normalization_stats_path", None),
    )

    if args.target_tracks:
        lookup = {f"{t['track']}-{t['curr_time']}": i for i, t in enumerate(dataset.tasks)}
        indices, missing = [], []
        for k in args.target_tracks:
            if k in lookup:
                indices.append(lookup[k])
            else:
                missing.append(k)
        if missing:
            print(f"WARNING: target tracks not found in split: {missing}")
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank,
                                     shuffle=args.shuffle, seed=seed)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    # Output dir
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = (f"{args.nomad_model}-dist{args.min_dist_cat}-{args.max_dist_cat}"
                f"-thresh{args.min_dist_threshold}"
                f"{'-drawn' if args.use_drawn_goal else ''}")
    if args.tag:
        run_name += f"-{args.tag}"
    if args.world_size > 1:
        run_name += f"-rank:ws-{args.rank}:{args.world_size}"
    log_dir = os.path.join(OUTPUT_ROOT, f"{datetime_str}:{run_name}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"saving figures to {log_dir}")

    camera_data_cache = {}

    count = 0
    for batch in dataloader:
        obs_images        = batch["obs_images"]                  # 1, ctx+1, 3, H, W
        goal_image        = batch["goal_image"]                  # 1, 3, H, W  (either raw or drawn)
        goal_obs          = batch["goal_obs"]                    # 1, 3, H, W
        context_poses     = batch["context_poses"]               # 1, ctx+1, 48
        deltas_gt         = batch["deltas"]                      # 1, T, 48
        first_pose        = batch["first_pose"]                  # 1, 1, 48
        xsens_offsets     = batch["xsens_offsets"]               # 1, 23, 3  (see misc.py)
        goal_image_coords = batch["goal_image_coords"]           # 1, 23, 2

        track       = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index  = batch["goal_index"].item()
        task_name   = f"{track}-{start_index}"  # matches --target_tracks key (track-curr_time)
        print(f"[{count}] {task_name}")

        skel = XsensSkeleton(xsens_offsets[0])

        # FK to get current + goal joint xyz in world frame + euler angles
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts)
        fp_xyz, fp_rpy = forward_kinematics_wrapper(first_pose, skel, return_euler=True)
        gt_xyz, _ = forward_kinematics_wrapper(gt_actions[:, -1:], skel, return_euler=True)
        current_body = fp_xyz[0, 0]
        goal_body_xyz = gt_xyz[0, 0]
        current_rpy = fp_rpy[0, 0]

        # Select source image for the plane
        if args.image_source == "goal_image":
            plane_img = goal_image[0]
        elif args.image_source == "current_obs":
            plane_img = obs_images[0, -1]
        elif args.image_source == "goal_obs":
            plane_img = goal_obs[0]
        elif args.image_source == "current_obs_with_wp":
            leaf_wp = goal_image_coords[:, XSensConstants.leaf_indices]   # 1, 4, 2
            plane_img = draw_waypoints(obs_images[:, -1], leaf_wp)[0]     # 3, H, W
        else:
            raise ValueError(f"unknown image_source: {args.image_source}")

        # Load per-track camera data once. Required — both the 3D plot's
        # image plane and the skinned-mesh collage panel depend on it.
        if track not in camera_data_cache:
            cam_path = os.path.join(args.camera_data_folder, track, "camera_data.pt")
            camera_data_cache[track] = torch.load(cam_path, weights_only=False)
        cam_data = camera_data_cache[track]
        T_mat = cam_data["T_C_pelvis"][start_index]
        fisheye_params = cam_data["fisheye_params"]
        R_C_pelvis = T_mat[:3, :3]
        t_C_pelvis = T_mat[:3, 3]

        _cam_pos_pelvis = (-R_C_pelvis.T.float() @ t_C_pelvis.float()).cpu().numpy()
        _head_pos_pelvis = current_body[XSensConstants.part_names.index("Head")].cpu().numpy() if hasattr(current_body, "cpu") else np.asarray(current_body[XSensConstants.part_names.index("Head")])
        _gap = _cam_pos_pelvis - _head_pos_pelvis
        print(f"  head→cam offset (pelvis frame): {_gap} m, |Δ|={np.linalg.norm(_gap):.4f} m")

        plot_kwargs = dict(
            current_body=current_body,
            goal_body=goal_body_xyz,
            current_rpy=current_rpy,
            goal_image=plane_img,
            R_C_pelvis=R_C_pelvis,
            t_C_pelvis=t_C_pelvis,
            fisheye_params=fisheye_params,
            image_distance_frac=args.image_distance_frac,
            view_azim_offset_deg=args.view_azim_offset,
            view_elev_deg=args.view_elev,
            skeleton_size=args.skeleton_size,
            show_camera_arrows=not args.no_camera_arrows,
            show_goal_joint_markers=not args.no_goal_markers,
            xsens_colors=not args.monochrome_skeletons,
            fig_size=(args.fig_size, args.fig_size),
            dpi=args.dpi,
        )
        pil_fig = plot_waypoint_projection(**plot_kwargs)

        task_dir = os.path.join(log_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        pil_fig.save(os.path.join(task_dir, "waypoint_projection.png"))
        save_image(plane_img.detach().cpu(), os.path.join(task_dir, "image_source.png"))
        save_image(obs_images[0, -1].detach().cpu(), os.path.join(task_dir, "current_obs.png"))
        save_image(goal_obs[0].detach().cpu(), os.path.join(task_dir, "goal_obs.png"))

        # --- Preview collage -------------------------------------------------
        # [curr_obs | goal_obs | curr+goal_skel | curr+skin+skel | curr+wp | 3D]
        upper_n = XSensConstants.upper_body_num_parts
        curr_obs_img = obs_images[0, -1]                                 # (3, H, W)
        leaf_wp_xy = goal_image_coords[:, XSensConstants.leaf_indices]   # 1, 4, 2

        curr_obs_pil = _tensor_to_pil(curr_obs_img)
        goal_obs_pil = _tensor_to_pil(goal_obs[0])
        skeleton_pil = _draw_goal_skeleton_on_obs(
            curr_obs_img, goal_image_coords[0, :upper_n], num_segments=upper_n,
        )
        waypoints_pil = _tensor_to_pil(draw_waypoints(obs_images[:, -1], leaf_wp_xy)[0])

        goal_pose = gt_actions[0, -1:]                               # (1, 48)
        mesh_pil = render_smpl_on_image(
            curr_obs_img, goal_pose,
            R_C_pelvis, t_C_pelvis, fisheye_params,
            image_size=curr_obs_img.shape[-1],
            alpha=args.skin_alpha,
            xsens_offsets=xsens_offsets[0],
            draw_skeleton=True,
            show_text=False,
        )

        # Save the panels not already on disk (current_obs/goal_obs already saved above).
        skeleton_pil.save(os.path.join(task_dir, "current_obs_with_goal_skeleton.png"))
        waypoints_pil.save(os.path.join(task_dir, "current_obs_with_waypoints.png"))
        mesh_pil.save(os.path.join(task_dir, "current_obs_with_goal_skin.png"))

        panels = [curr_obs_pil, goal_obs_pil, skeleton_pil, mesh_pil, waypoints_pil]

        collage = _build_preview_collage(panels, pil_fig)
        collage.save(os.path.join(log_dir, f"{task_name}.png"))

        # Optional multi-view contact sheet at several azimuth offsets so the
        # user can eyeball which angle renders best for each sample.
        if args.multi_view:
            _head_idx = XSensConstants.part_names.index("Head")
            base_elev, base_azim = _compute_matplotlib_view(
                _to_np(current_body), _to_np(goal_body_xyz),
                _to_np(current_rpy)[_head_idx],
                args.view_azim_offset, args.view_elev,
            )
            # delta_list = [-60.0, -30.0, 0.0, 30.0, 60.0]
            delta_list = [x for x in range(-120, 120, 30)]
            pano_kwargs = dict(plot_kwargs)
            pano_kwargs["dpi"] = max(120, args.dpi // 2)  # lighter for sheet
            tiles = []
            for d in delta_list:
                pano_kwargs["view_azim_override"] = base_azim + d
                pano_kwargs["view_elev_override"] = base_elev
                tiles.append(plot_waypoint_projection(**pano_kwargs))
            tile_w, tile_h = tiles[0].size
            sheet = Image.new("RGB", (tile_w * len(tiles), tile_h), (255, 255, 255))
            for i, t in enumerate(tiles):
                sheet.paste(t, (i * tile_w, 0))
            sheet.save(os.path.join(task_dir, "projection_pano.png"))

        count += 1
        if args.num_samples_to_plot > 0 and count >= args.num_samples_to_plot:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomad_model", default="draw_mask", choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", default=None)

    # Dataset knobs (mirror plan_cem.py / gt_waypoint_policy_rollouts.py)
    parser.add_argument("--peva_context_size", type=int, default=7)
    parser.add_argument("--min_dist_cat", type=int, default=8)
    parser.add_argument("--max_dist_cat", type=int, default=8)
    parser.add_argument("--min_dist_threshold", type=float, default=0.6)
    parser.add_argument("--curr_time_stride", type=int, default=1)
    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_to_plot", type=int, default=32)
    parser.add_argument("--target_tracks", type=str, nargs="+", default=None,
                        help="'track-curr_time' keys. When set, runs only these.")

    # Visualization knobs
    parser.add_argument("--image_source", default="current_obs_with_wp",
                        choices=["goal_image", "current_obs", "goal_obs", "current_obs_with_wp"],
                        help="Which image to paste on the plane. "
                             "'current_obs_with_wp' = current obs with GT leaf waypoints drawn.")
    parser.add_argument("--use_drawn_goal", action="store_true",
                        help="When --image_source=goal_image, use the config's goal_type "
                             "(== 'draw' annotates waypoints inside the dataset).")
    parser.add_argument("--image_distance_frac", type=float, default=0.3,
                        help="Fraction of min positive fwd distance to goal leaf joint.")
    parser.add_argument("--view_azim_offset", type=float, default=45.0,
                        help="Degrees to rotate the matplotlib camera around "
                             "world Z, from perpendicular-to-head-ray toward "
                             "the current head (so current is in foreground, "
                             "goal and image behind it).")
    parser.add_argument("--view_elev", type=float, default=25.0,
                        help="Camera elevation above horizontal (tilts the "
                             "view down onto M = midpoint of the two heads).")
    parser.add_argument("--multi_view", action="store_true",
                        help="Also save a horizontal contact sheet with 5 "
                             "azimuth variants per sample (easy angle picking).")
    parser.add_argument("--camera_data_folder", type=str,
                        default="/home/anw2067/scratch/nymeria_visibility_matrix",
                        help="Root directory containing per-track camera_data.pt files. "
                             "Required — the 3D image plane and skin panel both depend on it.")
    parser.add_argument("--skin_alpha", type=float, default=0.9,
                        help="Opacity of the SMPL mesh in the skinned panel.")
    parser.add_argument("--skeleton_size", type=float, default=0.7)
    parser.add_argument("--fig_size", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_camera_arrows", action="store_true")
    parser.add_argument("--no_goal_markers", action="store_true")
    parser.add_argument("--monochrome_skeletons", action="store_true",
                        help="Use two solid colors (blue/orange) instead of XSens gradient.")
    parser.add_argument("--tag", default="", help="Extra suffix for the output directory.")

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    if args.nomad_config is None:
        args.nomad_config = MODEL_DIRECTORY[args.nomad_model][0]

    main(args)
