"""Counterfactual waypoint policy visualization (modern re-do of fig3-42).

Reproduces the fig3-42 notebook's behavior using the legacy planning dataset
(`get_nymeria_dataset`) so both the waypoint bank and the per-task selection
match fig3-42 exactly. Rendering uses the new visualization stack
(`build_skeleton_top_seq`) with skeleton-text labels disabled.

For each planning task we:
  * collect a "waypoint bank" of leaf (4-joint) image coords from the first
    `--num_bank_tasks` batches of the dataset, bucketed by how many leaf
    joints are visible (1..4).
  * apply the notebook's inline visibility + min-distance filter
  * for every (num_visible, wp_idx) pair in the bank:
      - draw the bank's waypoints on the current observation -> counterfactual
        goal image
      - run the policy on it
      - render a skeleton-only overlay sequence on the counterfactual goal
        (skeleton text labels off)
      - save as {num_visible}-{wp_idx}-stacked_actions.png

Outputs:
  logs/figures/paper/paper_vis/counterfactuals/<datetime>:<run_name>/<task_name>/
    context_and_goal.png
    {num_visible}-{wp_idx}-stacked_actions.png  (one per counterfactual)
"""
import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append("/home/anw2067/visualnav-transformer/train")

from planning.sampling import policy_sample
from planning.utils import (_compute_part_distance_matrices, draw_waypoints,
                            get_nymeria_dataset, load_policy)
from planning.vis_utils import disable_logging
from planning.wrappers import build_skeleton_top_seq
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch


MODEL_DIRECTORY = {
    "draw_mask": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth",
    ),
    "3d_mask": (
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goal3d5050-waypointMask/config.yaml",
        "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goal3d5050-waypointMask/ema_9.pth",
    ),
}

OUTPUT_ROOT = "/home/anw2067/visualnav-transformer/train/logs/figures/paper/paper_vis/counterfactuals"


def _episode_passes_filter(goal_image_coords, first_pose, deltas, xsens_offsets,
                           keep_nonvisible_goal, min_dist_threshold):
    """Mirror of the fig3 notebook's inline visibility + min-distance filter."""
    if not keep_nonvisible_goal:
        visible = False
        for part in ["Pelvis", "Head", "R_Hand", "L_Hand"]:
            index = XSensConstants.part_names.index(part)
            if (goal_image_coords[0, index] != -1).all():
                visible = True
                break
        if not visible:
            return False

    skel = XsensSkeleton(xsens_offsets[0])
    gt_actions = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts)
    xyz_dist_matrix, _, _, _ = _compute_part_distance_matrices(
        first_pose[:, -1], gt_actions[:, -1], skel
    )
    visible_plus_head = (goal_image_coords != -1).all(dim=-1)[:, :XSensConstants.upper_body_num_parts]
    visible_plus_head[:, XSensConstants.part_names.index("Head")] = True
    masked = xyz_dist_matrix[:, XSensConstants.leaf_indices] * visible_plus_head[:, XSensConstants.leaf_indices]
    init_visible_plus_head = (masked.sum() / visible_plus_head.sum()).item()
    return init_visible_plus_head >= min_dist_threshold


def main(args):
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rank, world_size = args.rank, args.world_size
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disable_logging()

    policy, noise_scheduler, _, nomad_config = load_policy(
        args.nomad_config, args.nomad_checkpoint, device=device
    )
    pred_horizon  = nomad_config["len_traj_pred"]
    action_dim    = nomad_config["input_dims"]
    image_size_hw = nomad_config["image_size"]
    image_size    = image_size_hw[0]
    policy_ctx    = nomad_config["context_size"]

    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # --- legacy dataset (matches fig3-42 notebook) ---
    context_size = max(args.peva_context_size - 1, nomad_config["context_size"])
    dataset = get_nymeria_dataset(
        nomad_config,
        context_size=context_size,
        goal_timestep_offset=args.goal_timestep_offset,
    )

    def make_loader(viz=False):
        # Viz pass with explicit targets: skip the shuffle-scan and pull samples directly by
        # dataset index (same pattern as plan_cem_viz.py).
        if viz and args.target_tracks:
            key_to_idx = {f"{tr}-{t}": i for i, (tr, t, *_) in enumerate(dataset.index_to_data)}
            indices, missing = [], []
            for k in args.target_tracks:
                if k in key_to_idx:
                    indices.append(key_to_idx[k])
                else:
                    missing.append(k)
            if missing:
                print(f"WARNING: target tracks not found: {missing}")
            print(f"viz pass: pulling {len(indices)} target tracks by dataset index "
                  f"(idxs={indices})")
            return DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=1)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=args.shuffle, seed=seed,
        )
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1)

    # --- waypoint bank: notebook iterates first num_bank_tasks batches ---
    waypoints_bank = {1: [], 2: [], 3: [], 4: []}
    for idx, batch in enumerate(make_loader()):
        if idx >= args.num_bank_tasks:
            break
        wp = batch["goal_image_coords"][:, XSensConstants.leaf_indices].to(device)  # 1, 4, 2
        num_visible = (wp != -1).all(dim=-1).sum().item()
        if num_visible == 0:
            continue
        waypoints_bank[num_visible].append(wp)

    # --- append explicit dataset indices to the bank (last entries) ---
    for ds_idx in (args.bank_extra_indices or []):
        sample = dataset[ds_idx]
        gic = sample["goal_image_coords"]
        if not torch.is_tensor(gic):
            gic = torch.as_tensor(gic)
        wp = gic[XSensConstants.leaf_indices].unsqueeze(0).to(device)  # 1, 4, 2
        num_visible = (wp != -1).all(dim=-1).sum().item()
        if num_visible == 0:
            print(f"WARN: bank_extra_indices ds_idx={ds_idx} has 0 visible leaf joints, skipping")
            continue
        waypoints_bank[num_visible].append(wp)
        track = sample.get("dataset_track")
        track_index = sample.get("dataset_track_index")
        print(f"appended ds_idx={ds_idx} ({track}-{track_index}) to bank[num_visible={num_visible}]")

    # --- manually-added custom waypoints (extracted from a reference figure) ---
    # Order: [Pelvis, Head, R_Hand, L_Hand]; -1 = not visible. Pixel coords in image_size space.
    custom_waypoints_224 = [
        # from /home/anw2067/visualnav-transformer/image.png (hands-on-desk POV)
        [[-1, -1], [-1, -1], [158, 184], [132, 188]],
    ]
    for raw in custom_waypoints_224:
        wp = torch.tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)  # 1, 4, 2
        # rescale if model image_size != 224
        if image_size != 224:
            visible_mask = (wp != -1)
            wp = torch.where(visible_mask, wp * (image_size / 224.0), wp)
        num_visible = (wp != -1).all(dim=-1).sum().item()
        if num_visible == 0:
            continue
        waypoints_bank[num_visible].append(wp)
        print(f"appended custom waypoint {raw} to bank[num_visible={num_visible}]")

    bank_total = sum(len(v) for v in waypoints_bank.values())
    print(f"waypoint bank sizes: { {k: len(v) for k, v in waypoints_bank.items()} } "
          f"(total {bank_total})")

    # --- output dir ---
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f"{args.nomad_model}-seed{seed}"
    if args.goal_timestep_offset is not None:
        run_name += f"-goff{args.goal_timestep_offset}"
    if world_size > 1:
        run_name += f"-rank:ws-{rank}:{world_size}"
    log_dir = os.path.join(OUTPUT_ROOT, f"{datetime_str}:{run_name}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"saving counterfactual rollouts to {log_dir}")

    camera_data_cache = {}
    count = 0

    # --- viz pass: target_tracks → direct index lookup; otherwise same shuffle as bank ---
    for idx, batch in enumerate(make_loader(viz=True)):
        if not args.target_tracks and idx >= args.num_bank_tasks:
            break

        obs_images        = batch["obs_images"].to(device)            # 1, ctx+1, 3, H, W
        goal_image        = batch["goal_image"].to(device)            # 1, 3, H, W
        goal_obs          = batch["goal_obs"].to(device)              # 1, 3, H, W
        context_poses     = batch["context_poses"].to(device)         # 1, ctx+1, 48
        first_pose        = batch["first_pose"].to(device)            # 1, 1, 48
        xsens_offsets     = batch["xsens_offsets"].to(device)         # 1, 15, 3
        goal_image_coords = batch["goal_image_coords"].to(device)     # 1, 23, 2
        deltas            = batch["deltas"].to(device)                # 1, T, 48

        track       = batch["dataset_track"][0]
        track_index = batch["dataset_track_index"].item()
        task_name   = f"{track}-{track_index}"

        # When target_tracks is set, the Subset already restricts to targets — no filter,
        # no skip. Otherwise apply the notebook's inline visibility/min-distance filter.
        if not args.target_tracks:
            if not _episode_passes_filter(goal_image_coords, first_pose, deltas, xsens_offsets,
                                          args.keep_nonvisible_goal, args.min_dist_threshold):
                print(f"skip {task_name} (filter)")
                continue

        print(f"[{count}] {task_name}")
        task_dir = os.path.join(log_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # camera params for skeleton overlay
        if args.camera_data_folder:
            if track not in camera_data_cache:
                cam_path = os.path.join(args.camera_data_folder, track, "camera_data.pt")
                camera_data_cache[track] = torch.load(cam_path, weights_only=False)
            cam_data = camera_data_cache[track]
            T_mat = cam_data["T_C_pelvis"][track_index]
            fisheye_params = cam_data["fisheye_params"]
            R_C_pelvis = T_mat[:3, :3]
            t_C_pelvis = T_mat[:3, 3]
        else:
            fisheye_params, R_C_pelvis, t_C_pelvis = None, None, None

        # context_and_goal.png (mirror of fig3 notebook layout)
        save_img = torch.cat(
            [obs_images, goal_obs[None],
             torch.zeros_like(obs_images[:, :-2]), goal_image[None]],
            dim=1,
        )[0]
        save_image(save_img, os.path.join(task_dir, "context_and_goal.png"),
                   nrow=obs_images.shape[1])

        # --- counterfactual sweep ---
        N = args.num_samples_per_pair
        for num_visible in sorted(waypoints_bank.keys()):
            for wp_idx, wp in enumerate(waypoints_bank[num_visible]):
                counterfactual_goal = draw_waypoints(obs_images[:, -1], wp)         # 1, 3, H, W

                policy_obs = imagenet_norm(
                    obs_images[:, -policy_ctx - 1:].flatten(0, 1)
                ).unflatten(0, (1, policy_ctx + 1))
                goal_for_policy = imagenet_norm(counterfactual_goal)
                ctx_poses = context_poses[:, -policy_ctx - 1:]

                # batch by N to draw N independent diffusion samples in one call
                policy_obs_n      = policy_obs.expand(N, -1, -1, -1, -1).contiguous()
                goal_for_policy_n = goal_for_policy.expand(N, -1, -1, -1).contiguous()
                ctx_poses_n       = ctx_poses.expand(N, -1, -1).contiguous()

                pred_deltas_n = policy_sample(
                    policy, noise_scheduler,
                    policy_obs_n, goal_for_policy_n, ctx_poses_n,
                    pred_horizon, action_dim, device,
                )                                                                    # N, T, 48

                header = draw_waypoints(obs_images[:, -1], wp, radius=6)             # 1, 3, H, W

                for n in range(N):
                    pred_deltas = pred_deltas_n[n:n+1]                              # 1, T, 48

                    top_seq = build_skeleton_top_seq(
                        obs_images[-1, -1], pred_deltas, first_pose, xsens_offsets[0],
                        fisheye_params, R_C_pelvis, t_C_pelvis,
                        image_size, pred_deltas.shape[1],
                        overlay='skeleton', smpl_alpha=0.0,
                        show_text=False,
                    )                                                                # T, 3, H, W
                    top_seq_skin = build_skeleton_top_seq(
                        obs_images[-1, -1], pred_deltas, first_pose, xsens_offsets[0],
                        fisheye_params, R_C_pelvis, t_C_pelvis,
                        image_size, pred_deltas.shape[1],
                        overlay='both', smpl_alpha=0.9,
                        show_text=False,
                    )                                                                # T, 3, H, W

                    stacked = torch.cat([header.cpu(), top_seq.cpu()], dim=0)        # T+1, 3, H, W
                    save_image(
                        stacked,
                        os.path.join(task_dir, f"{num_visible}-{wp_idx}-s{n}-stacked_actions.png"),
                        nrow=stacked.shape[0],
                    )
                    stacked_skin = torch.cat([header.cpu(), top_seq_skin.cpu()], dim=0)  # T+1, 3, H, W
                    save_image(
                        stacked_skin,
                        os.path.join(task_dir, f"{num_visible}-{wp_idx}-s{n}-stacked_actions_skin.png"),
                        nrow=stacked_skin.shape[0],
                    )

        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomad_model", type=str, default="draw_mask",
                        choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true",
                        help="Match fig3-42: pass --shuffle (notebook ran with shuffle=True).")
    parser.add_argument("--num_bank_tasks", type=int, default=51,
                        help="Notebook breaks at idx > 50, so 51 batches are processed.")
    parser.add_argument("--bank_extra_indices", type=int, nargs='+', default=None,
                        help="Extra dataset indices to append to the waypoint bank (after the "
                             "shuffled scan), so they appear as the last entries per bucket.")
    parser.add_argument("--num_samples_per_pair", type=int, default=16,
                        help="Number of independent diffusion samples to draw per (task, waypoint).")
    parser.add_argument("--target_tracks", type=str, nargs='+', default=None,
                        help="If set, only viz these task_names ({track}-{track_index}). "
                             "Bank is unaffected. The viz pass scans the full dataset (no num_bank_tasks cap) "
                             "until all targets are found.")

    # filter knobs (notebook inline filter)
    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--goal_timestep_offset", type=int, default=None)

    # context size: legacy notebook used --peva_context_size 7
    parser.add_argument("--peva_context_size", type=int, default=7)

    parser.add_argument("--camera_data_folder", type=str,
                        default="/home/anw2067/scratch/nymeria_visibility_matrix")

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    if args.nomad_model is not None and args.nomad_config is None:
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    assert args.nomad_config and args.nomad_checkpoint

    main(args)
