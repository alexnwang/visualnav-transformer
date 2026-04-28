"""Policy-only rollouts conditioned on GROUND-TRUTH leaf waypoints.

For each planning task:
  * pull GT leaf (4-joint) image coordinates from `goal_image_coords`
  * draw them onto the current observation to form a waypoint-annotated goal image
  * run the policy N times in parallel and keep the best (min MJE) sample
    MJE = mean per-joint xyz error over upper-body joints at the final timestep
  * render skinned-mesh + skeleton overlays of the best predicted action sequence
    as well as the ground-truth action sequence, for side-by-side comparison

Outputs:
  logs/paper_vis/policy_rollouts/<datetime>:<run_name>/<task_idx_name>/
    context_frames.png          context stack (raw)
    goal_obs.png                GT goal observation
    goal_image_waypoints.png    curr obs with GT waypoints overlaid (policy input)
    pred_rollout_skin.png       2-row: [waypoints | pred skel+skin t=0..T-1 | GT wp]
                                       [curr_obs | GT frames t=0..T-1       | goal_obs]
    pred_rollout_noskin.png     same rows but skeleton-only overlay
    gt_rollout_skin.png         same layout but top row uses GT deltas
    gt_rollout_noskin.png       skeleton-only counterpart
    pred_wm_rollout_skin.png    (if --use_peva) bottom row = PEVA frames
    pred_wm_rollout_noskin.png  (if --use_peva) bottom row = PEVA frames
    wm_frame_t{i}.png           (if --use_peva) per-step PEVA frames
    task_data.pth               task metadata + pred/GT deltas + metrics
"""
import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms
from torchvision.utils import save_image


def _save_stacked_preview(src_paths, dst_path: str, max_height: int = 256):
    imgs = []
    for p in src_paths:
        img = PILImage.open(p).convert("RGB")
        img.thumbnail((10 ** 5, max_height), PILImage.LANCZOS)
        imgs.append(img)
    width = max(i.width for i in imgs)
    total_h = sum(i.height for i in imgs)
    stacked = PILImage.new("RGB", (width, total_h), (255, 255, 255))
    y = 0
    for i in imgs:
        stacked.paste(i, (0, y))
        y += i.height
    stacked.save(dst_path)

sys.path.append("/home/anw2067/visualnav-transformer/train")

from planning.nymeria_dataset import NymeriaPlanningDataset, build_planning_split
from planning.sampling import peva_sample, policy_sample
from planning.utils import (_compute_part_distance_matrices, draw_waypoints,
                            load_peva, load_policy)
from planning.vis_utils import disable_logging
from planning.wrappers import build_skeleton_top_seq
from planning.plotting_fns import save_action_obs_sequence_viz
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

OUTPUT_ROOT = "/home/anw2067/visualnav-transformer/train/logs/paper_vis/policy_rollouts"


def main(args):
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rank, world_size = args.rank, args.world_size
    gpu = rank % max(torch.cuda.device_count(), 1)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disable_logging()

    policy, noise_scheduler, _, nomad_config = load_policy(
        args.nomad_config, args.nomad_checkpoint, device=device
    )
    pred_horizon   = nomad_config["len_traj_pred"]
    action_dim     = nomad_config["input_dims"]
    image_size_hw  = nomad_config["image_size"]
    image_size     = image_size_hw[0]
    policy_ctx     = nomad_config["context_size"]

    # Optionally load PEVA world model for rolling out the best action sequence.
    peva_model = peva_diffusion = peva_vae = peva_stats = peva_config = None
    if args.use_peva:
        peva_model, _, peva_diffusion, peva_vae, peva_stats, peva_config = load_peva(
            args.peva_config, args.peva_checkpoint, device=device,
            inference_context_size=args.peva_context_size,
            diffusion_steps=args.peva_diffusion_steps,
        )

    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # --- dataset: match plan_cem.py (peva_context_size controls loaded ctx) ---
    data_config = nomad_config["datasets"]["nymeria"]
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
    dataset = NymeriaPlanningDataset(
        tasks_file=tasks_file,
        data_folder=data_config["data_folder"],
        image_size=image_size_hw,
        context_size=context_size,
        goal_type=nomad_config.get("goal_type", None),
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
        sampler = None
        dataloader = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False, num_workers=0)
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=args.shuffle, seed=seed)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    # --- output dir ---
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = (f"{args.nomad_model}-N{args.num_eval_samples}"
                f"-dist{args.min_dist_cat}-{args.max_dist_cat}")
    if args.use_peva:
        run_name += f"-peva-ds{args.peva_diffusion_steps}"
    if world_size > 1:
        run_name += f"-rank:ws-{rank}:{world_size}"
    log_dir = os.path.join(OUTPUT_ROOT, f"{datetime_str}:{run_name}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"saving rollouts to {log_dir}")

    camera_data_cache = {}
    count, skipped = 0, 0

    for batch in dataloader:
        if skipped < args.skip_tasks:
            skipped += 1
            continue

        obs_images        = batch["obs_images"].to(device)            # 1, ctx+1, 3, H, W
        goal_image        = batch["goal_image"].to(device)            # 1, 3, H, W
        goal_obs          = batch["goal_obs"].to(device)              # 1, 3, H, W
        context_poses     = batch["context_poses"].to(device)         # 1, ctx+1, 48
        deltas_gt         = batch["deltas"].to(device)                # 1, T, 48
        first_pose        = batch["first_pose"].to(device)            # 1, 1, 48
        xsens_offsets     = batch["xsens_offsets"].to(device)         # 1, 15, 3
        goal_image_coords = batch["goal_image_coords"].to(device)     # 1, 23, 2
        gt_frames         = batch["gt_frames"].to(device)             # 1, T, 3, H, W

        track       = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index  = batch["goal_index"].item()
        task_name   = f"{track}-s{start_index}-g{goal_index}"

        print(f"[{count}] {task_name}")

        task_dir = os.path.join(log_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # --- camera params for skin rendering ---
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

        # --- build GT leaf-waypoint goal image ---
        leaf_wp = goal_image_coords[:, XSensConstants.leaf_indices]                # 1, 4, 2
        goal_image_wp = draw_waypoints(obs_images[:, -1], leaf_wp)                 # 1, 3, H, W

        curr_obs_img = obs_images[0, -1]                                            # 3, H, W
        T_steps = deltas_gt.shape[1]

        def _render(pred_or_gt_deltas, overlay, alpha):
            return build_skeleton_top_seq(
                curr_obs_img, pred_or_gt_deltas, first_pose, xsens_offsets[0],
                fisheye_params, R_C_pelvis, t_C_pelvis, image_size, T_steps,
                overlay=overlay, smpl_alpha=alpha,
                show_text=not args.no_skeleton_text,
            )

        # --- task-invariant renders + saves (don't depend on policy seed) ---
        gt_top_skin   = _render(deltas_gt, 'both',     0.9)
        gt_top_noskin = _render(deltas_gt, 'skeleton', 0.0)

        save_image(obs_images[0], os.path.join(task_dir, "context_frames.png"),
                   nrow=obs_images.shape[1])
        save_image(goal_obs[0],      os.path.join(task_dir, "goal_obs.png"))
        save_image(goal_image[0],    os.path.join(task_dir, "goal_image.png"))
        save_image(goal_image_wp[0], os.path.join(task_dir, "goal_image_waypoints.png"))

        # curr_obs + GT goal-pose mesh (no skeleton) + leaf waypoints
        gt_top_mesh_only = _render(deltas_gt, 'skin', 0.9)
        goal_image_wp_mesh = draw_waypoints(gt_top_mesh_only[-1:], leaf_wp)  # 1, 3, H, W
        save_image(goal_image_wp_mesh[0],
                   os.path.join(task_dir, "goal_image_waypoints_mesh.png"))

        save_action_obs_sequence_viz(
            save_path=os.path.join(task_dir, "gt_rollout_skin.png"),
            goal_image=goal_image_wp[0], curr_obs=curr_obs_img, goal_obs=goal_obs[0],
            top_seq=gt_top_skin, bot_seq=gt_frames[0],
        )
        save_action_obs_sequence_viz(
            save_path=os.path.join(task_dir, "gt_rollout_noskin.png"),
            goal_image=goal_image_wp[0], curr_obs=curr_obs_img, goal_obs=goal_obs[0],
            top_seq=gt_top_noskin, bot_seq=gt_frames[0],
        )

        # --- policy sampling: N samples in parallel, keep top-K by MJE ---
        N = args.num_eval_samples
        K = min(args.top_k, N)
        policy_obs_1 = imagenet_norm(
            obs_images[:, -policy_ctx - 1:].flatten(0, 1)
        ).unflatten(0, (1, policy_ctx + 1))                                         # 1, ctx+1, 3, H, W
        goal_for_policy_1 = imagenet_norm(goal_image_wp)                            # 1, 3, H, W
        ctx_poses_1 = context_poses[:, -policy_ctx - 1:]                            # 1, ctx+1, 48
        policy_obs_N = policy_obs_1.expand(N, -1, -1, -1, -1)
        goal_N       = goal_for_policy_1.expand(N, -1, -1, -1)
        ctx_poses_N  = ctx_poses_1.expand(N, -1, -1)

        pred_deltas = policy_sample(
            policy, noise_scheduler,
            policy_obs_N, goal_N, ctx_poses_N,
            pred_horizon, action_dim, device,
        )                                                                           # N, T, 48

        first_pose_N = first_pose.expand(N, -1, -1)
        gt_actions_N = get_action_smpl_torch(first_pose_N, deltas_gt.expand(N, -1, -1),
                                             XSensConstants.upper_body_num_parts)
        pred_actions = get_action_smpl_torch(first_pose_N, pred_deltas,
                                             XSensConstants.upper_body_num_parts)
        skel_single = XsensSkeleton(xsens_offsets[0])
        xyz_dist, _, leaf_xyz, _ = _compute_part_distance_matrices(
            pred_actions[:, -1], gt_actions_N[:, -1], skel_single
        )
        mje = xyz_dist.mean(dim=-1)                                                 # N,
        topk_idx = torch.topk(mje, k=K, largest=False).indices.tolist()             # ascending MJE

        topk_pred_wm_skin_paths = []
        for rank, idx in enumerate(topk_idx):
            multi = K > 1
            rep_dir = os.path.join(task_dir, f"top{rank:02d}") if multi else task_dir
            os.makedirs(rep_dir, exist_ok=True)

            sample_mje  = float(mje[idx].item())
            sample_leaf = float(leaf_xyz[idx].item())
            sample_deltas = pred_deltas[idx:idx + 1]                                # 1, T, 48

            pred_top_skin   = _render(sample_deltas, 'both',     0.9)
            pred_top_noskin = _render(sample_deltas, 'skeleton', 0.0)

            save_action_obs_sequence_viz(
                save_path=os.path.join(rep_dir, "pred_rollout_skin.png"),
                goal_image=goal_image_wp[0], curr_obs=curr_obs_img, goal_obs=goal_obs[0],
                top_seq=pred_top_skin, bot_seq=gt_frames[0],
            )
            save_action_obs_sequence_viz(
                save_path=os.path.join(rep_dir, "pred_rollout_noskin.png"),
                goal_image=goal_image_wp[0], curr_obs=curr_obs_img, goal_obs=goal_obs[0],
                top_seq=pred_top_noskin, bot_seq=gt_frames[0],
            )

            if args.use_peva:
                peva_ctx_size    = peva_config["context_size"]
                peva_latent_size = image_size // 8
                peva_curr_obs = obs_images[:, -peva_ctx_size:]                      # 1, peva_ctx, 3, H, W
                with torch.no_grad():
                    wm_frames, _ = peva_sample(
                        peva_model, peva_diffusion, peva_vae, peva_stats,
                        peva_curr_obs, sample_deltas,
                        peva_ctx_size, peva_latent_size,
                        image_size, device,
                    )                                                               # 1, T, 3, H, W
                save_action_obs_sequence_viz(
                    save_path=os.path.join(rep_dir, "pred_wm_rollout_skin.png"),
                    goal_image=goal_image_wp[0], curr_obs=curr_obs_img, goal_obs=goal_obs[0],
                    top_seq=pred_top_skin, bot_seq=wm_frames[0],
                )
                save_action_obs_sequence_viz(
                    save_path=os.path.join(rep_dir, "pred_wm_rollout_noskin.png"),
                    goal_image=goal_image_wp[0], curr_obs=curr_obs_img, goal_obs=goal_obs[0],
                    top_seq=pred_top_noskin, bot_seq=wm_frames[0],
                )
                for t in range(wm_frames.shape[1]):
                    save_image(wm_frames[0, t].cpu(),
                               os.path.join(rep_dir, f"wm_frame_t{t:02d}.png"))
                topk_pred_wm_skin_paths.append(os.path.join(rep_dir, "pred_wm_rollout_skin.png"))

            torch.save({
                "track": track, "start_index": start_index, "goal_index": goal_index,
                "rank": rank, "sample_idx": idx,
                "first_pose": first_pose[0].cpu(),
                "context_poses": context_poses[0].cpu(),
                "deltas_gt": deltas_gt[0].cpu(),
                "pred_deltas_topk": sample_deltas[0].cpu(),
                "pred_deltas_all": pred_deltas.cpu(),
                "goal_image_coords": goal_image_coords[0].cpu(),
                "sample_mje": sample_mje,
                "sample_leaf_xyz": sample_leaf,
                "topk_idx": topk_idx,
                "mje_all": mje.cpu(),
            }, os.path.join(rep_dir, "task_data.pth"))

            with open(os.path.join(rep_dir, "metrics.txt"), "w") as f:
                f.write(f"rank:            {rank}\n")
                f.write(f"sample_idx:      {idx}\n")
                f.write(f"sample_mje:      {sample_mje:.4f}\n")
                f.write(f"sample_leaf_xyz: {sample_leaf:.4f}\n")

        # Low-res stacked preview: GT-skin on top, then top-K pred-wm-skin (best→worst).
        if args.use_peva and topk_pred_wm_skin_paths:
            _save_stacked_preview(
                [os.path.join(task_dir, "gt_rollout_skin.png"), *topk_pred_wm_skin_paths],
                os.path.join(log_dir, f"{task_name}__preview.png"),
                max_height=args.preview_max_height,
            )

        count += 1
        if args.num_samples_to_plan > 0 and count >= args.num_samples_to_plan:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomad_model", type=str, default="draw_mask",
                        choices=list(MODEL_DIRECTORY.keys()))
    parser.add_argument("--nomad_config", type=str, default=None)
    parser.add_argument("--nomad_checkpoint", type=str, default=None)

    parser.add_argument("-N", "--num_eval_samples", type=int, default=64,
                        help="Number of policy samples drawn in parallel per task")
    parser.add_argument("-K", "--top_k", type=int, default=1,
                        help="Save the top-K samples (lowest MJE) per task. "
                             "K>1 puts each sample in task_dir/top{NN}/ (rank 00 = best); "
                             "previews stack GT on top followed by all K pred_wm_rollout_skins.")

    # keep the same dataset knobs as plan_cem.py
    parser.add_argument("--peva_context_size", type=int, default=7,
                        help="Controls loaded context length (matches plan_cem.py)")
    parser.add_argument("--min_dist_cat", type=int, default=8)
    parser.add_argument("--max_dist_cat", type=int, default=8)
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--curr_time_stride", type=int, default=1)
    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_samples_to_plan", type=int, default=64)
    parser.add_argument("--skip_tasks", type=int, default=0)
    parser.add_argument("--preview_max_height", type=int, default=256,
                        help="Max height (px) for low-res preview copies saved into log_dir")
    parser.add_argument("--target_tracks", type=str, nargs="+", default=None,
                        help="List of 'track-curr_time' keys. When set, runs only these "
                             "tasks (in order) and ignores shuffle/world_size/skip_tasks/num_samples_to_plan.")

    parser.add_argument("--no_skeleton_text", action="store_true",
                        help="Skip drawing joint-name text labels on skeleton overlays.")

    parser.add_argument("--camera_data_folder", type=str,
                        default="/home/anw2067/scratch/nymeria_visibility_matrix",
                        help="Per-track camera_data.pt root (for skin overlays). "
                             "Set to empty string to skip skin rendering.")

    # PEVA world model (optional — roll out the best action sequence through it)
    parser.add_argument("--use_peva", action="store_true",
                        help="After picking the best MJE action sequence, roll "
                             "it out through the PEVA world model and save frames.")
    parser.add_argument("--peva_config", type=str,
                        default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    parser.add_argument("--peva_checkpoint", type=str,
                        default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    parser.add_argument("--peva_diffusion_steps", type=int, default=250,
                        help="Diffusion steps for PEVA WM rollout (higher = slower, better quality)")

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    if args.nomad_model is not None and args.nomad_config is None:
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    assert args.nomad_config and args.nomad_checkpoint

    main(args)
