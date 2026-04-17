"""Forward-pass prediction with a goal-image conditioned policy on planning tasks.

No CEM, no world model — just load the policy, feed it (obs, goal_obs), and
evaluate the predicted deltas against ground truth.

For each task: draw N samples, average their MJE. Repeat that opt_steps times,
take the min across steps. Report the final average over all tasks.
"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from planning.nymeria_dataset import NymeriaPlanningDataset, build_planning_split
from planning.sampling import policy_sample
from planning.utils import _compute_part_distance_matrices, load_policy
from planning.vis_utils import disable_logging
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rank = args.rank
    world_size = args.world_size
    gpu = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu)
    device = "cuda"

    disable_logging()

    # -- load policy --
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

    # -- dataset --
    data_config = nomad_config["datasets"]["nymeria"]
    tasks_file = build_planning_split(
        data_folder=data_config["data_folder"],
        traj_names_file=os.path.join(data_config["test"], "traj_names.txt"),
        split_save_path=os.path.join(
            data_config["test"],
            f"planning_split_dist{args.min_dist_cat}-{args.max_dist_cat}"
            f"_ctx{context_size}_thresh{args.min_dist_threshold}.pkl",
        ),
        context_size=context_size,
        min_dist_cat=args.min_dist_cat,
        max_dist_cat=args.max_dist_cat,
        min_dist_threshold=args.min_dist_threshold,
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        end_slack=data_config.get("end_slack", 0),
        curr_time_stride=args.curr_time_stride,
    )
    dataset = NymeriaPlanningDataset(
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
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=args.shuffle, seed=seed
    )
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    task_results_all = []   # best all_xyz per task
    task_results_leaf = []  # best leaf_xyz per task
    task_results_int = []   # best intermediate_xyz per task

    count = 0
    for _, batch in enumerate(dataloader):
        obs_images = batch["obs_images"].to(device)          # 1, ctx+1, 3, H, W
        goal_obs = batch["goal_obs"].to(device)              # 1, 3, H, W
        context_poses = batch["context_poses"].to(device)    # 1, ctx+1, 48
        deltas = batch["deltas"].to(device)                  # 1, T, 48
        first_pose = batch["first_pose"].to(device)          # 1, 1, 48
        xsens_offsets = batch["xsens_offsets"].to(device)    # 1, 15, 3

        track = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index = batch["goal_index"].item()
        task_name = f"{track}-s{start_index}-g{goal_index}"

        skel = XsensSkeleton(xsens_offsets[0])

        # ground-truth actions
        gt_actions = get_action_smpl_torch(
            first_pose, deltas, XSensConstants.upper_body_num_parts
        )

        # init baseline
        init_xyz_dist, _, init_leaf_xyz, _ = _compute_part_distance_matrices(
            first_pose[:, -1], gt_actions[:, -1], skel
        )
        all_xyz_init = init_xyz_dist.mean(dim=-1).item()
        leaf_xyz_init = init_leaf_xyz.mean().item()
        int_xyz_init = init_xyz_dist[:, XSensConstants.intermediate_indices].mean(dim=-1).item()

        # -- policy: opt_steps rounds of N samples, take min round --
        N = args.num_eval_samples
        policy_obs = imagenet_norm(
            obs_images[:, -context_size - 1 :].flatten(0, 1)
        ).unflatten(0, (1, context_size + 1))
        goal_img = imagenet_norm(goal_obs)

        # tile inputs to batch N samples in one forward pass
        policy_obs_N = policy_obs.expand(N, -1, -1, -1, -1)
        goal_img_N = goal_img.expand(N, -1, -1, -1)
        ctx_poses_N = context_poses[:, -context_size - 1 :].expand(N, -1, -1)
        first_pose_N = first_pose.expand(N, -1, -1)
        gt_actions_N = gt_actions.expand(N, -1, -1)

        step_all_xyz = []
        step_leaf_xyz = []
        step_int_xyz = []
        for _ in range(args.opt_steps):
            pred_deltas = policy_sample(
                policy, noise_scheduler,
                policy_obs_N, goal_img_N,
                ctx_poses_N,
                pred_horizon, action_dim, device,
            )
            pred_actions = get_action_smpl_torch(
                first_pose_N, pred_deltas, XSensConstants.upper_body_num_parts
            )
            xyz_dist, _, leaf_xyz, _ = _compute_part_distance_matrices(
                pred_actions[:, -1], gt_actions_N[:, -1], skel
            )
            int_xyz = xyz_dist[:, XSensConstants.intermediate_indices].mean(dim=-1)
            step_all_xyz.append(xyz_dist.mean(dim=-1).mean().item())
            step_leaf_xyz.append(leaf_xyz.mean().item())
            step_int_xyz.append(int_xyz.mean().item())

        best_all = min(step_all_xyz)
        best_leaf = min(step_leaf_xyz)
        best_int = min(step_int_xyz)
        task_results_all.append(best_all)
        task_results_leaf.append(best_leaf)
        task_results_int.append(best_int)
        count += 1

        print(f"[{count}] {task_name}  "
              f"all={best_all:.4f} (init {all_xyz_init:.4f})  "
              f"leaf={best_leaf:.4f} (init {leaf_xyz_init:.4f})  "
              f"int={best_int:.4f} (init {int_xyz_init:.4f})")

        if args.num_samples_to_plan > 0 and count >= args.num_samples_to_plan:
            break

    def _mean_sem(xs):
        return np.mean(xs), np.std(xs) / np.sqrt(len(xs))

    avg_all, sem_all = _mean_sem(task_results_all)
    avg_leaf, sem_leaf = _mean_sem(task_results_leaf)
    avg_int, sem_int = _mean_sem(task_results_int)
    print(f"\n=== Result (N={args.num_eval_samples}, o={args.opt_steps}, tasks={count}) ===")
    print(f"  all:  {avg_all:.4f} +/- {sem_all:.4f}")
    print(f"  leaf: {avg_leaf:.4f} +/- {sem_leaf:.4f}")
    print(f"  int:  {avg_int:.4f} +/- {sem_int:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomad_config", type=str, required=True)
    parser.add_argument("--nomad_checkpoint", type=str, required=True)
    parser.add_argument("-o", "--opt_steps", type=int, default=6,
                        help="Number of independent rounds (take min across these)")
    parser.add_argument("-N", "--num_eval_samples", type=int, default=64,
                        help="Number of policy samples averaged per round")
    parser.add_argument("--min_dist_cat", type=int, default=8)
    parser.add_argument("--max_dist_cat", type=int, default=8)
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--curr_time_stride", type=int, default=1)
    parser.add_argument("--num_samples_to_plan", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    main(args)
