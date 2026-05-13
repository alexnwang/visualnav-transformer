"""
Privileged 1-NN retrieval baseline for the Nymeria PEVA planning task.

Given a test task, retrieve the nearest training sample by matching full body
pose (pelvis xyz + 15 joint quats, world frame) at both curr_time and goal_time,
restricted to candidates from the same recording date. Apply the retrieved
PEVA delta sequence on top of the test task's *real* first_pose, run forward
kinematics, and compute MJE (mean per-joint xyz position error in meters)
against the ground-truth trajectory.

No world model, no DreamSIM, no rendering. CPU-only.

Usage:
    python plan_retrieval.py \
        --retrieval_index_path /scratch/anw2067/retrieval_index/cheat_h8_stride4.pt \
        --num_samples_to_plan 64 --shuffle
"""
import argparse
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DistributedSampler, DataLoader, Dataset

from planning.nymeria_dataset import (
    NymeriaPlanningDataset, build_planning_split,
    _actions_smpl, _pose_relpelvis, _NUM_SEG,
)
from planning.utils import _compute_part_distance_matrices, LEAF_INDICES
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import (
    get_delta_smpl, get_action_smpl_torch,
)


# ---------------------------------------------------------------------------
# Pose-only dataset (no image loading) — subclasses NymeriaPlanningDataset to
# reuse its tasks file and trajectory cache without modifying the base class.
# ---------------------------------------------------------------------------

class NymeriaPosesOnly(NymeriaPlanningDataset):
    def __getitem__(self, i: int) -> dict:
        task = self.tasks[i]
        track = task["track"]
        curr_time = task["curr_time"]
        goal_time = task["goal_time"]
        n_steps = goal_time - curr_time

        traj = self._get_trajectory(track)
        actions = _actions_smpl(traj, curr_time, n_steps)        # (n_steps, 48) start-relative
        deltas = get_delta_smpl(actions, num_segments=_NUM_SEG)  # (n_steps, 48)

        return {
            "dataset_index": i,
            "dataset_track": track,
            "start_index": curr_time,
            "goal_index": goal_time,
            "first_pose": _pose_relpelvis(traj, curr_time).float(),    # (1, 48)
            "deltas": deltas.float(),                                    # (n_steps, 48)
            "goal_pose": actions[-1:].clone().float(),                   # (1, 48)
            "pose_world_t": traj["all_parts"][curr_time, :15, 0, :].clone().float(),  # (15, 7)
            "pose_world_g": traj["all_parts"][goal_time, :15, 0, :].clone().float(),
            "xsens_offsets": traj["xsens_offsets"].float(),              # (15, 3)
        }


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _key_from_pose_world(pose_world: torch.Tensor) -> torch.Tensor:
    """pose_world: (15, 7) [quat_w, quat_x, quat_y, quat_z, x, y, z] -> (63,) key."""
    pelvis_xyz = pose_world[0, 4:]
    joint_quats = pose_world[:, :4].reshape(-1)
    return torch.cat([pelvis_xyz, joint_quats], dim=0)


def _angular_distance_quat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.arccos(dot)


def _key_distance(query_key: torch.Tensor, cand_keys: torch.Tensor,
                  pelvis_xyz_weight: float = 1.0) -> torch.Tensor:
    """Absolute-mode distance: pelvis-xyz Euclidean + sum over 15 joints of geodesic ang."""
    q_xyz = query_key[:3]
    q_quats = query_key[3:].reshape(15, 4)
    c_xyz = cand_keys[:, :3]
    c_quats = cand_keys[:, 3:].reshape(-1, 15, 4)
    pos_d = torch.norm(c_xyz - q_xyz.unsqueeze(0), dim=-1)
    ang_d = _angular_distance_quat(c_quats, q_quats.unsqueeze(0))
    return pelvis_xyz_weight * pos_d + ang_d.sum(dim=-1)


# ---------------------------------------------------------------------------
# Body-frame ("relative") keying — frame-invariant, location-invariant.
#   key_t_rel: 15 joint quats expressed *relative to pelvis at curr_time*
#   key_g_rel: 15 joint quats at goal_time, also relative to pelvis at curr_time
#   goal_offset_body: (p_g_xyz - p_t_xyz) rotated into pelvis frame at curr_time
# Distance: sum_j ang(rel_t_q[j], rel_t_c[j]) + sum_j ang(rel_g_q[j], rel_g_c[j])
#           + ||goal_offset_body_q - goal_offset_body_c||.
# ---------------------------------------------------------------------------

def _build_relative_keys(keys_t: torch.Tensor, keys_g: torch.Tensor):
    """
    Convert world-frame keys to body-frame keys, batched over candidates.

    keys_t, keys_g: (N, 63) = [pelvis_xyz (3), 15 joint quats wxyz (60)]

    Returns:
      rel_quats_t: (N, 15, 4) wxyz, joint orientations relative to pelvis at curr_time
      rel_quats_g: (N, 15, 4) wxyz, joint orientations at goal_time relative to pelvis at curr_time
      goal_offset_body: (N, 3) goal pelvis displacement in pelvis-at-curr_time frame
    """
    N = keys_t.shape[0]
    pelvis_xyz_t = keys_t[:, :3].numpy()
    pelvis_xyz_g = keys_g[:, :3].numpy()
    joint_quats_t = keys_t[:, 3:].reshape(N, 15, 4).numpy()  # wxyz
    joint_quats_g = keys_g[:, 3:].reshape(N, 15, 4).numpy()

    # pelvis_quat_t: per-candidate pelvis orientation at curr_time — used as the body frame
    pelvis_quat_t = joint_quats_t[:, 0, :]  # (N, 4) wxyz
    pelvis_rot_t = R.from_quat(pelvis_quat_t, scalar_first=True)  # (N,)
    pelvis_rot_t_inv = pelvis_rot_t.inv()

    # Body-frame orientation of each joint, at t and g, relative to pelvis_t
    rel_quats_t = np.zeros_like(joint_quats_t)
    rel_quats_g = np.zeros_like(joint_quats_g)
    for j in range(15):
        Rj_t = R.from_quat(joint_quats_t[:, j, :], scalar_first=True)
        Rj_g = R.from_quat(joint_quats_g[:, j, :], scalar_first=True)
        rel_quats_t[:, j, :] = (pelvis_rot_t_inv * Rj_t).as_quat(scalar_first=True)
        rel_quats_g[:, j, :] = (pelvis_rot_t_inv * Rj_g).as_quat(scalar_first=True)

    # Goal offset in body frame
    offset_world = pelvis_xyz_g - pelvis_xyz_t        # (N, 3)
    goal_offset_body = pelvis_rot_t_inv.apply(offset_world)  # (N, 3)

    return (torch.from_numpy(rel_quats_t).float(),
            torch.from_numpy(rel_quats_g).float(),
            torch.from_numpy(goal_offset_body).float())


def _relative_key_distance(
    q_rel_t: torch.Tensor, q_rel_g: torch.Tensor, q_offset: torch.Tensor,
    c_rel_t: torch.Tensor, c_rel_g: torch.Tensor, c_offset: torch.Tensor,
    pelvis_xyz_weight: float = 1.0,
) -> torch.Tensor:
    """
    Per-candidate relative-mode distance.
    q_rel_t/g: (15, 4) wxyz; q_offset: (3,)
    c_rel_t/g: (N, 15, 4) wxyz; c_offset: (N, 3)
    Returns: (N,)
    """
    ang_t = _angular_distance_quat(c_rel_t, q_rel_t.unsqueeze(0)).sum(dim=-1)  # (N,)
    ang_g = _angular_distance_quat(c_rel_g, q_rel_g.unsqueeze(0)).sum(dim=-1)
    pos_d = torch.norm(c_offset - q_offset.unsqueeze(0), dim=-1)
    return ang_t + ang_g + pelvis_xyz_weight * pos_d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_nomad_config(nomad_config_path: str) -> dict:
    with open(nomad_config_path) as f:
        return yaml.safe_load(f)


def main(args):
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    rank, world_size = args.rank, args.world_size
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    index_stem = os.path.splitext(os.path.basename(args.retrieval_index_path))[0]
    run_name = (f"retrieval-{args.key_mode}{'_anyEnv' if args.ignore_date else '_sameDate'}"
                f"-w{args.pelvis_xyz_weight}-h{args.horizon}-dist{args.min_dist_cat}-{args.max_dist_cat}-idx{index_stem}")
    if world_size > 1:
        run_name = f"{run_name}-rank:ws-{rank}:{world_size}"
    log_dir = f"logs/retrieval/{datetime_str}:{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[retrieval] log_dir={log_dir}", flush=True)

    # --- Build / load planning split (uses paths from nomad config like plan_cem.py) ---
    nomad_config = _load_nomad_config(args.nomad_config)
    data_config = nomad_config["datasets"]["nymeria"]
    if args.data_folder is not None:
        data_config["data_folder"] = args.data_folder
    # Match plan_cem.py:213 exactly so the split filename + length + shuffle order
    # are identical (DistributedSampler shuffle is determined by len(dataset) and seed).
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
    dataset = NymeriaPosesOnly(
        tasks_file=tasks_file,
        data_folder=data_config["data_folder"],
        image_size=(192, 192),  # placeholder; never used (no image loading)
        context_size=context_size,
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=args.shuffle, seed=seed)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    print(f"[retrieval] dataset size={len(dataset)}", flush=True)

    # --- Load retrieval index ---
    print(f"[retrieval] loading index {args.retrieval_index_path}", flush=True)
    idx = torch.load(args.retrieval_index_path, map_location="cpu", weights_only=False)
    assert idx["horizon"] == args.horizon, \
        f"Index horizon={idx['horizon']} != --horizon={args.horizon}"
    keys_t = idx["keys_t"].float()                  # (N, 63) — CPU
    keys_g = idx["keys_g"].float()                  # (N, 63)
    actions_idx = idx["actions"].float()            # (N, H, 48)
    dates = np.array(idx["dates"])                  # (N,) str
    tracks_idx = np.array(idx["tracks"])            # (N,) str
    curr_times_idx = idx["curr_times"].numpy()      # (N,) int64
    print(f"[retrieval] N={keys_t.shape[0]} unique_dates={len(set(dates.tolist()))}", flush=True)
    print(f"[retrieval] key_mode={args.key_mode}  ignore_date={args.ignore_date}", flush=True)

    # Pre-compute body-frame keys if requested (one-time, ~seconds for 300k candidates)
    rel_quats_t_idx = rel_quats_g_idx = goal_offset_body_idx = None
    if args.key_mode == "relative":
        print("[retrieval] precomputing body-frame relative keys for all candidates ...", flush=True)
        t_pre = time.time()
        rel_quats_t_idx, rel_quats_g_idx, goal_offset_body_idx = _build_relative_keys(keys_t, keys_g)
        print(f"[retrieval] precomputed in {time.time()-t_pre:.1f}s "
              f"shapes: rel_t={tuple(rel_quats_t_idx.shape)} offset={tuple(goal_offset_body_idx.shape)}",
              flush=True)

    # --- Per-task results bookkeeping ---
    per_task_records = []  # list of dicts saved at run end

    count, skipped = 0, 0
    for batch in dataloader:
        if skipped < args.skip_tasks:
            skipped += 1; count += 1; continue

        track = batch["dataset_track"][0]
        start_index = batch["start_index"].item()
        goal_index = batch["goal_index"].item()
        first_pose = batch["first_pose"][0]                 # (1, 48)
        gt_deltas = batch["deltas"][0]                       # (H, 48)
        goal_pose = batch["goal_pose"][0]                    # (1, 48)
        xsens_offsets = batch["xsens_offsets"][0]            # (15, 3)
        pose_world_t = batch["pose_world_t"][0]              # (15, 7)
        pose_world_g = batch["pose_world_g"][0]              # (15, 7)
        track_idx_name = f"{track}-s{start_index}-g{goal_index}"
        query_date = track[:8]

        # --- nearest neighbor; key_mode controls absolute vs body-frame ---
        same_self = torch.from_numpy(
            (tracks_idx == track) & (curr_times_idx == start_index)
        )
        if args.ignore_date:
            cand_mask = ~same_self
        else:
            date_mask = torch.from_numpy(dates == query_date)
            if not date_mask.any():
                print(f"  SKIP {track_idx_name}: no same-date candidates", flush=True)
                count += 1
                continue
            cand_mask = date_mask & ~same_self

        if args.key_mode == "absolute":
            query_key_t = _key_from_pose_world(pose_world_t)
            query_key_g = _key_from_pose_world(pose_world_g)
            d_total = (
                _key_distance(query_key_t, keys_t, args.pelvis_xyz_weight)
                + _key_distance(query_key_g, keys_g, args.pelvis_xyz_weight)
            )
        else:  # relative / body-frame
            q_kt = _key_from_pose_world(pose_world_t).unsqueeze(0)  # (1, 63)
            q_kg = _key_from_pose_world(pose_world_g).unsqueeze(0)
            q_rel_t, q_rel_g, q_offset = _build_relative_keys(q_kt, q_kg)
            d_total = _relative_key_distance(
                q_rel_t[0], q_rel_g[0], q_offset[0],
                rel_quats_t_idx, rel_quats_g_idx, goal_offset_body_idx,
                pelvis_xyz_weight=args.pelvis_xyz_weight,
            )
        d_total[~cand_mask] = float("inf")
        k_star = int(torch.argmin(d_total).item())
        retrieved_track = tracks_idx[k_star]
        retrieved_curr = int(curr_times_idx[k_star])
        retrieved_dist = float(d_total[k_star].item())

        retrieved_deltas = actions_idx[k_star]               # (H, 48)

        # --- MJE: forward-kinematics on (pred = retrieved deltas + real first_pose)
        # vs (gt = test task's real action sequence). All CPU, batch=1. ---
        skel = XsensSkeleton(xsens_offsets)
        pred_actions = get_action_smpl_torch(
            first_pose.unsqueeze(0), retrieved_deltas.unsqueeze(0),
            XSensConstants.upper_body_num_parts,
        )[0]  # (H, 48)
        gt_actions = get_action_smpl_torch(
            first_pose.unsqueeze(0), gt_deltas.unsqueeze(0),
            XSensConstants.upper_body_num_parts,
        )[0]  # (H, 48)

        H = pred_actions.shape[0]
        per_step_xyz = torch.zeros(H, _NUM_SEG)             # per-joint xyz dist (m), per timestep
        per_step_ang = torch.zeros(H, _NUM_SEG)             # angular dist (deg), per timestep
        for t in range(H):
            xyz_dist, ang_dist, _, _ = _compute_part_distance_matrices(
                pred_actions[t:t+1], gt_actions[t:t+1], skel,
            )
            per_step_xyz[t] = xyz_dist[0]
            per_step_ang[t] = ang_dist[0]

        # MJE summaries — three flavors per metric:
        #   leaf  = mean over [Pelvis, Head, R_Hand, L_Hand] (4 joints)
        #   int   = mean over the 11 internal joints (everything not in leaf)
        #   all   = mean over all 15 joints
        leaf_idx = LEAF_INDICES
        int_idx = torch.tensor([i for i in range(_NUM_SEG) if i not in leaf_idx.tolist()])

        def _agg(xyz_row: torch.Tensor) -> dict:
            return {
                "leaf": float(xyz_row[leaf_idx].mean().item()),
                "int":  float(xyz_row[int_idx].mean().item()),
                "all":  float(xyz_row.mean().item()),
            }

        # final step: per-joint xyz at t=H-1
        final_per_joint = per_step_xyz[-1]                                  # (15,)
        mje_final = _agg(final_per_joint)
        # avg over time: per-joint xyz averaged over all H timesteps
        avg_per_joint = per_step_xyz.mean(dim=0)                            # (15,)
        mje_avg = _agg(avg_per_joint)
        # init MJE: do-nothing baseline (pred = first_pose at goal time)
        first_xyz_dist, _, _, _ = _compute_part_distance_matrices(
            first_pose, gt_actions[-1:], skel,
        )
        init_per_joint = first_xyz_dist[0]                                  # (15,)
        mje_init = _agg(init_per_joint)

        print(
            f"[{count+1}] {track_idx_name}\n"
            f"    NN: {retrieved_track} curr={retrieved_curr} dist={retrieved_dist:.3f}\n"
            f"    final  leaf={mje_final['leaf']*1000:.1f}mm  int={mje_final['int']*1000:.1f}mm  all={mje_final['all']*1000:.1f}mm\n"
            f"    avg    leaf={mje_avg['leaf']*1000:.1f}mm  int={mje_avg['int']*1000:.1f}mm  all={mje_avg['all']*1000:.1f}mm\n"
            f"    init   leaf={mje_init['leaf']*1000:.1f}mm  int={mje_init['int']*1000:.1f}mm  all={mje_init['all']*1000:.1f}mm",
            flush=True,
        )

        per_task_records.append({
            "task_name": track_idx_name,
            "track": track,
            "start_index": start_index,
            "goal_index": goal_index,
            "retrieved_track": retrieved_track,
            "retrieved_curr_time": retrieved_curr,
            "retrieval_distance": retrieved_dist,
            "mje_final_m": mje_final,                           # dict: leaf/int/all
            "mje_avg_m": mje_avg,                               # dict: leaf/int/all
            "mje_init_m": mje_init,                             # dict: leaf/int/all
            "per_step_xyz_per_joint_m": per_step_xyz.tolist(),  # (H, 15)
            "per_step_ang_per_joint_deg": per_step_ang.tolist(),
        })

        count += 1
        if args.num_samples_to_plan > 0 and count >= args.num_samples_to_plan:
            break

    # --- aggregate ---
    if per_task_records:
        agg = {"n_tasks": len(per_task_records)}
        for which in ("final", "avg", "init"):
            for flavor in ("leaf", "int", "all"):
                vals = np.array([r[f"mje_{which}_m"][flavor] for r in per_task_records])
                agg[f"mje_{which}_{flavor}_mean_m"] = float(vals.mean())
                agg[f"mje_{which}_{flavor}_median_m"] = float(np.median(vals))
        print("=" * 50)
        print(f"[retrieval] aggregate over {agg['n_tasks']} tasks (mm):")
        print(f"  {'metric':<12} {'leaf':>10} {'int':>10} {'all':>10}")
        for which in ("final", "avg", "init"):
            l = agg[f"mje_{which}_leaf_mean_m"] * 1000
            i_ = agg[f"mje_{which}_int_mean_m"] * 1000
            a = agg[f"mje_{which}_all_mean_m"] * 1000
            print(f"  {which+' (mean)':<12} {l:>10.1f} {i_:>10.1f} {a:>10.1f}")
        for which in ("final", "avg", "init"):
            l = agg[f"mje_{which}_leaf_median_m"] * 1000
            i_ = agg[f"mje_{which}_int_median_m"] * 1000
            a = agg[f"mje_{which}_all_median_m"] * 1000
            print(f"  {which+' (med)':<12} {l:>10.1f} {i_:>10.1f} {a:>10.1f}")
    else:
        agg = {"n_tasks": 0}

    out = {"per_task": per_task_records, "aggregate": agg, "args": vars(args)}
    with open(f"{log_dir}/results.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"Wrote {log_dir}/results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_index_path", type=str, required=True,
                        help="Path to .pt index from scripts/build_retrieval_index.py")
    parser.add_argument("-H", "--horizon", type=int, default=8)
    parser.add_argument("--key_mode", type=str, choices=["absolute", "relative"], default="absolute",
                        help="absolute: world-frame pelvis xyz + joint quats (privileged location). "
                             "relative: body-frame joint quats + body-frame goal offset (location-invariant).")
    parser.add_argument("--ignore_date", action="store_true",
                        help="Drop the same-date candidate filter (search across all envs). "
                             "Recommended with --key_mode relative.")
    parser.add_argument("--pelvis_xyz_weight", type=float, default=1.0,
                        help="Weight applied to the pelvis-xyz Euclidean term in the distance "
                             "(absolute: world pelvis xyz; relative: body-frame goal offset). "
                             "Default 1.0 weighs it equally with each summed angular term.")

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_samples_to_plan", type=int, default=64)
    parser.add_argument("--skip_tasks", type=int, default=0)

    parser.add_argument("--min_dist_cat", type=int, default=8)
    parser.add_argument("--max_dist_cat", type=int, default=8)
    parser.add_argument("--min_dist_threshold", type=float, default=0.1)
    parser.add_argument("--keep_nonvisible_goal", action="store_true")
    parser.add_argument("--curr_time_stride", type=int, default=1)
    parser.add_argument("--peva_context_size", type=int, default=7,
                        help="Used to derive context_size (= max(peva_context_size-1, nomad_config_ctx)) "
                             "exactly as in plan_cem.py:213 so the split file + shuffle order match.")

    parser.add_argument("--data_folder", type=str,
                        default="/scratch/anw2067/nymeria_visibility_matrix")
    parser.add_argument("--nomad_config", type=str,
                        default="/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2025_12_09_11_24:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw/config.yaml",
                        help="Source of dataset paths (data_folder, test split). Any nomad config works.")

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    main(args)
