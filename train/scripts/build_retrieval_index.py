"""
Build a (privileged) retrieval index over the training split for the cheat-KNN
baseline in plan_retrieval.py.

For every valid (track, curr_time) at goal_offset = horizon, we store:
    key_t: (3 + 15*4 = 63,) float32 — pelvis xyz + 15 joint quaternions at curr_time, world frame
    key_g: (63,) float32          — same at goal_time
    action: (H, 48) float32        — PEVA delta sequence (matches plan_cem.py --algo peva)
    track: str
    date:  str (= track[:8])
    curr_time: int

Visibility / motion filters mirror build_planning_split (planning/nymeria_dataset.py:148-264)
so the candidate pool is comparable to the test pool.

Usage:
    python -m scripts.build_retrieval_index \
        --horizon 8 --stride 4 \
        --traj_names_file /home/anw2067/visualnav-transformer/train/data_splits/nymeria/train/traj_names.txt \
        --data_folder /scratch/anw2067/nymeria_visibility_matrix \
        --output /scratch/anw2067/retrieval_index/cheat_h8_stride4.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_TRAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TRAIN_ROOT not in sys.path:
    sys.path.insert(0, _TRAIN_ROOT)

from planning.nymeria_dataset import _actions_smpl, _NUM_SEG, _LEAF_IDX, _HEAD_IDX
from vint_train.data.misc import XSensConstants
from vint_train.training.nymeria_training_utils import get_delta_smpl


def extract_key(traj_data: dict, t: int) -> np.ndarray:
    """World-frame body pose at time t: pelvis xyz (3) + 15 joint quats (60) = 63 dims."""
    pelvis_xyz = traj_data["all_parts"][t, 0, 0, 4:].numpy()         # (3,)
    joint_quats = traj_data["all_parts"][t, :_NUM_SEG, 0, :4].numpy()  # (15, 4) wxyz
    return np.concatenate([pelvis_xyz, joint_quats.flatten()]).astype(np.float32)


def passes_filters(traj_data: dict, curr_time: int, goal_offset: int,
                   min_dist_threshold: float, keep_nonvisible_goal: bool):
    """Mirrors build_planning_split filter logic for one (curr_time, goal_offset)."""
    traj_len = len(traj_data["all_parts"])
    goal_time = curr_time + goal_offset
    if goal_time >= traj_len:
        return False

    proj = traj_data["image_projection_matrix"]
    proj_half = proj.shape[1] // 2
    target_idx = goal_offset + proj_half
    within_window = 0 <= target_idx < proj.shape[1]

    if within_window:
        coords = proj[curr_time, target_idx, :]  # (23, 2)
        leaf_part_names = ["Pelvis", "Head", "R_Hand", "L_Hand"]
        visible = any(
            (coords[XSensConstants.part_names.index(p)] != -1).all()
            for p in leaf_part_names
        )
    else:
        visible = False
        coords = None

    if not keep_nonvisible_goal and not visible:
        return False

    curr_xyz = traj_data["all_parts"][curr_time, :_NUM_SEG, 0, 4:]  # (15, 3)
    goal_xyz = traj_data["all_parts"][goal_time, :_NUM_SEG, 0, 4:]

    if coords is not None:
        vis_mask = (coords[:_NUM_SEG] != -1).all(dim=-1)
    else:
        vis_mask = torch.zeros(_NUM_SEG, dtype=torch.bool)
    vis_mask[_HEAD_IDX] = True

    leaf_vis = vis_mask[_LEAF_IDX]
    leaf_curr = curr_xyz[_LEAF_IDX]
    leaf_goal = goal_xyz[_LEAF_IDX]
    leaf_dists = torch.norm(leaf_curr - leaf_goal, dim=-1)

    if leaf_vis.sum() == 0:
        return False
    dist_score = (leaf_dists * leaf_vis.float()).sum() / leaf_vis.float().sum()
    return dist_score.item() >= min_dist_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, help="goal_offset in frames (e.g. 8)")
    ap.add_argument("--stride", type=int, default=4, help="curr_time stride within each trajectory")
    ap.add_argument("--traj_names_file", type=str,
                    default="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/train/traj_names.txt")
    ap.add_argument("--data_folder", type=str,
                    default="/scratch/anw2067/nymeria_visibility_matrix")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--min_dist_threshold", type=float, default=0.1)
    ap.add_argument("--keep_nonvisible_goal", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Use only first N trajectories (sanity)")
    args = ap.parse_args()

    H = args.horizon
    with open(args.traj_names_file) as f:
        traj_names = [l.strip() for l in f if l.strip()]
    if args.limit is not None:
        traj_names = traj_names[:args.limit]
    print(f"Trajectories: {len(traj_names)}  H={H} stride={args.stride}", flush=True)

    keys_t_list, keys_g_list, actions_list = [], [], []
    tracks_list, dates_list, curr_times_list = [], [], []

    t0 = time.time()
    for ti, traj_name in enumerate(traj_names):
        ep_path = os.path.join(args.data_folder, traj_name, "ep_info.pt")
        if not os.path.isfile(ep_path):
            print(f"  skip (missing): {ep_path}", flush=True)
            continue
        traj_data = torch.load(ep_path, weights_only=False)
        if traj_data["all_parts"].ndim == 3:
            # lite layout — re-introduce the missing axis _actions_smpl expects
            traj_data["all_parts"] = traj_data["all_parts"].unsqueeze(2)
        traj_data["all_parts"] = traj_data["all_parts"].to(torch.float32)
        traj_len = len(traj_data["all_parts"])
        last_curr = traj_len - H - 1
        if last_curr < 0:
            continue

        date = traj_name[:8]
        n_added = 0
        for curr_time in range(0, last_curr + 1, args.stride):
            if not passes_filters(traj_data, curr_time, H,
                                  args.min_dist_threshold, args.keep_nonvisible_goal):
                continue
            key_t = extract_key(traj_data, curr_time)
            key_g = extract_key(traj_data, curr_time + H)
            actions = _actions_smpl(traj_data, curr_time, H)            # (H, 48)
            deltas = get_delta_smpl(actions, num_segments=_NUM_SEG)     # (H, 48)
            keys_t_list.append(key_t)
            keys_g_list.append(key_g)
            actions_list.append(deltas.numpy().astype(np.float32))
            tracks_list.append(traj_name)
            dates_list.append(date)
            curr_times_list.append(curr_time)
            n_added += 1
        del traj_data

        if (ti + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{ti+1}/{len(traj_names)}] +{n_added} this traj, total={len(keys_t_list)} "
                  f"elapsed={elapsed:.0f}s rate={(ti+1)/elapsed:.1f} traj/s", flush=True)

    print(f"Total samples: {len(keys_t_list)}  total_time={time.time()-t0:.1f}s", flush=True)

    out = {
        "keys_t": torch.from_numpy(np.stack(keys_t_list)),                 # (N, 63)
        "keys_g": torch.from_numpy(np.stack(keys_g_list)),                 # (N, 63)
        "actions": torch.from_numpy(np.stack(actions_list)),               # (N, H, 48)
        "tracks": tracks_list,                                             # List[str]
        "dates": dates_list,                                               # List[str]
        "curr_times": torch.tensor(curr_times_list, dtype=torch.int64),    # (N,)
        "horizon": H,
        "stride": args.stride,
        "traj_names_file": args.traj_names_file,
        "data_folder": args.data_folder,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    torch.save(out, tmp)
    os.rename(tmp, args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
