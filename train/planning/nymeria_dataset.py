"""
NymeriaPlanningDataset: a minimal dataset for CEM planning.

Two-stage workflow:
  1. build_planning_split(...) — scans all tracks once, filters tasks by
     visibility and distance, and saves a pkl of (track, curr_time, goal_time)
     tuples.  Re-run whenever you want a different distance range or split.
  2. NymeriaPlanningDataset(tasks_file, ...) — loads the pre-filtered tasks
     and returns only the fields that plan_cem.py actually needs, with no
     in-loop discards.

Compared with ViNT_Nymeria_Dataset this class:
  - skips all training-specific fields (distance label, action_mask, goal_pos,
    gt_actions_with_initial, transformed images, etc.)
  - does not randomly sample a goal at load time — the goal is fixed per task
  - caches ep_info.pt per trajectory to avoid redundant disk I/O
"""

import os
import pickle
from typing import List, Optional, Tuple

import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torchvision.utils import draw_keypoints

from vint_train.data.data_utils import get_data_path, img_path_to_data, to_local_coords_3d
from vint_train.data.misc import XSensConstants
from vint_train.training.nymeria_training_utils import get_delta_smpl

# ---------------------------------------------------------------------------
# Pose helpers (standalone versions of ViNT_Nymeria_Dataset internals)
# ---------------------------------------------------------------------------

_NUM_SEG = XSensConstants.upper_body_num_parts  # 15
_LEAF_PART_NAMES = ["Pelvis", "Head", "R_Hand", "L_Hand"]
_LEAF_IDX = [XSensConstants.part_names.index(p) for p in _LEAF_PART_NAMES]
_HEAD_IDX = XSensConstants.part_names.index("Head")


def _pose_relpelvis(traj_data: dict, t: int) -> torch.Tensor:
    """
    Pose at time t expressed in the pelvis frame at time t.
    Mirrors ViNT_Nymeria_Dataset._compute_actions_nymeria_smpl_relpelvis
    called with curr_time == goal_time == t.

    Returns:
        pose: (1, 48)  [pelvis_xyz=0, joint_euler_rel_pelvis]
    """
    xyz = traj_data["all_parts"][t : t + 1, :_NUM_SEG, 0, 4:]   # (1, 15, 3)
    quat = traj_data["all_parts"][t : t + 1, :_NUM_SEG, 0, :4]  # (1, 15, 4)

    pelvis_xyz = xyz[:, :1].flatten(0, 1)   # (1, 3)
    pelvis_quat = quat[:, :1].flatten(0, 1) # (1, 4)

    xyz_flat = xyz.flatten(0, 1)    # (15, 3)
    quat_flat = quat.flatten(0, 1)  # (15, 4)

    pelvis_rot = R.from_quat(pelvis_quat, scalar_first=True)
    joints_rot = R.from_quat(quat_flat, scalar_first=True)

    rel_xyz = to_local_coords_3d(xyz_flat, pelvis_xyz, pelvis_quat)   # (15, 3)
    rel_euler = (pelvis_rot.inv() * joints_rot).as_euler("xyz", degrees=False)
    rel_euler = torch.from_numpy(rel_euler).float()  # (15, 3)

    rel_xyz = rel_xyz.unflatten(0, (1, _NUM_SEG))
    rel_euler = rel_euler.unflatten(0, (1, _NUM_SEG))

    root_xyz = rel_xyz[:, 0, :]        # (1, 3)  — always (0,0,0)
    euler_flat = rel_euler.flatten(1)   # (1, 45)
    pose = torch.cat((root_xyz, euler_flat), dim=-1)  # (1, 48)
    return pose


def _actions_smpl(traj_data: dict, curr_time: int, len_traj_pred: int) -> torch.Tensor:
    """
    Relative action sequence starting at curr_time.
    Mirrors ViNT_Nymeria_Dataset._compute_actions_nymeria_smpl (actions part only).

    XYZ positions are expressed relative to the initial pelvis frame.
    Each joint's orientation is expressed relative to its own initial orientation.

    Returns:
        actions: (len_traj_pred, 48)  — frame 0 is all zeros (initial pose)
                 followed by len_traj_pred-1 steps, dropped to len_traj_pred by
                 removing the t=curr_time frame.
    """
    end_index = min(curr_time + len_traj_pred + 1, len(traj_data["all_parts"]))

    xyz = traj_data["all_parts"][curr_time:end_index, :_NUM_SEG, 0, 4:]   # (T, 15, 3)
    quat = traj_data["all_parts"][curr_time:end_index, :_NUM_SEG, 0, :4]  # (T, 15, 4)
    actions_T = xyz.shape[0]

    # Initial pelvis reference (broadcast over all joints and timesteps for XYZ)
    pelvis_xyz_0 = xyz[0, 0:1]   # (1, 3)
    pelvis_quat_0 = quat[0, 0:1]  # (1, 4)

    # Per-joint initial quats, repeated T times for rotation computation
    start_quat_per_joint = quat[0].unsqueeze(0).expand(actions_T, -1, -1).flatten(0, 1)  # (T*15, 4)

    xyz_flat = xyz.flatten(0, 1)   # (T*15, 3)
    quat_flat = quat.flatten(0, 1) # (T*15, 4)

    start_rot = R.from_quat(start_quat_per_joint, scalar_first=True)
    joints_rot = R.from_quat(quat_flat, scalar_first=True)

    # XYZ relative to initial pelvis (broadcasts (1,3) against (T*15, 3))
    rel_xyz = to_local_coords_3d(xyz_flat, pelvis_xyz_0, pelvis_quat_0)
    rel_xyz = rel_xyz.unflatten(0, (actions_T, _NUM_SEG))

    rel_euler = (start_rot.inv() * joints_rot).as_euler("xyz", degrees=False)
    rel_euler = torch.from_numpy(rel_euler).float().unflatten(0, (actions_T, _NUM_SEG))

    root_xyz = rel_xyz[:, 0, :]        # (T, 3)
    euler_flat = rel_euler.flatten(1)   # (T, 45)
    actions = torch.cat((root_xyz, euler_flat), dim=-1)  # (T, 48)
    return actions[1:]  # drop t=curr_time (all zeros), return (len_traj_pred, 48)


def _goal_image_coords(traj_data: dict, curr_time: int, goal_time: int,
                       image_size: int) -> torch.Tensor:
    """
    2D joint projections at goal_time as seen from curr_time's camera.
    Returns (23, 2) tensor; coordinates are -1 if outside window or not visible.
    """
    proj = traj_data["image_projection_matrix"]
    proj_half = proj.shape[1] // 2
    target_idx = (goal_time - curr_time) + proj_half

    if not (0 <= target_idx < proj.shape[1]):
        return torch.full((23, 2), -1, dtype=torch.float32)

    coords = proj[curr_time, target_idx, :].clone()  # (23, 2)
    rotated = torch.empty_like(coords)
    rotated[:, 0] = 2880 - 1 - coords[:, 1]
    rotated[:, 1] = coords[:, 0]
    rotated = rotated / 2880 * image_size
    rotated[coords == -1] = -1
    return rotated


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_planning_split(
    data_folder: str,
    traj_names_file: str,
    split_save_path: str,
    context_size: int,
    min_dist_cat: int,
    max_dist_cat: int,
    goal_offsets: Optional[List[int]] = None,
    min_dist_threshold: float = 0.1,
    keep_nonvisible_goal: bool = False,
    waypoint_spacing: int = 1,
    end_slack: int = 0,
    curr_time_stride: int = 1,
    overwrite: bool = False,
) -> str:
    """
    Scan all tracks and save a list of valid planning tasks.

    Each task is a dict: {"track": str, "curr_time": int, "goal_time": int}.
    Filters out tasks where:
      - none of [Pelvis, Head, R_Hand, L_Hand] are visible in the goal image
        (unless keep_nonvisible_goal=True)
      - weighted mean distance of visible+head leaf joints < min_dist_threshold

    Args:
        data_folder: root directory with per-trajectory sub-folders
        traj_names_file: path to .txt file listing trajectory names (one per line)
        split_save_path: where to write the resulting pkl
        context_size: number of context frames required before curr_time
        min_dist_cat: minimum goal offset in frames
        max_dist_cat: maximum goal offset in frames (must be <= proj_half=32)
        goal_offsets: explicit list of per-curr_time goal offsets to try; if
                      None, uses range(min_dist_cat, max_dist_cat+1, waypoint_spacing)
        min_dist_threshold: minimum visible+head leaf joint displacement (m)
        keep_nonvisible_goal: if True, include tasks with no visible joints
        waypoint_spacing: frame stride
        end_slack: frames to ignore at trajectory end
        curr_time_stride: stride when iterating curr_time (reduces split size)
        overwrite: if False, load and return existing split without rebuilding

    Returns:
        split_save_path
    """
    if os.path.exists(split_save_path) and not overwrite:
        print(f"[build_planning_split] Loading existing split: {split_save_path}")
        return split_save_path

    with open(traj_names_file, "r") as f:
        traj_names = [l for l in f.read().split("\n") if l]

    if goal_offsets is None:
        goal_offsets = list(range(min_dist_cat, max_dist_cat + 1, waypoint_spacing))

    tasks = []
    for traj_name in traj_names:
        ep_path = os.path.join(data_folder, traj_name, "ep_info.pt")
        traj_data = torch.load(ep_path, weights_only=False)
        traj_len = len(traj_data["all_parts"])
        proj_half = traj_data["image_projection_matrix"].shape[1] // 2

        begin_time = context_size * waypoint_spacing
        end_time = traj_len - end_slack

        for curr_time in range(begin_time, end_time, curr_time_stride * waypoint_spacing):
            for offset in goal_offsets:
                goal_time = curr_time + offset * waypoint_spacing
                if goal_time >= traj_len:
                    continue

                # --- visibility check ---
                target_idx = offset * waypoint_spacing + proj_half
                within_window = 0 <= target_idx < traj_data["image_projection_matrix"].shape[1]

                if within_window:
                    coords = traj_data["image_projection_matrix"][curr_time, target_idx, :]  # (23, 2)
                    visible = any(
                        (coords[XSensConstants.part_names.index(p)] != -1).all()
                        for p in _LEAF_PART_NAMES
                    )
                else:
                    visible = False
                    coords = None

                if not keep_nonvisible_goal and not visible:
                    continue

                # --- distance check (world XYZ; invariant to rigid transforms) ---
                curr_xyz = traj_data["all_parts"][curr_time, :_NUM_SEG, 0, 4:]  # (15, 3)
                goal_xyz = traj_data["all_parts"][goal_time, :_NUM_SEG, 0, 4:]  # (15, 3)

                if coords is not None:
                    vis_mask = (coords[:_NUM_SEG] != -1).all(dim=-1)  # (15,)
                else:
                    vis_mask = torch.zeros(_NUM_SEG, dtype=torch.bool)
                vis_mask[_HEAD_IDX] = True  # always include head

                leaf_vis = vis_mask[_LEAF_IDX]          # (4,)
                leaf_curr = curr_xyz[_LEAF_IDX]          # (4, 3)
                leaf_goal = goal_xyz[_LEAF_IDX]          # (4, 3)
                leaf_dists = torch.norm(leaf_curr - leaf_goal, dim=-1)  # (4,)

                if leaf_vis.sum() == 0:
                    continue
                dist_score = (leaf_dists * leaf_vis.float()).sum() / leaf_vis.float().sum()

                if dist_score.item() < min_dist_threshold:
                    continue

                tasks.append({"track": traj_name, "curr_time": curr_time, "goal_time": goal_time})

        del traj_data

    print(f"[build_planning_split] {len(tasks)} tasks from {len(traj_names)} trajectories → {split_save_path}")
    os.makedirs(os.path.dirname(os.path.abspath(split_save_path)), exist_ok=True)
    with open(split_save_path, "wb") as f:
        pickle.dump(tasks, f)
    return split_save_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NymeriaPlanningDataset(Dataset):
    """
    Minimal Nymeria dataset for CEM planning.

    Loads from a pre-filtered tasks file (produced by build_planning_split).
    Returns only the fields consumed by plan_cem.py — no training-specific
    tensors, no random goal sampling, no normalization.

    Returned batch keys:
        obs_images            (context_size+1, 3, H, W)  — context frames + current frame
        goal_image            (3, H, W)  — raw goal frame, or keypoint-drawn
        goal_obs              (3, H, W)  — always the raw goal frame
        gt_frames             (n_steps, 3, H, W)  — GT frames from curr_time+1 to goal_time inclusive
        gt_image_coords_seq   (n_steps, 23, 2)    — 2D joint projections for each GT intermediate pose
        context_poses         (context_size+1, 48)
        deltas                (n_steps, 48)  — GT action deltas from curr_time to goal_time
        first_pose            (1, 48)
        goal_pose             (1, 48)  — goal pose in start-relative pelvis frame
        xsens_offsets         (15, 3)
        goal_image_coords     (23, 2)  — 2D joint projections at goal_time
        dataset_index         int
        dataset_track         str
        start_index           int  — curr_time frame index
        goal_index            int  — goal_time frame index

    n_steps = goal_time - curr_time (fixed per split by min/max_dist_cat).
    """

    def __init__(
        self,
        tasks_file: str,
        data_folder: str,
        image_size: Tuple[int, int],
        context_size: int,
        goal_type: Optional[str] = None,
        waypoint_spacing: int = 1,
        obs_type: str = "png",
    ):
        with open(tasks_file, "rb") as f:
            self.tasks: List[dict] = pickle.load(f)

        self.data_folder = data_folder
        self.image_size = image_size
        self.context_size = context_size
        self.goal_type = goal_type
        self.waypoint_spacing = waypoint_spacing
        self.obs_type = obs_type
        self._traj_cache: dict = {}

    def __len__(self) -> int:
        return len(self.tasks)

    def _get_trajectory(self, name: str) -> dict:
        if name not in self._traj_cache:
            d = torch.load(os.path.join(self.data_folder, name, "ep_info.pt"), weights_only=False)
            d.pop("xsens_xyz", None)
            d.pop("xsens_eulerxyz", None)
            for k in d:
                d[k] = d[k].to(torch.float32)
            self._traj_cache[name] = d
        return self._traj_cache[name]

    def _load_image(self, track: str, t: int) -> torch.Tensor:
        path = get_data_path(self.data_folder, track, t, data_type=self.obs_type)
        return img_path_to_data(path, self.image_size)

    def __getitem__(self, i: int) -> dict:
        task = self.tasks[i]
        track = task["track"]
        curr_time = task["curr_time"]
        goal_time = task["goal_time"]
        n_steps = goal_time - curr_time

        traj_data = self._get_trajectory(track)
        ws = self.waypoint_spacing

        # Context images: context frames + current frame (curr_time is last)
        context_times = list(range(
            curr_time - self.context_size * ws,
            curr_time + 1,
            ws,
        ))
        obs_images = torch.stack([self._load_image(track, t) for t in context_times], dim=0)

        # GT trajectory frames and image coords: curr_time+1 through goal_time inclusive
        gt_times = list(range(curr_time + 1, goal_time + 1))
        gt_frames = torch.stack([self._load_image(track, t) for t in gt_times], dim=0)  # (n_steps, 3, H, W)
        gt_image_coords_seq = torch.stack([
            _goal_image_coords(traj_data, curr_time, t, self.image_size[0])
            for t in gt_times
        ], dim=0)  # (n_steps, 23, 2)

        # Goal frame is the last GT frame
        goal_obs = gt_frames[-1]  # (3, H, W)
        goal_image_coords = gt_image_coords_seq[-1]  # (23, 2)

        # Context poses (each frame expressed in its own pelvis frame)
        context_poses = torch.cat([_pose_relpelvis(traj_data, t) for t in context_times], dim=0)

        # Initial pose
        first_pose = _pose_relpelvis(traj_data, curr_time)  # (1, 48)

        # Action deltas from curr_time to goal_time (in start-relative pelvis frame)
        actions = _actions_smpl(traj_data, curr_time, n_steps)  # (n_steps, 48)
        deltas = get_delta_smpl(actions, num_segments=_NUM_SEG)  # (n_steps, 48)

        # Goal pose: final step in start-relative frame
        goal_pose = actions[-1:].clone()  # (1, 48)

        # Goal image — raw, or keypoints drawn on current obs
        if self.goal_type == "draw":
            goal_image = self._draw_goal(obs_images[-1], goal_image_coords)
        else:
            goal_image = goal_obs

        ret = {
            "dataset_index": i,
            "dataset_track": track,
            "start_index": curr_time,
            "goal_index": goal_time,
            "obs_images": obs_images.float(),
            "goal_image": goal_image.float(),
            "goal_obs": goal_obs.float(),
            "gt_frames": gt_frames.float(),
            "gt_image_coords_seq": gt_image_coords_seq.float(),
            "context_poses": context_poses.float(),
            "deltas": deltas.float(),
            "first_pose": first_pose.float(),
            "goal_pose": goal_pose.float(),
            "goal_image_coords": goal_image_coords.float(),
        }

        if "xsens_offsets" in traj_data:
            ret["xsens_offsets"] = traj_data["xsens_offsets"][:_NUM_SEG].float()

        return ret

    @staticmethod
    def _draw_goal(obs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Draw leaf-joint keypoints on obs image."""
        img = obs.clone()
        for part, color in zip(_LEAF_PART_NAMES, ["red", "green", "blue", "yellow"]):
            idx = XSensConstants.part_names.index(part)
            pt = coords[idx]
            if (pt == -1).all():
                continue
            img = draw_keypoints(img, pt[None, None, :], colors=color, radius=4)
        return img
