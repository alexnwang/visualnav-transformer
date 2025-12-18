import json
import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.utils import draw_keypoints

from scipy.spatial.transform import Rotation as R

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
    to_local_coords_3d
)
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import forward_kinematics_wrapper, get_action_smpl_torch, get_delta_smpl, normalize_data_smpl_pose, normalize_data_smpl_pose_gaussian, set_gaussian_stats

class ViNT_Nymeria_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        gaussian_normalization_stats_path: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        transform: transforms,
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_goals: bool,
        len_traj_pred: int,
        context_size: int,
        goal_type: Optional[str] = None,
        preserve_pose_up_down: bool = False,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "png",
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            gaussian_normalization_stats_path (string): Path to the Gaussian normalization stats file.
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            image_size (tuple): Size of the image to load.
            transform (transform): Transform to apply to the image.
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            min_action_distance (int): Minimum distance to use for the action_mask
            max_action_distance (int): Maximum distance to use for the action_mask
            negative_goals (bool): Whether to use negative goal times
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            goal_type (str): Type of the goal. Can be "2d" or "point" or "2d5050" or "draw"
            preserve_pose_up_down (bool): Whether to preserve the pose up down orientation
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            obs_type (str): What data type to use for the observation. The only one supported is "image" for now.
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        self.traj_len_key = "all_parts"
        self.goal_type = goal_type
        assert self.goal_type in {None, "2d", "point", "2d5050", "draw"}, "goal_format must be one of 2d, point, or 2d5050"
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.transform = transform
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_goals = negative_goals
        self.len_traj_pred = len_traj_pred

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.preserve_pose_up_down = preserve_pose_up_down

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        # self._build_caches()
        

        self.num_action_params = 48 # xyz
            
        self.ACTION_STATS = {}

        action_stats = all_data_config['action_stats']
        if 'action_stats' in self.data_config:
            action_stats = self.data_config['action_stats']
        for key in action_stats:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)
        
        self._init_nymeria(gaussian_normalization_stats_path=gaussian_normalization_stats_path)

    def _init_nymeria(self, full_body: bool = False, gaussian_normalization_stats_path: str = None):
        if full_body:
            self.num_segments = XSensConstants.num_parts
        else:
            self.num_segments = XSensConstants.upper_body_num_parts
        
        # self._compute_actions = self._compute_actions_nymeria_smpl
        # self.normalize_data = normalize_data_smpl_pose
        # self.get_deltas = get_delta_smpl
        # self._compute_actions_smpl_relpelvis = self._compute_actions_nymeria_smpl_relpelvis
    
        if gaussian_normalization_stats_path is not None:
            f = None
            try:
                f = open(gaussian_normalization_stats_path)
            except FileNotFoundError:
                print("action stats not found, using default values")
            if f is not None:
                stats_json = json.load(f)
                stats_dict = {"mean": stats_json['pelvis_xyz']['mean'], "var": stats_json['pelvis_xyz']['var']}
                
                for part_name in XSensConstants.part_names[:XSensConstants.upper_body_num_parts]:
                    stats_dict["mean"] += stats_json['rpy'][part_name]['mean']
                    stats_dict["var"] += stats_json['rpy'][part_name]['var']
                
                self.ACTION_STATS = {
                    "mean": torch.tensor(stats_dict["mean"], dtype=torch.float32),
                    "var": torch.tensor(stats_dict["var"], dtype=torch.float32)
                }
                
                self.normalize_data = normalize_data_smpl_pose_gaussian
                set_gaussian_stats(self.ACTION_STATS)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        # self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time, data_type=self.obs_type)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            try:
                traj_len = len(traj_data["position"])
            except KeyError:
                if hasattr(self, "traj_len_key"):
                    traj_len = len(traj_data[self.traj_len_key])
                else:
                    raise KeyError(f"Trajectory {traj_name} does not have a 'position' key or 'traj_len_key' attribute set.")

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(self.min_dist_cat * self.waypoint_spacing, max_goal_dist + 1)
        if goal_offset == 0 and self.negative_goals:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time, data_type=self.obs_type)
        return img_path_to_data(image_path, self.image_size)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")
    

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_images = torch.stack([
            self._load_image(f, t) for f, t in context # these are C, H, W tensors
        ], dim=0)
        obs_image_transformed = self.transform(obs_images)

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time) # this is already a C, H, W tensor
        goal_image_transformed = self.transform(goal_image)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data[self.traj_len_key])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions_nymeria_smpl(curr_traj_data, curr_time, goal_time)
        
        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
            
        # Compute context poses
        context_poses = []
        for f, t in context:
            context_poses.append(self._compute_actions_nymeria_smpl_relpelvis(curr_traj_data, t, t, preserve_pose_up_down=self.preserve_pose_up_down)[1])
        context_poses = torch.cat(context_poses, dim=0)
        
        # get deltas from actions and normalize
        deltas_torch = get_delta_smpl(actions_torch, num_segments=self.num_segments)
        
        # normalize goals as well
        goal_pos = torch.as_tensor(goal_pos, dtype=torch.float32) # 1, 48
        
        # load first pose for visualizations
        _, first_pose = self._compute_actions_nymeria_smpl_relpelvis(curr_traj_data, curr_time, curr_time, preserve_pose_up_down=self.preserve_pose_up_down)
        
        # compute goal pose incl initial pose, and xyz
        gt_actions_with_initial = get_action_smpl_torch(first_pose[None], deltas_torch[None], XSensConstants.upper_body_num_parts)[:, -1] # 1, 48
        
        # xsens_skel = XsensSkeleton(offsets=curr_traj_data["xsens_offsets"])
        # goal_xyz_in_pelvis0_frame = forward_kinematics_wrapper(gt_actions_with_initial, xsens_skel, return_euler=False) # 1, 15, 3
        # gt_goal_xyz_in_world_frame = curr_traj_data["all_parts"][goal_time, :15, 0, 4:] # 1, 15, 3
        # inital_pelvis_location = curr_traj_data["all_parts"][curr_time, 0, 0, 4:] # 1, 7
        # print("destination error", (goal_xyz_in_pelvis0_frame-gt_goal_xyz_in_world_frame+ inital_pelvis_location[None]).mean())
        # init_xyz_in_pelvis0_frame = forward_kinematics_wrapper(first_pose[None], xsens_skel, return_euler=False) # 1, 15, 3
        # def_xsens_skel = XsensSkeleton()
        # def_goal_xyz_in_world_frame = forward_kinematics_wrapper(gt_actions_with_initial, def_xsens_skel, return_euler=False) # 1, 15, 3
        # print("default error", (def_goal_xyz_in_world_frame + inital_pelvis_location[None] - curr_traj_data["all_parts"][goal_time, :15, 0, 4:]).mean())
        # print("initial error", (init_xyz_in_pelvis0_frame+inital_pelvis_location[None] - curr_traj_data["all_parts"][curr_time, :15, 0, 4:]).mean())
        # def_init_xyz_in_world_frame = forward_kinematics_wrapper(first_pose[None], def_xsens_skel, return_euler=False) # 1, 15, 3
        # print("default initial error", (def_init_xyz_in_world_frame + inital_pelvis_location[None] - curr_traj_data["all_parts"][curr_time, :15, 0, 4:]).mean())
        
        if self.normalize:
            deltas_torch = self.normalize_data(deltas_torch, self.ACTION_STATS)
            # only deltas should be normalized as it is the output of the model.
            # goal_pos = self.normalize_data(goal_pos, self.ACTION_STATS)
            # first_pose = self.normalize_data(first_pose, self.ACTION_STATS)
            # context_poses = self.normalize_data(context_poses, self.ACTION_STATS)
        
        # compute action mask
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        # always return goal_image_coords
        target_idx = goal_time - curr_time + (curr_traj_data["image_projection_matrix"].shape[1] // 2)
        image_coords = curr_traj_data["image_projection_matrix"][curr_time, target_idx, :] # K, 2
        
        # Swap x and y, and invert the new y axis
        rotated_image_coords = torch.empty_like(image_coords)
        rotated_image_coords[:, 0] = 2880 - 1 - image_coords[:, 1]  # new x
        rotated_image_coords[:, 1] = image_coords[:, 0]  # new y
        # keep the -1 elements
        rotated_image_coords = rotated_image_coords / 2880 * self.image_size[0]
        rotated_image_coords[image_coords == -1] = -1
        image_coords = rotated_image_coords

        ret_dict = {
            "obs_image_transformed": obs_image_transformed.type(torch.float32),
            "goal_image_transformed": goal_image_transformed.type(torch.float32),
            "deltas": deltas_torch.type(torch.float32),
            "context_poses": context_poses.type(torch.float32),
            "distance": torch.as_tensor(distance, dtype=torch.int64),
            "goal_pos": torch.as_tensor(goal_pos, dtype=torch.float32),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.float32),
            "first_pose": torch.as_tensor(first_pose, dtype=torch.float32),
            "gt_actions_with_initial": torch.as_tensor(gt_actions_with_initial, dtype=torch.float32),
            "obs_images": obs_images.type(torch.float32),
            "goal_image": goal_image.type(torch.float32),
            "goal_image_coords": torch.as_tensor(image_coords, dtype=torch.float32),
        }
        
        if "xsens_offsets" in curr_traj_data:
            ret_dict["xsens_offsets"] = torch.as_tensor(curr_traj_data["xsens_offsets"], dtype=torch.float32)
        
        if self.goal_type == "point": # late fusion semi "cheat" model
            if False: #"xsens_offsets" in curr_traj_data:
                xsens_offsets = curr_traj_data["xsens_offsets"]
                xsens_skel = XsensSkeleton(offsets=curr_traj_data["xsens_offsets"])
            else:
                xsens_skel = XsensSkeleton()
                xsens_offsets = xsens_skel.offsets
            goal_xyz = forward_kinematics_wrapper(gt_actions_with_initial, xsens_skel, return_euler=False) # 1, 15, 3
            ret_dict["goal_pose_xyz"] = torch.as_tensor(goal_xyz, dtype=torch.float32)
            ret_dict['xsens_offsets'] = torch.as_tensor(xsens_offsets, dtype=torch.float32)
        elif self.goal_type in ["2d", "2d5050"]:
            ret_dict["goal_image_transformed"] = obs_image_transformed[-1]
            ret_dict["goal_image"] = obs_images[-1]
        elif self.goal_type == "draw":
            untransformed_current_obs = obs_images[-1]
            key_point_image_coords = torch.stack(
                [image_coords[XSensConstants.part_names.index(part_name)] for part_name in ["Pelvis","Head", "R_Hand", "L_Hand"]]
            , dim=0) # 4, 2
            for index, color in zip(range(len(key_point_image_coords)), ["red", "green", "blue", "yellow"]):
                if all(key_point_image_coords[index] == -1): continue
                points = key_point_image_coords[index:index+1, None, :]
                untransformed_current_obs = draw_keypoints(untransformed_current_obs, points, colors=color, radius=4)
            ret_dict["goal_image_transformed"] = self.transform(untransformed_current_obs)
            ret_dict["goal_image"] = untransformed_current_obs
        return ret_dict

    def _get_trajectory(self, trajectory_name):
        traj_data = torch.load(os.path.join(self.data_folder, trajectory_name, 'ep_info.pt'), weights_only=False)
        del traj_data['xsens_xyz']
        del traj_data['xsens_eulerxyz']
        for k, v in traj_data.items():
            traj_data[k] = v.to(torch.float32)
        return traj_data
    
    def _compute_actions_nymeria_smpl(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        goal_time = [min(goal_time, len(traj_data['all_parts']) - 1)]
        
        # absolute xyz and rpy
        actions_xyz = traj_data['all_parts'][start_index:end_index, :self.num_segments, 0, 4:]
        actions_quat = traj_data['all_parts'][start_index:end_index, :self.num_segments, 0, :4]
        goals_xyz = traj_data['all_parts'][goal_time, :self.num_segments, 0, 4:]
        goals_quat = traj_data['all_parts'][goal_time, :self.num_segments, 0, :4]
        
        actions_T = actions_xyz.shape[0] # Could be shorter than self.len_traj_pred
        start_xyz = actions_xyz[0].view(1, self.num_segments, 3) # relative to each segment
        
        start_quat_actions = actions_quat[0].view(1, self.num_segments, 4) # initial joint angles for each segment
        start_quat_actions = start_quat_actions.repeat((actions_T, 1, 1)) # tile so it can be applied to every timestep
        start_quat_goals = actions_quat[0].view(1, self.num_segments, 4) # relative to each segment
        start_quat_goals = start_quat_goals.repeat((len(goal_time), 1, 1))
        
        start_xyz, start_quat_actions, start_quat_goals = start_xyz.flatten(0, 1), start_quat_actions.flatten(0, 1), start_quat_goals.flatten(0, 1)
        actions_xyz, actions_quat, goals_xyz, goals_quat = actions_xyz.flatten(0, 1), actions_quat.flatten(0, 1), goals_xyz.flatten(0, 1), goals_quat.flatten(0, 1)
        
        start_rot_actions, start_rot_goals = R.from_quat(start_quat_actions, scalar_first=True), R.from_quat(start_quat_goals, scalar_first=True)
        actions_rot = R.from_quat(actions_quat, scalar_first=True)
        goals_rot = R.from_quat(goals_quat, scalar_first=True)
        
        rel_actions_xyz = to_local_coords_3d(actions_xyz, start_xyz[:1], start_quat_actions[:1]).unflatten(0, (actions_T, self.num_segments)) # relative to first pelvis for translation
        rel_goals_xyz = to_local_coords_3d(goals_xyz, start_xyz[:1], start_quat_goals[:1]).unflatten(0, (1, self.num_segments)) # relative to first pelvis for translation
        
        rel_actions_eulerxyz = (start_rot_actions.inv() * actions_rot).as_euler('xyz', degrees=False)
        rel_actions_eulerxyz = torch.from_numpy(rel_actions_eulerxyz)
        rel_actions_eulerxyz = rel_actions_eulerxyz.unflatten(0, (actions_T, self.num_segments))
        rel_goals_eulerxyz = (start_rot_goals.inv() * goals_rot).as_euler('xyz', degrees=False)
        rel_goals_eulerxyz = torch.from_numpy(rel_goals_eulerxyz)
        rel_goals_eulerxyz = rel_goals_eulerxyz.unflatten(0, (1, self.num_segments))
        
        actions_root_xyz = rel_actions_xyz[:, 0, :] # take the pelvis/head root xyz
        rel_actions_eulerxyz = rel_actions_eulerxyz.flatten(1)
        start_actions = torch.cat((actions_root_xyz, rel_actions_eulerxyz), dim=-1)
        actions = start_actions[1:]
        
        goal_root_xyz = rel_goals_xyz[:, 0, :]
        rel_goals_eulerxyz = rel_goals_eulerxyz.flatten(1)
        goal = torch.cat((goal_root_xyz, rel_goals_eulerxyz), dim=-1)
        
        return actions, goal
    
    def _compute_actions_nymeria_smpl_relpelvis(self, traj_data, curr_time, goal_time, preserve_pose_up_down: bool=False):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        goal_time = [min(goal_time, len(traj_data['all_parts']) - 1)]
                
        actions_xyz = traj_data['all_parts'][start_index:end_index, :self.num_segments, 0, 4:]
        actions_quat = traj_data['all_parts'][start_index:end_index, :self.num_segments, 0, :4]
        goals_xyz = traj_data['all_parts'][goal_time, :self.num_segments, 0, 4:]
        goals_quat = traj_data['all_parts'][goal_time, :self.num_segments, 0, :4]
        
        actions_T = actions_xyz.shape[0] # Could be shorter than self.len_traj_pred
        
        start_xyz = actions_xyz[0, :1].view(1, 1, 3) # relative to first pelvis
        start_quat = actions_quat[0, :1].view(1, 1, 4) # relative to first pelvis
        
        start_xyz, start_quat = start_xyz.flatten(0, 1), start_quat.flatten(0, 1)
        actions_xyz, actions_quat, goals_xyz, goals_quat = actions_xyz.flatten(0, 1), actions_quat.flatten(0, 1), goals_xyz.flatten(0, 1), goals_quat.flatten(0, 1)
        
        start_rot = R.from_quat(start_quat, scalar_first=True)
        actions_rot = R.from_quat(actions_quat, scalar_first=True)
        goals_rot = R.from_quat(goals_quat, scalar_first=True)
        
        if preserve_pose_up_down: # zero out roll and pitch
            start_euler = start_rot.as_euler('xyz', degrees=False) # (1, 3)
            start_euler[:, 0], start_euler[:, 1] = 0., 0.
            start_rot = R.from_euler('xyz', start_euler, degrees=False)
        
        rel_actions_xyz = to_local_coords_3d(actions_xyz, start_xyz, start_quat).unflatten(0, (actions_T, self.num_segments))
        rel_goals_xyz = to_local_coords_3d(goals_xyz, start_xyz, start_quat).unflatten(0, (1, self.num_segments))
        
        rel_actions_eulerxyz = (start_rot.inv() * actions_rot).as_euler('xyz', degrees=False)
        rel_actions_eulerxyz = torch.from_numpy(rel_actions_eulerxyz)
        rel_actions_eulerxyz = rel_actions_eulerxyz.unflatten(0, (actions_T, self.num_segments))
        rel_goals_eulerxyz = (start_rot.inv() * goals_rot).as_euler('xyz', degrees=False)
        rel_goals_eulerxyz = torch.from_numpy(rel_goals_eulerxyz)
        rel_goals_eulerxyz = rel_goals_eulerxyz.unflatten(0, (1, self.num_segments))
        
        actions_root_xyz = rel_actions_xyz[:, 0, :]
        rel_actions_eulerxyz = rel_actions_eulerxyz.flatten(1)
        start_actions = torch.cat((actions_root_xyz, rel_actions_eulerxyz), dim=-1)
        actions = start_actions[1:]
        
        goal_root_xyz = rel_goals_xyz[:, 0, :]
        rel_goals_eulerxyz = rel_goals_eulerxyz.flatten(1)
        goal = torch.cat((goal_root_xyz, rel_goals_eulerxyz), dim=-1)
        
        return actions, goal
