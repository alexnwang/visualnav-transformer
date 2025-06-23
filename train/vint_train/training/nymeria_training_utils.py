from io import BytesIO
from PIL import Image
import os
from typing import Optional
import imageio
from matplotlib import pyplot as plt
import numpy as np

import torch

from vint_train.data.data_utils import to_local_coords_3d

from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.axes3d import Axes3D
from vint_train.data.misc import XSensConstants
from vint_train.visualizing.nymeria_utils import plot_cond_goal_gt_pred, save_gif, unnormalize

def normalize_data_smpl_pose(data, stats):
    # nomalize to [0,1]
    ndata = data.clone()
    min = stats['min'][:, :3] # only translation
    max = stats['max'][:, :3] # only translation
    ndata[:, :3] = (data[:, :3] - min) / (max - min)
    # normalize to [-1, 1]
    ndata[:, :3] = ndata[:, :3] * 2 - 1
    return ndata

def unnormalize_data_smpl_pose(ndata, stats):
    data = ndata.clone()
    data[:, :3] = (data[:, :3] + 1) / 2
    min = stats['min'][:, :3].to(ndata.device) # only translation
    max = stats['max'][:, :3].to(ndata.device) # only translation
    data[:, :3] = data[:, :3] * (max - min) + min
    return data

def combine_actions_smpl_norm(delta_actions, stats, num_segments=15):
    """
    Assumes delta_actions are normalized so first unnormalized delta_actions and then 'Sums' and then normalizes again.
    'Sums' actions along time dimension to be a single action
    delta_actions: (B, T, 48)
    
    Returns:
    actions: (B, 48)
    """
    B, T = delta_actions.shape[0], delta_actions.shape[1]
    delta_actions = delta_actions.flatten(0, 1)
    delta_actions = unnormalize_data_smpl_pose(delta_actions, stats).unflatten(0, (B, T))
    delta_actions = combine_actions_smpl(delta_actions.cpu(), num_segments)
    return delta_actions

def combine_actions_smpl_norm_unnorm(delta_actions, stats, num_segments=15):
    """
    Assumes delta_actions are normalized so first unnormalized delta_actions and then 'Sums' and then normalizes again.
    'Sums' actions along time dimension to be a single action
    delta_actions: (B, T, 48)
    
    Returns:
    actions: (B, 48)
    """
    B, T = delta_actions.shape[0], delta_actions.shape[1]
    delta_actions = delta_actions.flatten(0, 1)
    delta_actions = unnormalize_data_smpl_pose(delta_actions, stats).unflatten(0, (B, T))
    delta_actions = combine_actions_smpl(delta_actions, num_segments)
    delta_actions = normalize_data_smpl_pose(delta_actions, stats)
    
    return delta_actions

def get_delta_smpl(actions, num_segments):
    actions_T = actions.shape[0]
    
    pelvis_xyz = actions[:, :3]
    pelvis_rotmat = actions[:, 3:6]
    start_pelvis_rotmat = pelvis_rotmat[:-1]
    rel_pelvis_xyz = to_local_coords_3d(pelvis_xyz[1:], pelvis_xyz[:-1], start_pelvis_rotmat)
        
    start_actions_rot = actions[:-1, 3:].unflatten(1, (num_segments, 3)).flatten(0, 1)
    start_actions_rot = R.from_euler('xyz', start_actions_rot, degrees=False)
    next_actions_rot = actions[1:, 3:].unflatten(1, (num_segments, 3)).flatten(0, 1)
    next_actions_rot = R.from_euler('xyz', next_actions_rot, degrees=False)
    
    rel_actions_rot = start_actions_rot.inv() * next_actions_rot
    rel_actions_rot = rel_actions_rot.as_euler('xyz', degrees=False)
    rel_actions_rot = torch.from_numpy(rel_actions_rot).float()
    rel_actions_rot = rel_actions_rot.unflatten(0, (actions_T-1, num_segments))
    rel_actions_rot = rel_actions_rot.flatten(1)
    
    delta_actions = actions.clone()
    delta_actions[1:] = torch.cat((rel_pelvis_xyz, rel_actions_rot), dim=1)
        
    return delta_actions

def combine_actions_smpl(delta_actions, num_segments=15):
    """
    'Sums' actions along time dimension to be a single action
    delta_actions: (B, T, 48)
    
    Returns:
    actions: (B, 48)
    """
    B, T = delta_actions.shape[0], delta_actions.shape[1]
    all_delta_rot = delta_actions[:, :, 3:].unflatten(-1, (num_segments, 3))
    all_delta_pelvis_xyz = delta_actions[:, :, :3]

    combined_rot = torch.zeros_like(all_delta_rot[:, 0]) # (B, 15, 3)
    combined_rot = combined_rot.flatten(0, 1)
    combined_rot = R.from_euler('xyz', combined_rot, degrees=False)
    
    combined_pelvis_xyz = torch.zeros_like(all_delta_pelvis_xyz[:, 0])
    
    for i in range(T):
        next_delta_rot = all_delta_rot[:, i] # (B, 15, 3)
        next_delta_rot = next_delta_rot.flatten(0, 1)
        next_delta_rot = R.from_euler('xyz', next_delta_rot, degrees=False)

        next_delta_pelvis_xyz = all_delta_pelvis_xyz[:, i]

        combined_pelvis_rot = combined_rot[::num_segments]
        curr_rel_pelvis_xyz = combined_pelvis_rot.apply(np.array(next_delta_pelvis_xyz))
        curr_rel_pelvis_xyz = torch.from_numpy(curr_rel_pelvis_xyz).float()
        combined_pelvis_xyz = combined_pelvis_xyz + curr_rel_pelvis_xyz
        
        combined_rot = combined_rot * next_delta_rot

    combined_rot = combined_rot.as_euler('xyz', degrees=False)
    combined_rot = torch.from_numpy(combined_rot).float()
    combined_rot = combined_rot.unflatten(0, (B, num_segments))
        
    combined_action = torch.cat((combined_pelvis_xyz, combined_rot.flatten(1)), dim=1)

    return combined_action

def get_action_smpl_torch(first_pose, delta_actions, num_segments):
    """
    Generates the sequence of actions following deltas
    
    first_pose (B, 1, 48): first pose in smpl format relative to pelvis. Assumes not normalized
    deltas (B, T, 48): delta actions. Assumes not normalized
    
    Returns:
    actions (B, T, 48): actions
    """
    device = delta_actions.device
    first_pose = first_pose.cpu()
    delta_actions = delta_actions.cpu()
    
    B, T = delta_actions.shape[0], delta_actions.shape[1]
    all_delta_rot = delta_actions[:, :, 3:].unflatten(-1, (num_segments, 3))
    all_delta_pelvis_xyz = delta_actions[:, :, :3]

    combined_rot = first_pose[:, 0, 3:].unflatten(-1, (num_segments, 3))
    combined_rot = combined_rot.flatten(0, 1)
    combined_rot = R.from_euler('xyz', combined_rot, degrees=False)

    combined_pelvis_xyz = first_pose[:, 0, :3]

    all_actions = []
    
    for i in range(T):
        next_delta_rot = all_delta_rot[:, i] # (B, 15, 3)
        next_delta_rot = next_delta_rot.flatten(0, 1)
        next_delta_rot = R.from_euler('xyz', next_delta_rot, degrees=False)

        next_delta_pelvis_xyz = all_delta_pelvis_xyz[:, i]

        combined_pelvis_rot = combined_rot[::num_segments]
        curr_rel_pelvis_xyz = combined_pelvis_rot.apply(np.array(next_delta_pelvis_xyz))
        curr_rel_pelvis_xyz = torch.from_numpy(curr_rel_pelvis_xyz).float()
        combined_pelvis_xyz = combined_pelvis_xyz + curr_rel_pelvis_xyz
        
        combined_rot = combined_rot * next_delta_rot
        
        curr_combined_rot = combined_rot.as_euler('xyz', degrees=False)
        curr_combined_rot = torch.from_numpy(curr_combined_rot).float()
        curr_combined_rot = curr_combined_rot.unflatten(0, (B, num_segments))
        curr_combined_pelvis_xyz = combined_pelvis_xyz.clone()
        curr_combined_action = torch.cat((curr_combined_pelvis_xyz, curr_combined_rot.flatten(1)), dim=1)
        curr_combined_action = curr_combined_action.unsqueeze(1)
        all_actions.append(curr_combined_action)

    all_actions = torch.cat(all_actions, dim=1)

    return all_actions.to(device)

def euler_to_rotmat(euler):
    # Assume (B, 3) -> (B, 3, 3)
    euler_rot = R.from_euler('xyz', euler, degrees=False)
    rotmat = euler_rot.as_matrix()
    rotmat = torch.from_numpy(rotmat).float()
    return rotmat

def forward_kinematics_wrapper(abs_smpl_pose, xsens_skel, num_segments=15):
    B = abs_smpl_pose.shape[0]
    root_xyz = abs_smpl_pose[:, :3]

    smpl_pose_euler = abs_smpl_pose[:, 3:].unflatten(1, (num_segments, 3))
    smpl_pose_rotmat = euler_to_rotmat(smpl_pose_euler.flatten(0, 1))
    smpl_pose_rotmat = smpl_pose_rotmat.unflatten(0, (B, num_segments))

    body_translation = xsens_skel.forward_kinematics(root_xyz, smpl_pose_rotmat, to_smpl=False, to_mvnx=False)
    body_translation = body_translation[:, :num_segments]

    return body_translation

def plot_images_and_actions_full_body(image_plot_dir, # save location
                                      name, # trajectory id, like index
                                      cur_obs_image, # observation (context) images
                                      cur_goal_image, # goal images
                                      cur_first_pose, # first pose in smpl format relative to pelvis
                                      gt_deltas, # ground truth actions relative to initial pose
                                      deltas, # generated actions from NoMad
                                      xsens_skel, # XSens skeleton for forward kinematics
                                      ):
    plot_f = os.path.join(image_plot_dir, f'{name}_full_body_traj.gif')
    cur_first_pose = cur_first_pose[None]
    
    viz_obs_image = cur_obs_image.detach().cpu().permute(1, 2, 0).numpy()
    viz_goal_image = cur_goal_image.detach().cpu().permute(1, 2, 0).numpy()
    
    deltas = {k: v.detach().cpu() for k, v in deltas.items()}
    pred_actions = {k: get_action_smpl_torch(cur_first_pose, v.unsqueeze(0), XSensConstants.upper_body_num_parts)[0].detach().cpu() for k, v in deltas.items()}
    
    gt_actions = get_action_smpl_torch(cur_first_pose, gt_deltas.unsqueeze(0), XSensConstants.upper_body_num_parts)[0].detach().cpu()
    
    full_body_frames = []
    for j in range(gt_actions.shape[0]):
        curr_gt_actions = forward_kinematics_wrapper(gt_actions[j:j+1], xsens_skel)[0]
        curr_pred_actions = {k: forward_kinematics_wrapper(v[j:j+1], xsens_skel)[0] for k, v in pred_actions.items()}
        img = plot_cond_goal_gt_pred(viz_obs_image, viz_goal_image, gt=curr_gt_actions, **curr_pred_actions)
        
        full_body_frame = np.array(img).transpose(2, 0, 1)[:3]
        full_body_frames.append(full_body_frame)

    full_body_frames = np.stack(full_body_frames)
    save_gif(full_body_frames, plot_f, fps=4)
    return plot_f
    