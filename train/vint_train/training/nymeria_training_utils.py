import os
from matplotlib import pyplot as plt
import numpy as np

import torch

from vint_train.data.data_utils import to_local_coords_3d

from scipy.spatial.transform import Rotation as R

def normalize_data_smpl_pose(data, stats):
    # nomalize to [0,1]
    ndata = data.clone()
    min = stats['min'][:, :3].to(ndata.device) # only translation
    max = stats['max'][:, :3].to(ndata.device) # only translation
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
    B = actions.shape[0]
    actions_T = actions.shape[1]
    
    pelvis_xyz = actions[:, :, :3] # B, T, 3
    pelvis_rotmat = actions[:, :, 3:6] # B, T, 3
    start_pelvis_rotmat = pelvis_rotmat[:, :-1] # B, T-1, 3
    rel_pelvis_xyz = to_local_coords_3d(pelvis_xyz[:, 1:].flatten(0, 1), pelvis_xyz[:, :-1].flatten(0, 1), start_pelvis_rotmat.flatten(0, 1))
    rel_pelvis_xyz = rel_pelvis_xyz.unflatten(0, (B, actions_T-1)) # B, T-1, 3
        
    start_actions_rot = actions[:, :-1, 3:].unflatten(2, (num_segments, 3)).flatten(0, 2)
    start_actions_rot = R.from_euler('xyz', start_actions_rot, degrees=False)
    next_actions_rot = actions[:, 1:, 3:].unflatten(2, (num_segments, 3)).flatten(0, 2)
    next_actions_rot = R.from_euler('xyz', next_actions_rot, degrees=False)
    
    rel_actions_rot = start_actions_rot.inv() * next_actions_rot
    rel_actions_rot = rel_actions_rot.as_euler('xyz', degrees=False)
    rel_actions_rot = torch.from_numpy(rel_actions_rot).float()
    rel_actions_rot = rel_actions_rot.unflatten(0, (B, actions_T-1, num_segments))
    rel_actions_rot = rel_actions_rot.flatten(2) # B, T-1, num_segments*3
    
    delta_actions = actions.clone()
    delta_actions[:, 1:] = torch.cat((rel_pelvis_xyz, rel_actions_rot), dim=2)
        
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

# def visualize_diffusion_action_distribution_full_body(
#     ema_model: nn.Module, # model
#     noise_scheduler: DDPMScheduler, # diffusion schedule
#     batch_obs_images: torch.Tensor, # context images
#     batch_goal_images: torch.Tensor, # goal images
#     batch_viz_obs_images: torch.Tensor, # viz images
#     batch_viz_goal_images: torch.Tensor, # viz images
#     batch_action_label: torch.Tensor, # label
#     batch_distance_labels: torch.Tensor, # label for distance
#     batch_goal_pos: torch.Tensor, # goal pos label
#     device: torch.device,
#     eval_type: str,
#     project_folder: str,
#     epoch: int,
#     num_images_log: int,
#     num_samples: int = 30,
#     use_wandb: bool = True,
# ):
#     visualize_path = os.path.join(
#         project_folder,
#         "visualize",
#         eval_type,
#         f"epoch{epoch}",
#         "action_sampling_distribution_full_body",
#     )
    
#     if not os.path.isdir(visualize_path):
#         os.makedirs(visualize_path)
        
#     max_batch_size = batch_obs_images.shape[0]
    
#     num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
#     print(f"num_images_log: {num_images_log}")
#     # pull the first num_images_log images
#     batch_obs_images = batch_obs_images[:num_images_log]
#     batch_goal_images = batch_goal_images[:num_images_log]
#     batch_action_label = batch_action_label[:num_images_log]
#     batch_goal_pos = batch_goal_pos[:num_images_log]
    
#     len_traj_pred = batch_action_label.shape[1]
#     action_dim = batch_action_label.shape[2]
    
#     # split into batches
#     batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
#     batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

#     # sample each batch
#     uc_actions_list = []
#     gc_actions_list = []
#     gc_distances_list = []
#     for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
#         model_output_dict = model_output(
#             ema_model,
#             noise_scheduler,
#             obs,
#             goal,
#             len_traj_pred,
#             action_dim,
#             num_samples,
#             device=device,
#         )
#         uc_actions_list.append(model_output_dict['uc_actions'])
#         gc_actions_list.append(model_output_dict['gc_actions'])
#         gc_distances_list.append(model_output_dict['gc_distance'])
        
#     # concatenate
#     uc_actions_list = torch.cat(uc_actions_list, dim=0) # (B, T, 48)
#     gc_actions_list = torch.cat(gc_actions_list, dim=0)
#     gc_distances_list = torch.cat(gc_distances_list, dim=0)
    
#     # split into each sampling
#     uc_actions_list = torch.tensor_split(uc_actions_list, num_images_log, dim=0)
#     gc_actions_list = torch.tensor_split(gc_actions_list, num_images_log, dim=0)
#     gc_distances_list = torch.tensor_split(gc_distances_list, num_images_log, dim=0)
    
#     gc_distances_avg = [torch.mean(dist) for dist in gc_distances_list]
#     gc_distances_std = [torch.std(dist) for dist in gc_distances_list]
    
#     assert len(uc_actions_list) == len(gc_actions_list) == num_images_log, f"{len(uc_actions_list)} != {len(gc_actions_list)} != {num_images_log}"
    
#     np_distance_labels = to_numpy(batch_distance_labels)
#     uc_actions_list = [to_numpy(actions) for actions in uc_actions_list]
#     gc_actions_list = [to_numpy(actions) for actions in gc_actions_list]
#     gc_distances_list = [to_numpy(dist) for dist in gc_distances_list]
    
#     # for each example, plot the actions
#     for i in range(num_images_log):
#         fig, ax = plt.subplots(1, 3 + num_samples, figsize=(6 * (3 + num_samples), 8))
        
#         unconditioned_actions = uc_actions_list[i]
#         goal_conditioned_actions = gc_actions_list[i]
#         gt_actions = to_numpy(batch_action_label[i])
    
# unnormalize = transforms.Normalize(
#     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
# )

# def plot_images_and_actions_fully_body(obs_image, goal_image, gt_actions, pred_deltas, add_first_pose, norm_stats, xsens_skel):
#     viz_obs_image = unnormalize(obs_image[0].detach().cpu())[-1]  # take last img from context length of 4
#     viz_obs_image = viz_obs_image.permute(1, 2, 0).numpy() # H, W, C
#     viz_goal_image = unnormalize(goal_image[0].detach().cpu())
#     viz_goal_image = viz_goal_image.permute(1, 2, 0).numpy() # H, W, C
    
#     deltas = pred_deltas
        

# def plot_images_and_actions_full_body(self, image_plot_dir, traj_id, i, traj, cur_obs_image, cur_goal_image, cur_first_pose, gt_actions, deltas, preds, loss, topk_idx, stats, xsens_skel):
#     plot_f = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}_full_body_traj.gif')
#     viz_idx = topk_idx[0].item()
    
#     viz_obs_image = unnormalize(cur_obs_image[0].detach().cpu())[-1] # take last img
#     viz_obs_image = viz_obs_image.permute(1, 2, 0).numpy()
#     viz_goal_image = unnormalize(cur_goal_image[0].detach().cpu())
#     viz_goal_image = viz_goal_image.permute(1, 2, 0).numpy()
    
#     preds = preds[viz_idx]
#     cur_first_pose = cur_first_pose[:1] # All the same
#     deltas = deltas[viz_idx]
    
#     deltas = deltas.detach().cpu()
#     loss = loss.detach().cpu()
#     loss = round(loss[viz_idx].item(), 3)

#     gt_actions = gt_actions[traj]
#     gt_delta_actions = get_delta_smpl(gt_actions, XSensConstants.upper_body_num_parts)
#     gt_actions = get_action_smpl_torch(cur_first_pose, gt_delta_actions.unsqueeze(0), XSensConstants.upper_body_num_parts, stats)
#     gt_actions = gt_actions[0].detach().cpu()
    
#     pred_actions = get_action_smpl_torch(cur_first_pose, deltas.unsqueeze(0), XSensConstants.upper_body_num_parts, stats)
#     pred_actions = pred_actions[0].detach().cpu()
    
#     full_body_frames = []
#     for j in range(pred_actions.shape[0]):
#         curr_gt_actions = forward_kinematics_wrapper(gt_actions[j:j+1], xsens_skel)[0]
#         curr_pred_actions = forward_kinematics_wrapper(pred_actions[j:j+1], xsens_skel)[0]
#         img = plot_cond_goal_gt_pred(viz_obs_image, viz_goal_image, curr_gt_actions, curr_pred_actions)
        
#         full_body_frame = np.array(img).transpose(2, 0, 1)[:3]
#         full_body_frames.append(full_body_frame)
    
#     full_body_frames = np.stack(full_body_frames)
#     save_gif(full_body_frames, plot_f, fps=4)    
    
