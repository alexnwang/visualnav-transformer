from io import BytesIO
from PIL import Image
import os
from typing import Optional
import imageio
from matplotlib import pyplot as plt
import numpy as np

import torch

from mpl_toolkits.mplot3d.axes3d import Axes3D
from vint_train.data.misc import XSensConstants

def unnormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return image_tensor * std + mean

def plot_cond_goal_gt_pred(obs_image, goal_image, **kwargsactions):
    ncols = 2 + len(kwargsactions)
    fig = plt.figure(figsize=(4*ncols, 6))
    
    axes = []
    for i in range(ncols):
        ax = fig.add_subplot(1, ncols, i + 1, projection='3d' if i >= 2 else None)
        axes.append(ax)
    
    axes[0].imshow(obs_image)
    axes[0].axis('off')
    axes[0].set_title(f"Observation")
    axes[1].imshow(goal_image)
    axes[1].axis('off')
    axes[1].set_title(f"Goal")
    
    for i, (key, actions) in enumerate(kwargsactions.items()):
        ax = axes[i + 2]
        plot_trajs_and_points_full_body(
            ax,
            actions,
            None,
            XSensConstants.kintree_parents[:XSensConstants.upper_body_num_parts],
            XSensConstants.color_skeleton[:XSensConstants.upper_body_num_parts],
        )
        ax.set_title(f"{key} Action")
    
    # make the plot large
    # fig.set_size_inches(17, 6)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.15)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def plot_trajs_and_points_full_body(
    ax: Axes3D,
    actions: np.array,
    pose_rotmat: Optional[np.ndarray] = None,
    kintree_parents: Optional[list] = None,
    color_skeleton: Optional[list] = None,
    size: int = 2,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        actions: (23, 3) # joints
    """
    if kintree_parents is None:
        kintree_parents = XSensConstants.kintree_parents
        color_skeleton = XSensConstants.color_skeleton
    
    if not isinstance(actions, list):
        actions = [actions]
        
    for action in actions:
        # Plot actions
        ax.scatter(action[:, 0], action[:, 1], action[:, 2], c='k', s=10)
        
        # Plot bones
        for i, parent in enumerate(kintree_parents):
            c_joint = action[i]
            
            xs = [c_joint[0]]
            ys = [c_joint[1]]
            zs = [c_joint[2]]
            
            # if pose_rotmat is not None and i == 6:
            if pose_rotmat is not None:
                j_rotmat = pose_rotmat[i]
                ax.scatter(c_joint[0], c_joint[1], c_joint[2], c='k', s=30)
                ax.quiver(c_joint[0], c_joint[1], c_joint[2], j_rotmat[0, 0], j_rotmat[1, 0], j_rotmat[2, 0], color='r', length=0.15, normalize=True) # +X Forward (Red)
                ax.quiver(c_joint[0], c_joint[1], c_joint[2], j_rotmat[0, 1], j_rotmat[1, 1], j_rotmat[2, 1], color='g', length=0.15, normalize=True) # +Y Left (Green)
                ax.quiver(c_joint[0], c_joint[1], c_joint[2], j_rotmat[0, 2], j_rotmat[1, 2], j_rotmat[2, 2], color='b', length=0.15, normalize=True) # +Z Up (Blue)
            
            if parent != -1:
                p_joint = action[parent]
                
                xs.append(p_joint[0])
                ys.append(p_joint[1])
                zs.append(p_joint[2])
                
            ax.plot(xs, ys, zs, color=color_skeleton[i] / 255., linewidth=4, alpha=0.9)
        
    # Origin
    ax.scatter(0, 0, 0, c='k', s=30)
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=size/4, normalize=True) # +X Forward (Red)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=size/4, normalize=True) # +Y Left (Green)
    
    z_scale = 1. / size
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=size/4 * z_scale, normalize=True) # +Z Up (Blue)
    
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-0.25, 0.5)
    
    ax.set_xticks(np.linspace(-size, size, 5))
    ax.set_yticks(np.linspace(-size, size, 5))
    ax.set_zticks(np.linspace(-0.25, 0.5, 4))
    
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    
    ax.tick_params(pad=0)
    
    ax.set_box_aspect([2, 2, 0.75])
    
def save_gif(array, out_path, fps=4):
    """
    Save a video array as a GIF.
    
    Args:
        array: np.ndarray of shape (T, C, H, W) or (T, H, W, C)
        out_path: output .gif path
        fps: frames per second
    """
    # Convert (T, C, H, W) → (T, H, W, C)
    if array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))

    # Normalize to uint8 if needed
    if array.dtype != np.uint8:
        array = np.clip(array * 255, 0, 255).astype(np.uint8)

    # Save
    imageio.mimsave(out_path, array, format='GIF', fps=fps)