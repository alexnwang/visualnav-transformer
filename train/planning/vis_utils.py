import shutil
from nymeria.download_utils import DownloadManager
from nymeria.definitions import DataGroups
from nymeria.data_provider import SequencePathProvider, NymeriaDataProvider
from nymeria.definitions import Subpaths, VrsFiles
from nymeria.recording_data_provider import create_recording_data_provider
from projectaria_tools.core import data_provider
from projectaria_tools.core import sophus
from projectaria_tools.core.sensor_data import TimeDomain

from loguru import logger

import math
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import os
import yaml 
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vint_train.models.nomad.conditional_uned1dnomad import ConditionalUnet1D_NoMaD
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.nymeria_training_utils import unnormalize_data_smpl_pose_gaussian, forward_kinematics_wrapper
from vint_train.training.train_eval_loop import (
    load_model as load_model_from_ckpt
)
from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from vint_train.data.vint_dataset import ViNT_Nymeria_Dataset
from vint_train.training.train_utils import model_output

def disable_logging():
    # logger.disable(None)
    logger.disable("nymeria")
    logger.disable("projectaria_tools")
    logger.disable("VrsDataProvider")
    logger.disable("MpsDataPathsProvider")
    logger.disable("ProgressLogger")
    logger.disable("MultiRecordFileReader")

def download_episode(json_path, save_dir, ep):
    """Download a single episode - must be a top-level function for multiprocessing"""
    dl = DownloadManager(Path(json_path), out_rootdir=Path(save_dir))
    dl.download(match_key=ep, selected_groups=[DataGroups.recording_head, DataGroups.body_motion], ignore_existing=True)
    # remove possibly full dir os.path.join(save_dir, ep, "recording_head", "mps")
    # if os.path.exists(os.path.join(save_dir, ep, "recording_head", "mps")):
    #     shutil.rmtree(os.path.join(save_dir, ep, "recording_head", "mps"))
    # if os.path.exists(os.path.join(save_dir, ep, "recording_head", "data", "et.vrs")):
    #     os.remove(os.path.join(save_dir, ep, "recording_head", "data", "et.vrs"))
    return ep

def load_camera_model(path, ep):
    path = Path(os.path.join(path, ep))
    seq_pd = SequencePathProvider(path)
    vrs_dp = data_provider.create_vrs_data_provider(str(seq_pd.recording_head / VrsFiles.motion))
    return vrs_dp.get_device_calibration().get_camera_calib("camera-rgb")

def project_if_out_of_frame(cam_model, xyz, max_solid_angle=math.pi/2):
    if math.atan2(np.linalg.norm(xyz[:2]), xyz[2]) <= max_solid_angle:
        return cam_model.project_no_checks(xyz[:, None])
    else:
        return None
    
def rotate_coords(coords, image_size):
    rotated_coords = np.empty_like(coords)
    rotated_coords[0] = 2880 - 1 - coords[1]
    rotated_coords[1] = coords[0]
    rotated_coords = rotated_coords / 2880 * image_size
    return rotated_coords

def pose_to_image_coords(pose, cam_model, xsens_offsets, T_C_pelvis, image_size=224) -> torch.Tensor:
    """
    pose: B, 48
    cam_model: CameraCalibration
    xsens_offsets: 23, 3
    T_C_pelvis: Sophus SE3
    
    Returns:
        image_coords: B, 15, 2
    """
    device = pose.device
    xsens_skel = XsensSkeleton(offsets=xsens_offsets)
    pose_xyz, pose_rpy = forward_kinematics_wrapper(pose, xsens_skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, 15, 3
    
    R_C_pelvis = torch.from_numpy(T_C_pelvis.rotation().to_matrix()).to(device, dtype=pose.dtype)
    t_C_pelvis = torch.from_numpy(T_C_pelvis.translation()).to(device, dtype=pose.dtype)
    
    # Ensure t_C_pelvis is the right shape: (3,) or (1, 3) -> (3,)
    if t_C_pelvis.dim() > 1:
        t_C_pelvis = t_C_pelvis.squeeze(0)
    
    # convert everything to camera frame
    # pose_xyz: (B, 15, 3) - 15 body parts, each with 3D coordinates
    # R_C_pelvis: (3, 3) - rotation matrix from pelvis frame to camera frame
    # t_C_pelvis: (3,) - translation vector from pelvis frame to camera frame
    # Transform: R_C_pelvis @ p + t_C_pelvis for each point p
    # Use matmul to transform each 3D point: (B, 15, 3) @ (3, 3).T -> (B, 15, 3)
    pose_xyz_cam = (torch.matmul(pose_xyz, R_C_pelvis.T) + t_C_pelvis).detach().cpu().numpy().astype(np.float64) # B, 15, 3
    
    res = []
    for b in range(pose.shape[0]):
        res.append([])
        for i in range(pose_xyz_cam.shape[1]): # for each body part
            coords = project_if_out_of_frame(cam_model, pose_xyz_cam[b, i, :])
            if coords is not None:
                rotated_coords = rotate_coords(coords, image_size)
                res[b].append(torch.tensor(rotated_coords))
            else:
                res[b].append(torch.tensor([-1, -1]))
        res[b] = torch.stack(res[b], dim=0)
    image_coords = torch.stack(res, dim=0)
    return image_coords
    
    
def get_T_C_pelvis(nymeria_dp, index):
    """
    nymeria_dp: NymeriaDataProvider
    index: int
    """
    start_ns, end_ns = nymeria_dp._NymeriaDataProvider__get_timespan_ns()
    curr_ns = start_ns + index * 0.25 * 1e9
    
    T_C_Hd = nymeria_dp.recording_head.vrs_dp.get_device_calibration().get_camera_calib("camera-rgb").get_transform_device_camera().inverse()
    T_Hd_Wd = nymeria_dp.recording_head.get_pose(curr_ns, time_domain=TimeDomain.TIME_CODE)[0].transform_world_device.inverse()
    
    # get pelvis-wd
    data = nymeria_dp.get_synced_poses(curr_ns, return_orientation=True)
    parts = data['parts']
    pelvis = parts[XSensConstants.part_names.index("Pelvis"), 0]
    T_Wd_P = sophus.SE3.from_quat_and_translation(pelvis[:1, None], pelvis[1:4, None], pelvis[4:, None])
    
    return T_C_Hd @ T_Hd_Wd @ T_Wd_P

def draw_image_coords(draw, goal_image_coords, color=(255, 255, 255), num_segments=XSensConstants.upper_body_num_parts, show_text=True, radius=3):
    num_visible = 0
    for part_name in XSensConstants.part_names[:num_segments]:
        index = XSensConstants.part_names.index(part_name)
        parent_index = XSensConstants.kintree_parents[index]

        point = goal_image_coords[0, index]
        if all(point == -1):
            continue
        num_visible += 1

        draw.ellipse([point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius], fill=color)
        if any(x in part_name for x in ["Pelvis", "Head", "Hand"]) and show_text:
            draw.text((point[0], point[1]), part_name, fill=color)
        if parent_index != -1:
            parent_point = goal_image_coords[0, parent_index]
            if not all(parent_point == -1):
                draw.line([*point, *parent_point], fill=color)
    return int(num_visible)

def get_num_visible(goal_image_coords, num_segments=XSensConstants.upper_body_num_parts):
    return int((1.-(goal_image_coords[0, :num_segments] == -1).all(-1).float()).sum())
