import os as _os
_os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
_os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

import shutil
import OpenGL.error
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
from vint_train.training.nymeria_training_utils import unnormalize_data_smpl_pose_gaussian, forward_kinematics_wrapper, euler_to_rotmat
from vint_train.training.train_eval_loop import (
    load_model as load_model_from_ckpt
)
from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from vint_train.data.vint_dataset import ViNT_Nymeria_Dataset
from vint_train.training.train_utils import model_output
from utilities.camera_projection import project_fisheye624_torch, rotate_aria_pixels_torch

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


def pose_to_image_coords_v2(pose, R_C_pelvis, t_C_pelvis, fisheye_params, xsens_offsets, image_size=224) -> torch.Tensor:
    """
    Vectorized replacement for pose_to_image_coords (v2, uses pure-Python Fisheye624).

    Args:
        pose           : (B, 48)         SMPL pose tensor
        R_C_pelvis     : (3, 3) tensor   rotation    pelvis → camera
        t_C_pelvis     : (3,)   tensor   translation pelvis → camera
        fisheye_params : (15,)  tensor   Fisheye624 intrinsics
        xsens_offsets  : (J, 3) tensor   skeleton joint offsets
        image_size     : int             output image resolution

    Returns:
        image_coords : (B, 15, 2) tensor; (-1, -1) for out-of-FOV joints
    """

    device = pose.device
    xsens_skel = XsensSkeleton(offsets=xsens_offsets)
    pose_xyz, _ = forward_kinematics_wrapper(pose, xsens_skel, XSensConstants.upper_body_num_parts, return_euler=True)  # B, 15, 3

    pose_cam = torch.matmul(pose_xyz, R_C_pelvis.T.to(device=device, dtype=pose.dtype)) + t_C_pelvis.to(device=device, dtype=pose.dtype)  # B, 15, 3
    B, J, _ = pose_cam.shape
    flat = pose_cam.reshape(-1, 3)

    pixels, valid = project_fisheye624_torch(flat, fisheye_params.to(device=device, dtype=pose.dtype))
    rotated = rotate_aria_pixels_torch(pixels, image_size)

    rotated = rotated.clone()
    rotated[~valid] = -1.0
    return rotated.reshape(B, J, 2)


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


# ---------------------------------------------------------------------------
# SMPL skinned-mesh rendering
# ---------------------------------------------------------------------------

_SMPL_MODEL_DIR = '/home/anw2067/scratch/smpl/models'

# Rotation that maps XSens world frame → SMPL frame.
# Derived from the to_smpl transform already used in this codebase.
_R_XSENS2SMPL = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)

# XSens upper-body joint index → SMPL body_pose index k
# (body_pose[k] drives SMPL joint k+1; L3 is dropped — no SMPL equivalent)
_XSENS_TO_SMPL_BP = {
    1:  2,   # L5          → Spine1   (SMPL j3)
    # L3 dropped
    3:  5,   # T12         → Spine2   (SMPL j6)
    4:  8,   # T8          → Spine3   (SMPL j9)
    5: 11,   # Neck        → Neck     (SMPL j12)
    6: 14,   # Head        → Head     (SMPL j15)
    7: 13,   # R_Shoulder  → R_Collar (SMPL j14)
    8: 16,   # R_UpperArm  → R_Shldr  (SMPL j17)
    9: 18,   # R_Forearm   → R_Elbow  (SMPL j19)
   10: 20,   # R_Hand      → R_Wrist  (SMPL j21)
   11: 12,   # L_Shoulder  → L_Collar (SMPL j13)
   12: 15,   # L_UpperArm  → L_Shldr  (SMPL j16)
   13: 17,   # L_Forearm   → L_Elbow  (SMPL j18)
   14: 19,   # L_Hand      → L_Wrist  (SMPL j20)
}

# SMPL kintree: parent joint index for each of the 24 SMPL joints
_SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]

# SMPL-X joint index → XSens upper-body joint index (inverse of _XSENS_TO_SMPL_BP + pelvis)
_SMPLX_JOINT_TO_XSENS = {
    0:  0,   # Pelvis      ↔ Pelvis
    3:  1,   # Spine1      ↔ L5
    6:  3,   # Spine2      ↔ T12
    9:  4,   # Spine3      ↔ T8
    12: 5,   # Neck        ↔ Neck
    13: 11,  # L_Collar    ↔ L_Shoulder
    14: 7,   # R_Collar    ↔ R_Shoulder
    15: 6,   # Head        ↔ Head
    16: 12,  # L_Shoulder  ↔ L_UpperArm
    17: 8,   # R_Shoulder  ↔ R_UpperArm
    18: 13,  # L_Elbow     ↔ L_Forearm
    19: 9,   # R_Elbow     ↔ R_Forearm
    20: 14,  # L_Wrist     ↔ L_Hand
    21: 10,  # R_Wrist     ↔ R_Hand
}

_smpl_model_singleton = None
_renderer_singleton = None
_renderer_size = None


def _get_smpl_model():
    global _smpl_model_singleton
    if _smpl_model_singleton is None:
        import smplx
        _smpl_model_singleton = smplx.create(
            _SMPL_MODEL_DIR, model_type='smplx', gender='neutral', batch_size=1,
        )
    return _smpl_model_singleton


_renderer_render_count = 0
_RENDERER_RESET_INTERVAL = 50  # reset every N renders to reclaim GPU memory


def _reset_renderer():
    global _renderer_singleton, _renderer_size, _renderer_render_count
    if _renderer_singleton is not None:
        _renderer_singleton.delete()
    _renderer_singleton = None
    _renderer_size = None
    _renderer_render_count = 0


def _get_renderer(image_size):
    global _renderer_singleton, _renderer_size, _renderer_render_count
    _renderer_render_count += 1
    needs_reset = _renderer_render_count >= _RENDERER_RESET_INTERVAL
    if _renderer_singleton is None or _renderer_size != image_size or needs_reset:
        import pyrender
        if _renderer_singleton is not None:
            _renderer_singleton.delete()
        _renderer_singleton = pyrender.OffscreenRenderer(image_size, image_size)
        _renderer_size = image_size
        _renderer_render_count = 0
    return _renderer_singleton


def render_smpl_on_image(curr_obs_img, pose, R_C_pelvis, t_C_pelvis,
                         fisheye_params, image_size, alpha=0.7,
                         xsens_offsets=None, draw_skeleton=False,
                         mesh_color=(0.4, 0.6, 0.8)):
    """
    Render a skinned SMPL body mesh composited onto curr_obs_img.

    Upper body is driven by the XSens pose.  Lower body (hips, knees, ankles,
    feet) and hands are held in neutral T-pose (identity local rotations).

    Args:
        curr_obs_img  : (3, H, W) float tensor [0, 1]
        pose          : (1, 48) tensor — pelvis XYZ + 15×3 global XSens Euler (xyz)
        R_C_pelvis    : (3, 3) tensor — rotation  pelvis-frame → camera-frame
        t_C_pelvis    : (3,)   tensor — translation pelvis-frame → camera-frame
        fisheye_params: (15,)  tensor — Fisheye624 parameter vector
        image_size    : int
        alpha         : float, mesh opacity [0, 1]
        xsens_offsets : (23, 3) tensor, optional — XSens skeleton offsets.
                        When provided, SMPL-X vertices are corrected via LBS
                        weights so that mapped joints match XSens FK positions.

    Returns:
        PIL.Image with the SMPL mesh blended onto the observation
    """
    import pyrender
    import trimesh
    from scipy.spatial.transform import Rotation as R_scipy

    pose_np = pose[0].detach().cpu().float().numpy()       # (48,)
    pelvis_xyz = pose_np[:3]                                # (3,) XSens world
    euler_angles = pose_np[3:].reshape(15, 3)              # (15, 3)

    # 1 — XSens global rotation matrices for the 15 upper-body joints
    rotmats_xsens = euler_to_rotmat(
        torch.from_numpy(euler_angles).float()
    ).numpy()  # (15, 3, 3)

    # 2 — Re-express global rotations in SMPL coordinate frame
    #     R_smpl = R_x2s @ R_xsens @ R_x2s^T
    rotmats_smpl_global = (
        np.matmul(np.matmul(_R_XSENS2SMPL, rotmats_xsens), _R_XSENS2SMPL.T)
    )  # (15, 3, 3)

    # 3 — Build dict of SMPL-joint-index → global rotation for all mapped joints
    smpl_global = {}
    smpl_global[0] = rotmats_smpl_global[0]                # Pelvis
    for xs_idx, bp_idx in _XSENS_TO_SMPL_BP.items():
        smpl_global[bp_idx + 1] = rotmats_smpl_global[xs_idx]

    # 4 — Convert global → local rotations for each mapped joint.
    #     Unmapped joints (lower body, hands) keep identity → neutral T-pose.
    #     R_local_i = R_global_parent^T @ R_global_i
    body_pose_aa = np.zeros((21, 3), dtype=np.float32)     # identity = zero rotvec; SMPL-X body has 21 joints
    for bp_idx in range(21):
        smpl_j = bp_idx + 1
        if smpl_j not in smpl_global:
            continue
        parent_j = _SMPL_PARENTS[smpl_j]
        R_parent = smpl_global.get(parent_j, np.eye(3, dtype=np.float32))
        R_local = R_parent.T @ smpl_global[smpl_j]
        body_pose_aa[bp_idx] = R_scipy.from_matrix(R_local).as_rotvec()

    global_orient_aa = R_scipy.from_matrix(smpl_global[0]).as_rotvec().astype(np.float32)

    # 5 — SMPL forward pass
    smpl = _get_smpl_model()
    with torch.no_grad():
        output = smpl(
            global_orient=torch.from_numpy(global_orient_aa).unsqueeze(0),   # (1, 3)
            body_pose=torch.from_numpy(body_pose_aa.flatten()).unsqueeze(0),  # (1, 63)
            betas=torch.zeros(1, 10),
            expression=torch.zeros(1, 10),
            return_verts=True,
        )
    verts_smpl = output.vertices[0].numpy()   # (10475, 3), SMPL-X frame
    faces = smpl.faces                         # (F, 3)

    # Center mesh on the SMPL-X pelvis joint so that pelvis is at origin,
    # matching the pelvis-relative convention used by FK and the camera transform.
    smplx_pelvis_pos = output.joints[0, 0].detach().cpu().numpy()  # (3,)
    verts_smpl = verts_smpl - smplx_pelvis_pos[None, :]

    # 6 — SMPL frame → XSens frame, offset by pelvis displacement.
    #     The FK skeleton starts from root_xyz (= pelvis_xyz), so all joint
    #     positions include it.  The mesh must match: add pelvis_xyz so that
    #     the subsequent T_C_pelvis transform produces consistent camera-frame
    #     coordinates for both skeleton and skin.
    verts_world = verts_smpl @ _R_XSENS2SMPL + pelvis_xyz[None, :]   # (10475, 3)

    # 7 — XSens world → camera frame  (row-vector convention)
    R_C = R_C_pelvis.detach().cpu().numpy().astype(np.float32)
    t_C = t_C_pelvis.detach().cpu().numpy().astype(np.float32)
    verts_cam = verts_world @ R_C.T + t_C[None, :]   # (10475, 3), OpenCV: Z-forward

    # 7b — Fisheye pre-warp: adjust X,Y so that pyrender's pinhole projection
    #       reproduces the full Fisheye624 result.  For each vertex (X,Y,Z):
    #         fisheye(X,Y,Z) → raw pixel (u_fish, v_fish)
    #         X' = (u_fish - cu) * Z / f,  Y' = (v_fish - cv) * Z / f
    #       Then pinhole(X',Y',Z) = f*X'/Z + cu = u_fish.  Exact at vertices.
    params = fisheye_params.detach().cpu().numpy()
    f_val, cu, cv = float(params[0]), float(params[1]), float(params[2])

    verts_cam_torch = torch.from_numpy(verts_cam).float()
    fish_raw, fish_valid = project_fisheye624_torch(
        verts_cam_torch, fisheye_params.cpu().float()
    )  # fish_raw: (N, 2) raw sensor pixels [u, v]
    fish_raw_np = fish_raw.numpy()
    fish_valid_np = fish_valid.numpy()

    Z = verts_cam[:, 2]
    verts_cam_warped = verts_cam.copy()
    verts_cam_warped[:, 0] = (fish_raw_np[:, 0] - cu) * Z / f_val
    verts_cam_warped[:, 1] = (fish_raw_np[:, 1] - cv) * Z / f_val
    # invalid verts (behind camera / outside FOV) — leave as-is, pyrender clips them
    verts_cam_warped[~fish_valid_np] = verts_cam[~fish_valid_np]

    # 8 — Aria sensor is mounted 90° CW relative to display frame.
    verts_cam_warped = verts_cam_warped[:, [1, 0, 2]] * np.array([-1., 1., 1.], dtype=np.float32)

    # 9 — OpenCV → OpenGL convention (pyrender camera looks down -Z)
    verts_gl = verts_cam_warped * np.array([1., -1., -1.], dtype=np.float32)

    # Pinhole intrinsics (unchanged — the warp makes pinhole match fisheye)
    scale = image_size / 2880.0
    fx = fy = f_val * scale
    cx = (2879.0 - cv) * scale
    cy = cu * scale
    
    # 10 — Render with PyRender
    mesh_trimesh = trimesh.Trimesh(
        vertices=verts_gl.astype(np.float64), faces=faces, process=False
    )
    obs_np = (
        curr_obs_img.permute(1, 2, 0).detach().cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)
    obs_pil = Image.fromarray(obs_np)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*mesh_color, 1.0),
        metallicFactor=0.2,
        roughnessFactor=0.6,
    )
    mesh_r = pyrender.Mesh.from_trimesh(mesh_trimesh, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    scene.add(mesh_r)
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    scene.add(cam, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=4.0)
    scene.add(light, pose=np.eye(4))

    renderer = _get_renderer(image_size)
    try:
        color_render, depth_render = renderer.render(scene)
    except OpenGL.error.GLError:
        _reset_renderer()
        renderer = _get_renderer(image_size)
        try:
            color_render, depth_render = renderer.render(scene)
        except OpenGL.error.GLError:
            logger.warning("OpenGL render failed twice, skipping SMPL overlay")
            _reset_renderer()
            color_render = None

    if color_render is not None:
        render_pil = Image.fromarray(color_render)
        min_depth = 0.01
        fade_range = 0.03
        alpha_map = np.zeros_like(depth_render)
        visible = depth_render > 0
        alpha_map[visible] = np.clip(
            (depth_render[visible] - min_depth) / fade_range, 0.0, 1.0
        ) * alpha
        mask = (alpha_map * 255).astype(np.uint8)
        obs_pil.paste(render_pil, mask=Image.fromarray(mask, mode='L'))

    # Draw FK skeleton on top of the rendered mesh
    if draw_skeleton and xsens_offsets is not None:
        # Compute FK joints for skeleton overlay
        xsens_skel = XsensSkeleton(offsets=xsens_offsets)
        fk_joints, _ = forward_kinematics_wrapper(
            pose, xsens_skel, XSensConstants.upper_body_num_parts, return_euler=True
        )
        fk_joints = fk_joints[0].detach().cpu().numpy()  # (15, 3)
        fk_pelvis_rel = fk_joints - fk_joints[0:1]
        fk_joints_xsens = fk_pelvis_rel + pelvis_xyz[None, :]  # (15, 3) XSens frame
        fk_joints_cam = fk_joints_xsens @ R_C.T + t_C[None, :]
        fk_cam_torch = torch.from_numpy(fk_joints_cam).float()
        fk_px, fk_valid = project_fisheye624_torch(fk_cam_torch, fisheye_params.cpu().float())
        fk_disp = rotate_aria_pixels_torch(fk_px, image_size)  # (15, 2)

        # Mark invalid joints as (-1, -1) for draw_image_coords
        fk_disp_np = fk_disp.clone()
        fk_disp_np[~fk_valid] = -1.0
        image_coords = fk_disp_np.unsqueeze(0)  # (1, 15, 2)

        draw = ImageDraw.Draw(obs_pil)
        draw_image_coords(draw, image_coords)

    return obs_pil
