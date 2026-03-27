import sys; sys.path.insert(0, '.')
import numpy as np
import torch
from pathlib import Path
from train.planning.vis_utils import load_camera_model, get_T_C_pelvis, disable_logging, pose_to_image_coords, pose_to_image_coords_v2
from nymeria.data_provider import NymeriaDataProvider # type: ignore
from train.planning.nymeria_dataset import _pose_relpelvis
from train.utils.camera_projection import project_fisheye624_np

disable_logging()

TRACK   = '20230816_s1_jeffery_bryant_act0_p5w199'
VRS_DIR = '/home/anw2067/visualnav-transformer/nymeria_camera_dir'
EP_DIR  = '/home/anw2067/scratch/nymeria_visibility_matrix'

cam_model  = load_camera_model(VRS_DIR, TRACK)
cam_data   = torch.load(f'{EP_DIR}/{TRACK}/camera_data.pt', weights_only=False)
ep_info    = torch.load(f'{EP_DIR}/{TRACK}/ep_info.pt', weights_only=False)
nymeria_dp = NymeriaDataProvider(sequence_rootdir=Path(f'{VRS_DIR}/{TRACK}'), load_wrist=False, load_observer=False)

fisheye_params = cam_data['fisheye_params']
xsens_offsets  = ep_info['xsens_offsets'].float()
T = len(ep_info['all_parts'])
indices = np.linspace(0, T-1, 10, dtype=int)

# --- full pipeline comparison (float32, realistic use case) ---
print("=== Full pipeline (FK → project → rotate) ===")
diffs = []
for t in indices:
    T_C_pelvis = get_T_C_pelvis(nymeria_dp, int(t))
    R    = torch.from_numpy(T_C_pelvis.rotation().to_matrix()).float()
    tvec = torch.from_numpy(T_C_pelvis.translation()).float().squeeze()
    pose = _pose_relpelvis(ep_info, int(t)).float()

    old = pose_to_image_coords(pose, cam_model, xsens_offsets, T_C_pelvis)
    new = pose_to_image_coords_v2(pose, R, tvec, fisheye_params.float(), xsens_offsets)

    mask = (old != -1).all(-1) & (new != -1).all(-1)
    if mask.any():
        diffs.append((old - new)[mask].abs().max().item())

print(f'  float32  max={max(diffs):.4f}px  mean={np.mean(diffs):.4f}px')

# --- raw projection comparison (float64, isolates the math) ---
print("\n=== Raw projection only (float64) ===")
rng = np.random.default_rng(0)
pts = rng.uniform(-0.5, 0.5, (500, 3))
pts[:, 2] = rng.uniform(0.5, 5.0, 500)

old_px = np.stack([np.array(cam_model.project_no_checks(p.reshape(3,1))).flatten() for p in pts])
new_px, _ = project_fisheye624_np(pts, fisheye_params.double().numpy())

print(f'  max={np.abs(old_px - new_px).max():.2e}px  mean={np.abs(old_px - new_px).mean():.2e}px')
