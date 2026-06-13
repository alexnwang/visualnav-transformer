"""Diagnostic: is the model's conditioning being used? Is HL→LL exposure
bias the bottleneck? Loads the small EfficientNet baseline checkpoint
(`hdp-modeA-bs256-lr1e-4-4xL40s-Lkin0`) and runs three sample modes:

  vanilla   — normal sampling (HL diffusion → LL diffusion).
  zero_cond — replace `cond` with zeros (test if encoder output matters).
  gt_hl     — skip HL diffusion, use GT HL waypoints for LL cross-attn.
              (Test if HL→LL exposure bias is the bottleneck.)

Reports `init_mje` (no-motion baseline) and `eval/mje_final` per mode on
the same batches.
"""
import os, sys, time
sys.path.insert(0, '/home/anw2067/visualnav-transformer/train')
sys.path.insert(0, '/home/anw2067/hdp')

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from rk_diffuser.nymeria.dataset import NymeriaHDPDataset
from rk_diffuser.nymeria.model import HDPNymeria
from rk_diffuser.nymeria.fk import (
    cumulate_deltas, forward_kinematics, euler_xyz_to_rotmat,
)
from vint_train.training.nymeria_training_utils import (
    unnormalize_data_smpl_pose_gaussian, set_gaussian_stats,
)
from vint_train.data.misc import XsensSkeleton

CKPT = "/home/anw2067/hdp/logs/hdp-modeA-bs256-lr1e-4-4xL40s-Lkin0/model_126570.pt"
N_BATCHES = 8           # 8 × 64 = 512 eval samples per mode
BATCH_SIZE = 64

# ---- build dataset (matches eval split exactly) ----
print("Loading dataset...", flush=True)
ds = NymeriaHDPDataset(
    data_folder='/scratch/anw2067/nymeria_visibility_matrix',
    data_split_folder='/home/anw2067/visualnav-transformer/train/data_splits/nymeria/test',
    gaussian_normalization_stats_path='/home/anw2067/visualnav-transformer/train/nymeria_nomad_mean_var_stats.json',
    dataset_name='nymeria', image_size=(96, 96),
    transform=transforms.Compose([transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    waypoint_spacing=1, min_dist_cat=8, max_dist_cat=8,
    min_action_distance=0, max_action_distance=20,
    negative_goals=False, len_traj_pred=8, context_size=3,
    goal_type=None, preserve_pose_up_down=False, end_slack=0,
    goals_per_obs=1, normalize=True, obs_type='png',
    waypoint_mask_prob=None,
    goal_body_parts=['Pelvis', 'Head', 'R_Hand', 'L_Hand'],
    t_hl=1, dt=8,
)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=8, pin_memory=True)

# ---- build model (matches the small EfficientNet baseline config) ----
print("Building model...", flush=True)
model = HDPNymeria(
    t_hl=1, t_ll=8, dt=8,
    num_keypoints=4, delta_dim=48, context_size=3,
    encoding_dim=512, encoder_backbone="efficientnet-b0",
    image_size=(96, 96), proprioception=True,
    encoder_pool_features=True, encoder_pool_curr_obs=False,
    encoder_pos_enc_3d=False, encoder_project_encoding=False,
    encoder_mha_num_heads=4, encoder_mha_num_layers=4, encoder_mha_ff_factor=4,
    n_timesteps=100, predict_epsilon=True, clip_x0=1.0, ll_clip_x0=1.0,
    hl_dim=128, hl_n_head=4, hl_num_encoder_layers=2, hl_num_decoder_layers=2,
    ll_dim=128, ll_n_head=4, ll_num_decoder_layers=2,
    backbone_dropout=0.1,
).cuda()

print(f"Loading checkpoint: {CKPT}", flush=True)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
# We saved both "model" (live) and "ema" (EMA copy). Eval should use EMA.
state = ckpt.get("ema", ckpt.get("model", ckpt))
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:    print(f"  WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
if unexpected: print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
model.eval()
print(f"step trained: {ckpt.get('step', '?')}", flush=True)

# NB: the gaussian stats global is already set by `NymeriaHDPDataset.__init__`
# above (it calls `set_gaussian_stats(self.ACTION_STATS)` while loading the
# stats JSON). The unnormalize call uses the global, so no extra setup here.

# ---- helpers ----
skel = XsensSkeleton()

@torch.no_grad()
def fk_kp(root, eul):
    rot = euler_xyz_to_rotmat(eul.view(-1, 15, 3))
    return forward_kinematics(skel, root, rot)

@torch.no_grad()
def mje_from_pred_deltas(pred_deltas_norm, first_pose, gt_actions_with_initial):
    B, T, _ = pred_deltas_norm.shape
    pred_deltas = unnormalize_data_smpl_pose_gaussian(
        pred_deltas_norm.reshape(B * T, -1)
    ).reshape(B, T, -1)
    first_pose = first_pose.reshape(B, -1)
    gt = gt_actions_with_initial.reshape(B, -1)
    pred_xyz, pred_rot = cumulate_deltas(first_pose, pred_deltas, num_segments=15)
    pred_kp = forward_kinematics(skel, pred_xyz[:, -1], pred_rot[:, -1])
    gt_kp = fk_kp(gt[:, :3], gt[:, 3:])
    return float((pred_kp - gt_kp).pow(2).sum(-1).sqrt().mean())

@torch.no_grad()
def init_mje(first_pose, gt_actions_with_initial):
    B = first_pose.shape[0]
    fp = first_pose.reshape(B, -1)
    gt = gt_actions_with_initial.reshape(B, -1)
    init_kp = fk_kp(fp[:, :3], fp[:, 3:])
    gt_kp = fk_kp(gt[:, :3], gt[:, 3:])
    return float((init_kp - gt_kp).pow(2).sum(-1).sqrt().mean())

@torch.no_grad()
def sample_modes(batch, mode):
    """Return predicted LL deltas (B, T_LL, 48) under the given mode."""
    cond = model.encode(batch)
    if mode == 'zero_cond':
        cond = torch.zeros_like(cond)
    B = cond.size(0); device = cond.device
    use_gt_hl = mode in ('gt_hl', 'gt_hl_no_clip')
    no_ll_clip = mode in ('no_ll_clip', 'gt_hl_no_clip')

    # Optionally disable LL sample-time clip for this run.
    saved_clip = model.ll_diffusion.clip_x0
    if no_ll_clip:
        model.ll_diffusion.clip_x0 = None
    try:
        # ---- HL ----
        if use_gt_hl:
            hl_pred = batch['waypoints'].flatten(-2, -1).to(device)
        else:
            hl_shape = (B, model.t_hl, model.waypoint_dim)
            def hl_denoise(x, t): return model.hl_backbone(x, t, cond)
            hl_pred = model.hl_diffusion.p_sample_loop(hl_shape, hl_denoise, device)
        # ---- LL ----
        ll_shape = (B, model.t_ll, model.delta_dim)
        def ll_denoise(x, t): return model.ll_backbone(x, t, cond, hl_pred)
        ll_pred = model.ll_diffusion.p_sample_loop(ll_shape, ll_denoise, device)
    finally:
        model.ll_diffusion.clip_x0 = saved_clip
    return ll_pred

# ---- modes ----
# 'vanilla'        — baseline sampling
# 'gt_hl'          — GT HL waypoints, sampled LL
# 'no_ll_clip'     — vanilla + disable LL sample-time clip (sets clip_x0=None)
# 'gt_hl_no_clip'  — GT HL + disable LL sample-time clip
modes = ['vanilla', 'gt_hl', 'no_ll_clip', 'gt_hl_no_clip']
results = {m: [] for m in modes}
init_vals = []

print(f"\nRunning {N_BATCHES} batches × bs={BATCH_SIZE} per mode...", flush=True)
for i, batch in enumerate(loader):
    if i >= N_BATCHES: break
    batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    init_vals.append(init_mje(batch['first_pose'], batch['gt_actions_with_initial']))
    for mode in modes:
        t0 = time.time()
        ll_pred = sample_modes(batch, mode)
        mje = mje_from_pred_deltas(ll_pred, batch['first_pose'],
                                    batch['gt_actions_with_initial'])
        results[mode].append(mje)
        print(f"  batch {i+1}/{N_BATCHES}  mode={mode:10s}  mje={mje:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

print("\n" + "=" * 70)
print(f"init_mje (no-motion baseline):                 {np.mean(init_vals):.4f}")
print("-" * 70)
for m in modes:
    delta = np.mean(results[m]) - np.mean(init_vals)
    print(f"  mode={m:10s}  eval/mje_final: {np.mean(results[m]):.4f}  "
          f"(Δ vs init: {delta:+.4f})")
print("=" * 70)
print()
print("Interpretation:")
print("  - no_ll_clip ≪ vanilla       : clip_x0=1.0 was pinning motion to zero")
print("  - gt_hl_no_clip ≪ no_ll_clip : HL→LL exposure bias still hurts on top")
