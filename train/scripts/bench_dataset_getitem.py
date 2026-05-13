"""
Microbenchmark of ViNT_Nymeria_Dataset.__getitem__ end-to-end and per-block.

Run inside the singularity container so image symlinks resolve into the squashfs:

    srun --pty --cpus-per-task=4 --mem=16G --time=00:10:00 \\
        singularity exec --overlay /scratch/anw2067/nymeria.sqf:ro \\
        /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif bash -l
    conda activate nomad_train2
    python train/scripts/bench_dataset_getitem.py
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, "/home/anw2067/visualnav-transformer/train")
os.chdir("/home/anw2067/visualnav-transformer/train")

import torch
from PIL import Image as PILImage
from torchvision import transforms

from vint_train.data import data_utils as DU
from vint_train.data import vint_dataset as VD


T = {}


def rec(name, dt):
    T.setdefault(name, []).append(dt)


def wrap_method(obj, name):
    f = getattr(obj, name)
    def w(*a, **kw):
        t = time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            rec(name, time.perf_counter() - t)
    setattr(obj, name, w)


def wrap_module_attr(mod, name):
    f = getattr(mod, name)
    def w(*a, **kw):
        t = time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            rec(name, time.perf_counter() - t)
    setattr(mod, name, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-folder", default="/scratch/anw2067/nymeria_visibility_matrix_lite_dist8")
    ap.add_argument("--split-dir",   default="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/train/")
    ap.add_argument("--min-dist", type=int, default=8)
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--len-traj-pred", type=int, default=8)
    ap.add_argument("--context-size", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"data_folder: {args.data_folder}")
    print(f"split_dir:   {args.split_dir}")
    print(f"dist=[{args.min_dist},{args.max_dist}]  context_size={args.context_size}  len_traj_pred={args.len_traj_pred}")

    tx = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = VD.ViNT_Nymeria_Dataset(
        data_folder=args.data_folder,
        data_split_folder=args.split_dir,
        dataset_name="nymeria",
        image_size=[224, 224],
        transform=tx,
        waypoint_spacing=1,
        min_dist_cat=args.min_dist,
        max_dist_cat=args.max_dist,
        min_action_distance=0,
        max_action_distance=20,
        negative_goals=False,
        len_traj_pred=args.len_traj_pred,
        context_size=args.context_size,
        goal_type="draw",
        preserve_pose_up_down=False,
        end_slack=0,
        goals_per_obs=1,
        normalize=True,
        gaussian_normalization_stats_path="nymeria_nomad_mean_var_stats.json",
        waypoint_mask_prob="uniform",
        goal_body_parts=None,
    )
    print(f"len={len(ds):,}")

    wrap_method(ds, "_get_trajectory")
    wrap_method(ds, "_load_image")
    wrap_method(ds, "_compute_actions_nymeria_smpl")
    wrap_method(ds, "_compute_actions_nymeria_smpl_relpelvis")
    wrap_module_attr(VD, "get_action_smpl_torch")
    wrap_module_attr(VD, "get_delta_smpl")
    wrap_module_attr(VD, "draw_keypoints")
    wrap_module_attr(DU, "resize_and_aspect_crop")

    _orig_load = VD.torch.load
    def _timed_load(*a, **kw):
        t = time.perf_counter()
        try:
            return _orig_load(*a, **kw)
        finally:
            rec("torch.load", time.perf_counter() - t)
    VD.torch.load = _timed_load

    _orig_open = PILImage.open
    def _timed_open(*a, **kw):
        t = time.perf_counter()
        try:
            return _orig_open(*a, **kw)
        finally:
            rec("PIL.Image.open", time.perf_counter() - t)
    PILImage.open = _timed_open
    DU.Image.open = _timed_open

    rng = random.Random(args.seed)
    indices = [rng.randint(0, len(ds) - 1) for _ in range(args.warmup + args.samples)]

    print(f"warmup {args.warmup}...")
    for i in indices[:args.warmup]:
        ds[i]
    T.clear()

    print(f"timing {args.samples}...")
    ov = []
    for i in indices[args.warmup:]:
        t = time.perf_counter()
        ds[i]
        ov.append(time.perf_counter() - t)

    def stats(xs):
        return f"n={len(xs):4d} mean={statistics.mean(xs)*1000:7.3f}ms med={statistics.median(xs)*1000:7.3f}ms"

    print(f"\n=== End-to-end __getitem__ ({args.samples}) ===")
    print(f"  {stats(ov)}  total={sum(ov)*1000:.1f}ms")

    e2e = sum(ov)
    print(f"\n=== Per-block per-getitem (avg) ===")
    print(f"  {'block':<42s} {'calls/get':>9s} {'ms/call':>9s} {'ms/get':>9s} {'%e2e':>6s}")
    rows = sorted(
        [(n, len(xs) / args.samples, sum(xs) / len(xs) * 1000, sum(xs) / args.samples * 1000, 100 * sum(xs) / e2e)
         for n, xs in T.items()],
        key=lambda r: -r[3],
    )
    for n, c, mc, mg, p in rows:
        print(f"  {n:<42s} {c:9.2f} {mc:9.3f} {mg:9.3f} {p:5.1f}%")


if __name__ == "__main__":
    main()
