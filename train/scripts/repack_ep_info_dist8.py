"""
Repack ep_info.pt files into a slimmer parallel directory for goal_type=draw,
min_dist_cat=max_dist_cat=8, upper-body (15 segments).

Reads:  /scratch/anw2067/nymeria_visibility_matrix/<traj>/ep_info.pt   (~73 MB)
Writes: /scratch/anw2067/nymeria_visibility_matrix_lite_dist8/<traj>/ep_info.pt   (~1.8 MB)
        /scratch/anw2067/nymeria_visibility_matrix_lite_dist8/<traj>/images       (symlink, target copied verbatim)

Trim:
  all_parts                (T, 23, 2, 7) f32  -> (T, 15, 7) f32           [:, :15, 0, :]
  image_projection_matrix  (T, 65, 23, 2) i32 -> (T, 23, 2)  i16          [:, 40, :, :]   (target_idx for dist_cat=8)
  xsens_offsets            (23, 3) f32        -> kept as-is
  depth_matrix, xsens_xyz, xsens_eulerxyz: dropped.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import torch

SRC_BASE = "/scratch/anw2067/nymeria_visibility_matrix"
DST_BASE = "/scratch/anw2067/nymeria_visibility_matrix_lite_dist8"
TARGET_IDX = 40
NUM_SEGMENTS = 15


def convert_one(traj):
    src_dir = os.path.join(SRC_BASE, traj)
    dst_dir = os.path.join(DST_BASE, traj)
    dst_ep = os.path.join(dst_dir, "ep_info.pt")

    t0 = time.time()
    d = torch.load(os.path.join(src_dir, "ep_info.pt"), weights_only=False)
    out = {
        "all_parts":               d["all_parts"][:, :NUM_SEGMENTS, 0, :].contiguous(),
        "image_projection_matrix": d["image_projection_matrix"][:, TARGET_IDX, :, :].to(torch.int16).contiguous(),
        "xsens_offsets":           d["xsens_offsets"].contiguous(),
    }
    os.makedirs(dst_dir, exist_ok=True)
    tmp = dst_ep + ".tmp"
    torch.save(out, tmp)
    os.rename(tmp, dst_ep)

    src_images = os.path.join(src_dir, "images")
    if os.path.islink(src_images):
        dst_images = os.path.join(dst_dir, "images")
        target = os.readlink(src_images)
        if os.path.lexists(dst_images):
            os.remove(dst_images)
        os.symlink(target, dst_images)

    return traj, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="for testing: process only the first N trajectories")
    args = ap.parse_args()

    trajs = sorted(
        d for d in os.listdir(SRC_BASE)
        if os.path.isdir(os.path.join(SRC_BASE, d))
        and os.path.isfile(os.path.join(SRC_BASE, d, "ep_info.pt"))
    )
    if args.limit is not None:
        trajs = trajs[:args.limit]
    print(f"Converting {len(trajs)} trajectories with {args.workers} workers")
    print(f"  src: {SRC_BASE}")
    print(f"  dst: {DST_BASE}")
    print(f"  TARGET_IDX={TARGET_IDX}  NUM_SEGMENTS={NUM_SEGMENTS}")

    os.makedirs(DST_BASE, exist_ok=True)

    t_start = time.time()
    done = 0
    errors = []
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(convert_one, trajs, chunksize=4):
            done += 1
            if done % 100 == 0 or done == len(trajs):
                rate = done / (time.time() - t_start)
                eta = (len(trajs) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(trajs)}] elapsed={time.time()-t_start:.1f}s  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t_start
    total_bytes = sum(
        os.path.getsize(os.path.join(DST_BASE, t, "ep_info.pt"))
        for t in trajs
        if os.path.isfile(os.path.join(DST_BASE, t, "ep_info.pt"))
    )
    print(f"\nDone in {elapsed:.1f}s. Output total: {total_bytes/1024/1024/1024:.2f} GB across {len(trajs)} trajectories")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
