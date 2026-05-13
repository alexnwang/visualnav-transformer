"""
Fit a PCA basis over (horizon, 48) unnormalized SMPL action-delta chunks for use
as a low-rank sampling prior in CEM planning (PEVA path).

The action representation here exactly matches what `plan_cem.py --algo peva`
samples and feeds into the WM:
    actions = _actions_smpl(traj_data, curr_time, horizon)   # (H, 48), absolute
    deltas  = get_delta_smpl(actions, num_segments=15)       # (H, 48), unnormalized

Each chunk is flattened to H*48 and stacked. PCA is fit incrementally so we
don't have to materialize the full (N, H*48) matrix.

Usage:
    python -m scripts.fit_action_pca \
        --horizon 8 --K 8 --stride 4 \
        --traj_names_file /home/anw2067/visualnav-transformer/train/data_splits/nymeria/train/traj_names.txt \
        --data_folder /scratch/anw2067/nymeria_visibility_matrix_lite_dist8 \
        --output /scratch/anw2067/action_pca/h8_k8.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

# Project root on sys.path so we can import planning/* and vint_train/*
_TRAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TRAIN_ROOT not in sys.path:
    sys.path.insert(0, _TRAIN_ROOT)

from planning.nymeria_dataset import _actions_smpl
from vint_train.training.nymeria_training_utils import get_delta_smpl

NUM_SEGMENTS = 15


def iter_chunks(traj_names, data_folder, horizon, stride):
    """
    Yield (H, 48) unnormalized delta chunks for every valid curr_time in every
    trajectory.

    Handles both the original ep_info layout (all_parts shape (T, 23, 2, 7))
    and the lite/dist8 layout (all_parts shape (T, 15, 7)). _actions_smpl is
    written against the original layout, so we re-introduce the missing axis
    when needed.
    """
    for traj_name in traj_names:
        ep_path = os.path.join(data_folder, traj_name, "ep_info.pt")
        if not os.path.isfile(ep_path):
            print(f"  skip (missing): {ep_path}", flush=True)
            continue
        traj_data = torch.load(ep_path, weights_only=False)
        # Adapt lite layout (T, 15, 7) -> (T, 15, 1, 7) so _actions_smpl's
        # 4-axis indexing works. _actions_smpl only ever reads index 0 of axis 2.
        if traj_data["all_parts"].ndim == 3:
            traj_data["all_parts"] = traj_data["all_parts"].unsqueeze(2)
        traj_data["all_parts"] = traj_data["all_parts"].to(torch.float32)
        traj_len = len(traj_data["all_parts"])
        # _actions_smpl returns (horizon, 48); curr_time + horizon must be < traj_len
        # to ensure get_delta_smpl can compute frame-to-frame deltas.
        last_curr = traj_len - horizon - 1
        if last_curr < 0:
            continue
        for curr_time in range(0, last_curr + 1, stride):
            actions = _actions_smpl(traj_data, curr_time, horizon)           # (H, 48)
            deltas = get_delta_smpl(actions, num_segments=NUM_SEGMENTS)      # (H, 48)
            yield deltas.flatten().numpy().astype(np.float32)                # (H*48,)
        del traj_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, help="Action chunk horizon (e.g. 8)")
    ap.add_argument("--K", type=int, required=True, help="Number of principal components to keep")
    ap.add_argument("--stride", type=int, default=4, help="curr_time stride within each trajectory")
    ap.add_argument("--traj_names_file", type=str,
                    default="/home/anw2067/visualnav-transformer/train/data_splits/nymeria/train/traj_names.txt")
    ap.add_argument("--data_folder", type=str,
                    default="/scratch/anw2067/nymeria_visibility_matrix_lite_dist8")
    ap.add_argument("--output", type=str, required=True, help="Output .pt path")
    ap.add_argument("--limit", type=int, default=None, help="Use only first N trajectories (sanity)")
    ap.add_argument("--batch_size", type=int, default=4096, help="IncrementalPCA partial_fit batch size")
    args = ap.parse_args()

    H = args.horizon
    D = 48
    feat_dim = H * D
    assert args.K <= feat_dim, f"K={args.K} must be <= H*48={feat_dim}"
    assert args.K <= args.batch_size, f"IncrementalPCA requires batch_size >= K (got {args.batch_size} < {args.K})"

    with open(args.traj_names_file, "r") as f:
        traj_names = [l.strip() for l in f if l.strip()]
    if args.limit is not None:
        traj_names = traj_names[:args.limit]
    print(f"Trajectories: {len(traj_names)}  H={H} D={D} K={args.K} stride={args.stride}", flush=True)

    ipca = IncrementalPCA(n_components=args.K)
    buf = np.empty((args.batch_size, feat_dim), dtype=np.float32)
    buf_n = 0
    n_total = 0
    t0 = time.time()

    def flush():
        nonlocal buf_n
        if buf_n >= args.K:
            ipca.partial_fit(buf[:buf_n])
        else:
            print(f"  WARN: dropped final partial batch of {buf_n} (< K={args.K})", flush=True)
        buf_n = 0

    for chunk in iter_chunks(traj_names, args.data_folder, H, args.stride):
        buf[buf_n] = chunk
        buf_n += 1
        n_total += 1
        if buf_n == args.batch_size:
            flush()
            if (n_total // args.batch_size) % 10 == 0:
                rate = n_total / (time.time() - t0)
                print(f"  [{n_total} chunks] rate={rate:.0f}/s elapsed={time.time()-t0:.0f}s", flush=True)
    flush()

    print(f"Total chunks: {n_total}  fit_time={time.time()-t0:.1f}s", flush=True)

    out = {
        "components": torch.from_numpy(ipca.components_.astype(np.float32)),                    # (K, H*48)
        "mean": torch.from_numpy(ipca.mean_.astype(np.float32)),                                # (H*48,)
        "explained_variance": torch.from_numpy(ipca.explained_variance_.astype(np.float32)),    # (K,)
        "explained_variance_ratio": torch.from_numpy(ipca.explained_variance_ratio_.astype(np.float32)),  # (K,)
        "horizon": H,
        "action_dim": D,
        "K": args.K,
        "n_train_samples": n_total,
        "stride": args.stride,
        "traj_names_file": args.traj_names_file,
        "data_folder": args.data_folder,
    }
    cum_var = float(out["explained_variance_ratio"].sum())
    print(f"Cumulative explained variance ratio (top {args.K}): {cum_var:.4f}", flush=True)
    print(f"Per-component ratio: {out['explained_variance_ratio'].numpy()}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    torch.save(out, tmp)
    os.rename(tmp, args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
