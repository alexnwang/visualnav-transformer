"""
Preprocessing script: extract Fisheye624 intrinsics and per-timestep T_C_Pelvis
transforms from NymeriaDataProvider and save as camera_data.pt per track.

Output per track  ({ep_folder}/{track}/camera_data.pt):
    fisheye_params : (15,)     float64  Fisheye624 intrinsic parameter vector
    T_C_pelvis     : (T, 4, 4) float64  SE3 matrix for each timestep index

Usage:
    python preprocess_camera_data.py \\
        --ep_folder /scratch/nymeria_visibility_matrix \\
        --vrs_folder /path/to/nymeria_camera_dir \\
        --split nymeria/train nymeria/test \\
        [--json_file /path/to/visibility_no_data.json] \\
        [--overwrite]
"""

import argparse
import os
import shutil
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from nymeria.data_provider import NymeriaDataProvider  # type: ignore
from nymeria.definitions import VrsFiles  # type: ignore
from planning.vis_utils import load_camera_model, get_T_C_pelvis, disable_logging, download_episode

DATA_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data_splits")


def se3_to_matrix(T) -> np.ndarray:
    """Convert a Sophus SE3 object to a (4, 4) numpy matrix."""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = T.rotation().to_matrix()
    mat[:3, 3] = T.translation()
    return mat


def ensure_recording_head(vrs_folder: str, track: str, json_file: str):
    """Download recording_head for a track if the VRS file is not present."""
    vrs_path = os.path.join(vrs_folder, track, "recording_head", VrsFiles.motion)
    if not os.path.exists(vrs_path):
        print(f"[download] {track} — recording_head not found, downloading...")
        download_episode(json_file, vrs_folder, track)


def preprocess_track(
    ep_folder: str,
    vrs_folder: str,
    track: str,
    json_file: str | None,
    overwrite: bool = False,
) -> bool:
    disable_logging()
    out_path = os.path.join(ep_folder, track, "camera_data.pt")

    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {track} — already exists")
        return True

    ep_info_path = os.path.join(ep_folder, track, "ep_info.pt")
    if not os.path.exists(ep_info_path):
        print(f"[skip] {track} — ep_info.pt not found in {ep_folder}")
        return False

    if json_file is not None:
        ensure_recording_head(vrs_folder, track, json_file)

    vrs_track_path = os.path.join(vrs_folder, track, "recording_head", VrsFiles.motion)
    if not os.path.exists(vrs_track_path):
        print(f"[skip] {track} — recording_head VRS not found in {vrs_folder}")
        return False

    traj_data = torch.load(ep_info_path, weights_only=False)
    T = len(traj_data["all_parts"])
    del traj_data

    recording_head_dir = os.path.join(vrs_folder, track, "recording_head")
    for attempt in range(2):
        try:
            cam_model = load_camera_model(vrs_folder, track)
            fisheye_params = np.array(cam_model.projection_params(), dtype=np.float64)
            nymeria_dp = NymeriaDataProvider(
                sequence_rootdir=Path(os.path.join(vrs_folder, track)),
                load_wrist=False,
                load_observer=False,
            )
            break
        except Exception as e:
            if attempt == 0 and json_file is not None:
                print(f"[retry] {track} — load failed ({e}), deleting and re-downloading...")
                shutil.rmtree(recording_head_dir, ignore_errors=True)
                download_episode(json_file, vrs_folder, track)
            else:
                print(f"[error] {track} — failed to load data provider after retry: {e}")
                return False

    T_C_pelvis_all = np.zeros((T, 4, 4), dtype=np.float64)
    failed = 0
    for t in range(T):
        try:
            T_C_pelvis_all[t] = se3_to_matrix(get_T_C_pelvis(nymeria_dp, t))
        except Exception:
            T_C_pelvis_all[t] = np.nan
            failed += 1

    if failed > 0:
        print(f"[warn] {track} — {failed}/{T} timesteps failed (stored as NaN)")

    torch.save(
        {
            "fisheye_params": torch.from_numpy(fisheye_params),   # (15,)
            "T_C_pelvis":     torch.from_numpy(T_C_pelvis_all),   # (T, 4, 4)
        },
        out_path,
    )
    # Reclaim disk: the source recording_head VRS (~600MB-2GB per track) is
    # only needed to extract the two tensors above. Once camera_data.pt is on
    # disk we don't need the raw VRS again. Each worker handles its own track
    # so this doesn't race with others. Idempotency comes from the
    # skip-on-existing check at the top of preprocess_track.
    shutil.rmtree(recording_head_dir, ignore_errors=True)
    print(f"[done] {track} — {T} timesteps → {out_path}")
    return True


def load_tracks_from_splits(splits: list[str]) -> list[str]:
    """Deduplicated list of track names from one or more data split directories."""
    tracks = []
    seen = set()
    for split in splits:
        traj_file = os.path.join(DATA_SPLITS_DIR, split, "traj_names.txt")
        if not os.path.exists(traj_file):
            print(f"[warn] traj_names.txt not found for split: {split}")
            continue
        with open(traj_file) as f:
            for line in f:
                t = line.strip()
                if t and t not in seen:
                    tracks.append(t)
                    seen.add(t)
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_folder",
                        default="/home/anw2067/scratch/nymeria_visibility_matrix",
                        help="Root directory containing per-track ep_info.pt files; "
                             "camera_data.pt will be saved here too")
    parser.add_argument("--vrs_folder",
                        default="/home/anw2067/visualnav-transformer/nymeria_camera_dir",
                        help="Root directory containing per-track recording_head VRS files")
    parser.add_argument("--split", nargs="+", required=True,
                        help="One or more split paths relative to train/data_splits/, "
                             "e.g. nymeria/train nymeria/test")
    parser.add_argument("--json_file",
                        default="/home/anw2067/visualnav-transformer/data_jsons/visibility_no_data.json",
                        help="Nymeria download JSON for auto-downloading missing recording_head data")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if camera_data.pt already exists")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel worker processes (default 4)")
    args = parser.parse_args()

    tracks = load_tracks_from_splits(args.split)
    if not tracks:
        print("No tracks found. Check your --split arguments.")
        return

    print(f"Processing {len(tracks)} tracks from splits: {args.split} with {args.num_workers} workers")
    worker_fn = partial(preprocess_track, args.ep_folder, args.vrs_folder,
                        json_file=args.json_file, overwrite=args.overwrite)
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_fn, tracks)
    print(f"\nDone: {sum(results)}/{len(tracks)} tracks processed.")


if __name__ == "__main__":
    main()
