"""Calibrate video_main_rgb mp4 (30fps) -> dataset index (4fps) offset.

Map is linear: mp4_frame = round(A + 7.5*idx). We find A by scanning the whole video
(downsampled) and choosing the A that minimizes total MAE across several spread-out stored
224 frames simultaneously (robust to recurring scenes). Orientation is 'none' (mp4 is upright,
same framing as the stored frames). Then verifies the task indices at full detail.
"""
import argparse, os
import numpy as np
import cv2
from PIL import Image

SLOPE = 7.5  # 30fps / 4fps


def load_stored_small(data_folder, track, idx, s):
    p = os.path.join(data_folder, track, "images", f"{idx}.png")
    if not os.path.isfile(p):
        return None
    return np.asarray(Image.open(p).convert("RGB").resize((s, s), Image.Resampling.LANCZOS), np.float32)


def main(args):
    mp4 = os.path.join(args.vrs_root, args.track, "video_main_rgb.mp4")
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = args.small
    print(f"mp4 frames={n}; reading all at {s}x{s}...", flush=True)
    cds = np.empty((n, s, s, 3), np.float32)
    f = 0
    while f < n:
        ok, bgr = cap.read()
        if not ok:
            break
        cds[f] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (s, s), interpolation=cv2.INTER_AREA)
        f += 1
    cap.release()
    cds = cds[:f]
    print(f"read {f} frames", flush=True)

    # calibration indices (spread); keep those that exist
    cal = [i for i in args.cal_idxs if load_stored_small(args.data_folder, args.track, i, s) is not None]
    stored = {i: load_stored_small(args.data_folder, args.track, i, s) for i in cal}
    print(f"calibration idxs: {cal}", flush=True)

    mae = {i: np.abs(cds - stored[i][None]).mean(axis=(1, 2, 3)) for i in cal}  # (f,) per idx
    Amax = f - int(SLOPE * max(cal)) - 1
    bestA, bestcost = None, 1e18
    for A in range(0, max(1, Amax)):
        cost = 0.0
        for i in cal:
            fr = int(round(A + SLOPE * i))
            if fr >= f:
                cost = 1e18; break
            cost += mae[i][fr]
        if cost < bestcost:
            bestA, bestcost = A, cost
    print(f"BEST offset A={bestA}  mean-MAE/idx={bestcost/len(cal):.2f}", flush=True)
    for i in cal:
        fr = int(round(bestA + SLOPE * i))
        print(f"  idx={i:5d} -> mp4_frame={fr}  MAE={mae[i][fr]:.2f}", flush=True)
    for i in args.idxs:
        fr = int(round(bestA + SLOPE * i))
        print(f"TASK idx={i} -> mp4_frame={fr}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--idxs", type=int, nargs="+", required=True, help="task indices to map")
    p.add_argument("--cal_idxs", type=int, nargs="+",
                   default=[50, 74, 300, 800, 1500, 2500], help="spread-out indices for offset fit")
    p.add_argument("--vrs_root", default="/home/anw2067/scratch/temp_nymeria_large_videos")
    p.add_argument("--data_folder", default="/scratch/anw2067/nymeria_visibility_matrix")
    p.add_argument("--small", type=int, default=64)
    main(p.parse_args())
