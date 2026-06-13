"""Extract high-res current-obs frames from a track's video_main_rgb mp4 (1408x1408,
30fps), mapped to dataset indices and VALIDATED against the stored 224 frames.

Map: mp4_frame = round(A + 7.5*idx). A is per-track; found by scanning the whole mp4
(downsampled) and minimizing total MAE vs several spread stored frames at once. Each
extracted frame is then re-checked (downsample->224, MAE vs stored) before saving.

Saves highres_frames/{track}/{idx}.png at native 1408x1408 (upright, full fisheye —
same framing as the stored frames, just 40x the pixels).
"""
import argparse, os, shutil, subprocess
import numpy as np
import cv2
from PIL import Image

SLOPE = 7.5  # 30fps / 4fps


def stored_small(data_folder, track, idx, s):
    p = os.path.join(data_folder, track, "images", f"{idx}.png")
    if not os.path.isfile(p):
        return None
    return np.asarray(Image.open(p).convert("RGB").resize((s, s), Image.Resampling.LANCZOS), np.float32)


def read_all_small(mp4, s):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cds = np.empty((n, s, s, 3), np.float32)
    f = 0
    while f < n:
        ok, bgr = cap.read()
        if not ok:
            break
        cds[f] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (s, s), interpolation=cv2.INTER_AREA)
        f += 1
    cap.release()
    return cds[:f]


def calibrate_A(cds, data_folder, track, cal_idxs, s):
    cal = [i for i in cal_idxs if stored_small(data_folder, track, i, s) is not None]
    if not cal:
        raise RuntimeError("no calibration stored frames found")
    stored = {i: stored_small(data_folder, track, i, s) for i in cal}
    mae = {i: np.abs(cds - stored[i][None]).mean(axis=(1, 2, 3)) for i in cal}
    f = len(cds)
    Amax = f - int(SLOPE * max(cal)) - 1
    bestA, bestc = 0, 1e18
    for A in range(0, max(1, Amax)):
        c = 0.0
        for i in cal:
            fr = int(round(A + SLOPE * i))
            if fr >= f:
                c = 1e18; break
            c += mae[i][fr]
        if c < bestc:
            bestA, bestc = A, c
    return bestA, bestc / len(cal), cal, mae


def extract_frames_seq(mp4, frame_nos):
    """Frame-accurate full-res extraction of several frames in ONE sequential pass.
    Sequential cv2 read() is frame-accurate (unlike cap.set() seeking, which can land
    off keyframes) and uses the exact same frame numbering as read_all_small's scan."""
    want = set(int(f) for f in frame_nos)
    cap = cv2.VideoCapture(mp4)
    out, f, last = {}, 0, max(want)
    while f <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if f in want:
            out[f] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        f += 1
    cap.release()
    return out


def main(args):
    mp4 = os.path.join(args.vrs_root, args.track, "video_main_rgb.mp4")
    print(f"scanning {mp4} at {args.small}x{args.small}...", flush=True)
    cds = read_all_small(mp4, args.small)
    print(f"  {len(cds)} frames", flush=True)

    cal_idxs = sorted(set(args.cal_idxs + list(args.curr_times)))
    A, meanmae, cal, mae = calibrate_A(cds, args.data_folder, args.track, cal_idxs, args.small)
    print(f"offset A={A}  mean-MAE/idx={meanmae:.2f}  (cal idxs {cal})", flush=True)

    out_dir = os.path.join(args.out, args.track)
    os.makedirs(out_dir, exist_ok=True)
    idxs = sorted(args.curr_times)
    frame_of = {ct: int(round(A + SLOPE * ct)) for ct in idxs}
    frames = extract_frames_seq(mp4, frame_of.values())   # one sequential full-res pass

    written = []
    for ct in idxs:
        fr = frame_of[ct]
        if fr not in frames:
            raise RuntimeError(f"mp4 has no frame {fr} for idx={ct} (only {len(cds)} frames)")
        native = Image.fromarray(frames[fr])
        out_png = os.path.join(out_dir, f"{ct}.png")
        native.save(out_png)
        # report calibration MAE at this index (coarse), then fine validate vs stored 224
        coarse = mae[ct][fr] if ct in mae else float("nan")
        st = stored_small(args.data_folder, args.track, ct, 224)
        if st is not None:
            cand = np.asarray(native.resize((224, 224), Image.Resampling.LANCZOS), np.float32)
            fine = float(np.abs(cand - st).mean())
        else:
            fine = float("nan")
        flag = "" if (fine == fine and fine < args.mae_thresh) else "  <-- CHECK (high MAE)"
        print(f"idx={ct} -> mp4_frame={fr}  size={native.size}  coarseMAE={coarse:.2f}  "
              f"fineMAE(224)={fine:.2f}{flag}  wrote {out_png}", flush=True)
        written.append((ct, out_png))

    # optional: stitch the extracted frames (in index order) into a high-res obs-sequence
    # webp + gif. step_ms defaults to 250 (4 fps = the data's real-time playback).
    if args.stitch_to:
        os.makedirs(os.path.dirname(os.path.abspath(args.stitch_to)), exist_ok=True)
        pil = [Image.open(p).convert("RGB") for _, p in written]
        pil[0].save(args.stitch_to, format="WEBP", save_all=True, append_images=pil[1:],
                    duration=[args.step_ms] * len(pil), loop=0, lossless=True, method=4)
        print(f"stitched {len(pil)} frames -> {args.stitch_to}", flush=True)
        if args.stitch_to.endswith(".webp"):
            gif = args.stitch_to[:-5] + ".gif"
            if shutil.which("convert"):
                subprocess.run(["convert", "-coalesce", args.stitch_to, gif], check=True)
                print(f"converted -> {gif}", flush=True)
            else:
                print(f"[skip gif] ImageMagick 'convert' not found; webp written only", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--curr_times", type=int, nargs="+", required=True)
    p.add_argument("--cal_idxs", type=int, nargs="+", default=[50, 300, 800, 1500, 2500])
    p.add_argument("--vrs_root", default="/home/anw2067/scratch/temp_nymeria_large_videos")
    p.add_argument("--data_folder", default="/scratch/anw2067/nymeria_visibility_matrix")
    p.add_argument("--out", default="/home/anw2067/visualnav-transformer/train/logs/highres_frames")
    p.add_argument("--small", type=int, default=64)
    p.add_argument("--mae_thresh", type=float, default=12.0)
    p.add_argument("--stitch_to", default=None,
                   help="if set, write an obs-sequence webp (+gif) of the curr_times in index order")
    p.add_argument("--step_ms", type=int, default=250, help="per-frame duration in the stitched webp")
    main(p.parse_args())
