"""Make 2x3 animated WebPs from policy_rollouts task dirs.

Layout per frame (2 rows x 3 cols):
    (0,0) curr_obs (static)             (gt_rollout_skin.png  row1 col 0)
    (0,1) curr_obs + GT skel+skin       (gt_rollout_skin.png  row0 col 1..T)
    (0,2) GT observation frames         (gt_rollout_skin.png  row1 col 1..T)
    (1,0) curr_obs + waypoints (static) (gt_rollout_skin.png  row0 col 0)
    (1,1) curr_obs + pred skel+skin     (pred_wm_rollout_skin.png row0 col 1..T)
    (1,2) world-model frames            (pred_wm_rollout_skin.png row1 col 1..T)

Source PNGs are torchvision-grid tiled with pad=2, tile=224, T+2 columns × 2 rows.
"""
import argparse
import io
import os
from functools import lru_cache
from glob import glob

import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image

TILE = 224
PAD = 2

# Labels support matplotlib mathtext: $...$ for math, e.g. r"$x_t$", r"$\hat{a}_{1:H}$".
PANEL_LABELS = [
    ["Current observation",                       "Ground truth actions",   "Ground truth observations"],
    ["Input\nCurrent observation\n+ waypoints",   "Generated actions",      "Generated observations"],
]


@lru_cache(maxsize=64)
def render_label(text: str, fontsize: float = 10.0, dpi: int = 100,
                 stroke: float = 1.2) -> Image.Image:
    """Render text (mathtext supported) to a tight transparent RGBA PIL image.

    Pixel height ≈ fontsize * dpi / 72. Defaults give ~11 px tall text.
    """
    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    fig.patch.set_alpha(0)
    fig.text(
        0, 0, text, fontsize=fontsize, color="white",
        va="bottom", ha="left",
        path_effects=[patheffects.withStroke(linewidth=stroke, foreground="black")],
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").copy()


def crop_tile(img: Image.Image, row: int, col: int) -> Image.Image:
    x = PAD + col * (TILE + PAD)
    y = PAD + row * (TILE + PAD)
    return img.crop((x, y, x + TILE, y + TILE))


def num_inner_frames(img: Image.Image) -> int:
    n_cols = (img.width - PAD) // (TILE + PAD)
    return n_cols - 2  # subtract goal_image col + goal_obs col


def make_gif_for_task(task_dir: str, out_path: str, fps: float = 3.0,
                      inner_pad: int = 2, hold_first_last_ms: int = 0):
    gt_path = os.path.join(task_dir, "gt_rollout_skin.png")
    wm_path = os.path.join(task_dir, "pred_wm_rollout_skin.png")
    if not (os.path.exists(gt_path) and os.path.exists(wm_path)):
        return False

    gt_img = Image.open(gt_path).convert("RGB")
    wm_img = Image.open(wm_path).convert("RGB")
    T = min(num_inner_frames(gt_img), num_inner_frames(wm_img))

    n_cols, n_rows = 3, 2
    panel_w = n_cols * TILE + (n_cols - 1) * inner_pad
    panel_h = n_rows * TILE + (n_rows - 1) * inner_pad

    # Static left-column tiles (constant across all frames).
    curr_obs       = crop_tile(gt_img, 1, 0)  # bottom-row col 0 = curr_obs
    curr_obs_wp    = crop_tile(gt_img, 0, 0)  # top-row    col 0 = curr_obs + waypoints

    def col_x(c):
        return c * (TILE + inner_pad)

    def row_y(r):
        return r * (TILE + inner_pad)

    rgb_frames = []
    for t in range(T):
        col = t + 1  # skip leading goal_image column in source
        gt_skin    = crop_tile(gt_img, 0, col)
        gt_obs     = crop_tile(gt_img, 1, col)
        pred_skin  = crop_tile(wm_img, 0, col)
        wm_obs     = crop_tile(wm_img, 1, col)

        frame = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        frame.paste(curr_obs,    (col_x(0), row_y(0)))
        frame.paste(gt_skin,     (col_x(1), row_y(0)))
        frame.paste(gt_obs,      (col_x(2), row_y(0)))
        frame.paste(curr_obs_wp, (col_x(0), row_y(1)))
        frame.paste(pred_skin,   (col_x(1), row_y(1)))
        frame.paste(wm_obs,      (col_x(2), row_y(1)))

        for r in range(n_rows):
            for c in range(n_cols):
                lbl = render_label(PANEL_LABELS[r][c])
                x = col_x(c) + 6
                y = row_y(r) + 4
                frame.paste(lbl, (x, y), lbl)
        rgb_frames.append(frame)

    base_dur = int(round(1000.0 / fps))
    durations = [base_dur] * len(rgb_frames)
    if hold_first_last_ms > 0 and len(rgb_frames) >= 2:
        durations[0] = hold_first_last_ms
        durations[-1] = hold_first_last_ms

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Animated WebP, lossless — keeps the skeleton-leaf dot colors exact.
    rgb_frames[0].save(
        out_path, format="WEBP", save_all=True, append_images=rgb_frames[1:],
        duration=durations, loop=0, lossless=True, quality=100, method=6,
    )
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str,
                   help="Path to a single run dir under policy_rollouts/, e.g. "
                        "logs/paper_vis/policy_rollouts/2026_04_26_23_01_33:...")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--hold_first_last_ms", type=int, default=0,
                   help="Optional extra hold (ms) on first and last frames.")
    p.add_argument("--out_root", type=str,
                   default="/home/anw2067/visualnav-transformer/train/logs/paper_vis/policy_rollouts_gifs")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir.rstrip("/"))
    run_name = os.path.basename(run_dir)
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    task_dirs = sorted(d for d in glob(os.path.join(run_dir, "*")) if os.path.isdir(d))
    print(f"run: {run_name}")
    print(f"out: {out_dir}")
    print(f"tasks: {len(task_dirs)}")

    made, skipped = 0, 0
    for td in task_dirs:
        task_name = os.path.basename(td)
        out_path = os.path.join(out_dir, f"{task_name}.webp")
        ok = make_gif_for_task(td, out_path, fps=args.fps,
                               hold_first_last_ms=args.hold_first_last_ms)
        if ok:
            print(f"  ok: {task_name}.gif")
            made += 1
        else:
            print(f"  skip (missing pngs): {task_name}")
            skipped += 1
    print(f"done — made {made}, skipped {skipped}")


if __name__ == "__main__":
    main()
