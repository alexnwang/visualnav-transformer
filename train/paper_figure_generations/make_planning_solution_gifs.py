"""Make 2x4 animated WebPs from cem_viz task dirs (planning solutions).

Layout per frame (2 rows x 4 cols):
    (0,0) curr_obs (static)                     gt_action_obs_seq_skin.png  row1 col 0
    (0,1) curr_obs + GT skel+skin (animated)    gt_action_obs_seq_skin.png  row0 col 1..T
    (0,2) GT observations (animated)            gt_action_obs_seq_skin.png  row1 col 1..T
    (0,3) black (static)
    (1,0) curr_obs + planning wps (static)      best_wp/rollout_0_*/action_obs_seq.png row0 col 0
    (1,1) curr_obs + planned skel+skin (anim)   best_wp/rollout_0_*/action_obs_seq.png row0 col 1..T
    (1,2) WM observations (animated)            best_wp/rollout_0_*/action_obs_seq.png row1 col 1..T
    (1,3) goal observation (static)             goal_obs.png

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

PANEL_LABELS = [
    ["Current observation",                       "Ground truth actions", "Ground truth observations", ""],
    ["Current observation\n+ planning waypoints", "Planning actions",     "Planning observations",     "Goal observation"],
]


@lru_cache(maxsize=64)
def render_label(text: str, fontsize: float = 10.0, dpi: int = 100,
                 stroke: float = 1.2) -> Image.Image:
    """Render text (mathtext supported) to a tight transparent RGBA PIL image."""
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
    return n_cols - 2


def find_rollout_dirs(task_dir: str, which: str) -> list[str]:
    """which in {'best_wp','best_mje'} — return all rollout_*/ under that subdir, sorted."""
    parent = os.path.join(task_dir, which)
    if not os.path.isdir(parent):
        return []
    return sorted(glob(os.path.join(parent, "rollout_*")))


def _save_webp(rgb_frames, out_path, fps, hold_first_last_ms):
    base_dur = int(round(1000.0 / fps))
    durations = [base_dur] * len(rgb_frames)
    if hold_first_last_ms > 0 and len(rgb_frames) >= 2:
        durations[0] = hold_first_last_ms
        durations[-1] = hold_first_last_ms
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgb_frames[0].save(
        out_path, format="WEBP", save_all=True, append_images=rgb_frames[1:],
        duration=durations, loop=0, lossless=True, quality=100, method=6,
    )


def make_gif_for_rollout(task_dir: str, rollout_dir: str,
                         out_path_with_gt: str, out_path_no_gt: str,
                         fps: float = 3.0, inner_pad: int = 2,
                         hold_first_last_ms: int = 0):
    gt_path = os.path.join(task_dir, "gt_action_obs_seq_skin.png")
    plan_path = os.path.join(rollout_dir, "action_obs_seq.png")
    goal_obs_path = os.path.join(task_dir, "goal_obs.png")
    if not (os.path.exists(gt_path) and os.path.exists(plan_path)
            and os.path.exists(goal_obs_path)):
        return False

    gt_img   = Image.open(gt_path).convert("RGB")
    plan_img = Image.open(plan_path).convert("RGB")
    goal_obs = Image.open(goal_obs_path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
    T = min(num_inner_frames(gt_img), num_inner_frames(plan_img))

    n_cols = 4
    curr_obs        = crop_tile(gt_img,   1, 0)
    curr_obs_planwp = crop_tile(plan_img, 0, 0)
    black_tile      = Image.new("RGB", (TILE, TILE), (0, 0, 0))

    def col_x(c): return c * (TILE + inner_pad)
    def row_y(r): return r * (TILE + inner_pad)

    def paint_labels(frame, rows):
        """rows is iterable of (row_idx_in_frame, row_idx_in_PANEL_LABELS)."""
        for frame_r, label_r in rows:
            for c in range(n_cols):
                txt = PANEL_LABELS[label_r][c]
                if not txt:
                    continue
                lbl = render_label(txt)
                frame.paste(lbl, (col_x(c) + 6, row_y(frame_r) + 4), lbl)

    # --- Variant A: 2x4 with GT on top ---
    panel_w_2 = n_cols * TILE + (n_cols - 1) * inner_pad
    panel_h_2 = 2 * TILE + inner_pad
    frames_with_gt = []
    for t in range(T):
        col = t + 1
        gt_skin   = crop_tile(gt_img,   0, col)
        gt_obs    = crop_tile(gt_img,   1, col)
        plan_skin = crop_tile(plan_img, 0, col)
        wm_obs    = crop_tile(plan_img, 1, col)

        frame = Image.new("RGB", (panel_w_2, panel_h_2), (255, 255, 255))
        frame.paste(curr_obs,        (col_x(0), row_y(0)))
        frame.paste(gt_skin,         (col_x(1), row_y(0)))
        frame.paste(gt_obs,          (col_x(2), row_y(0)))
        frame.paste(black_tile,      (col_x(3), row_y(0)))
        frame.paste(curr_obs_planwp, (col_x(0), row_y(1)))
        frame.paste(plan_skin,       (col_x(1), row_y(1)))
        frame.paste(wm_obs,          (col_x(2), row_y(1)))
        frame.paste(goal_obs,        (col_x(3), row_y(1)))
        paint_labels(frame, [(0, 0), (1, 1)])
        frames_with_gt.append(frame)

    # --- Variant B: 1x4 planning-only (drop GT row) ---
    panel_h_1 = TILE
    frames_no_gt = []
    for t in range(T):
        col = t + 1
        plan_skin = crop_tile(plan_img, 0, col)
        wm_obs    = crop_tile(plan_img, 1, col)

        frame = Image.new("RGB", (panel_w_2, panel_h_1), (255, 255, 255))
        frame.paste(curr_obs_planwp, (col_x(0), row_y(0)))
        frame.paste(plan_skin,       (col_x(1), row_y(0)))
        frame.paste(wm_obs,          (col_x(2), row_y(0)))
        frame.paste(goal_obs,        (col_x(3), row_y(0)))
        paint_labels(frame, [(0, 1)])
        frames_no_gt.append(frame)

    _save_webp(frames_with_gt, out_path_with_gt, fps, hold_first_last_ms)
    _save_webp(frames_no_gt,   out_path_no_gt,   fps, hold_first_last_ms)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", type=str, nargs="+",
                   help="One or more cem_viz run dirs (each containing per-task subdirs).")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--hold_first_last_ms", type=int, default=0)
    p.add_argument("--which", choices=("best_wp", "best_mje", "both"), default="both",
                   help="Which planner rollout subdir(s) to pull from.")
    p.add_argument("--out_root", type=str,
                   default="/home/anw2067/visualnav-transformer/train/logs/paper_vis/cem_viz_gifs")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        run_dir = os.path.abspath(run_dir.rstrip("/"))
        run_name = os.path.basename(run_dir)
        out_dir = os.path.join(args.out_root, run_name)
        os.makedirs(out_dir, exist_ok=True)

        task_dirs = sorted(d for d in glob(os.path.join(run_dir, "*"))
                           if os.path.isdir(d) and not os.path.basename(d).startswith("_"))

        print(f"run: {run_name}")
        print(f"out: {out_dir}")
        print(f"tasks: {len(task_dirs)}")

        which_list = ["best_wp", "best_mje"] if args.which == "both" else [args.which]
        made, skipped = 0, 0
        for td in task_dirs:
            task_name = os.path.basename(td)
            for which in which_list:
                rollout_dirs = find_rollout_dirs(td, which)
                if not rollout_dirs:
                    print(f"  skip (no {which}/): {task_name}")
                    skipped += 1
                    continue
                for rd in rollout_dirs:
                    rollout_name = os.path.basename(rd)
                    sub_out_dir = os.path.join(out_dir, task_name, which)
                    out_with_gt = os.path.join(sub_out_dir, f"{rollout_name}__with_gt.webp")
                    out_no_gt   = os.path.join(sub_out_dir, f"{rollout_name}__no_gt.webp")
                    ok = make_gif_for_rollout(td, rd, out_with_gt, out_no_gt,
                                              fps=args.fps,
                                              hold_first_last_ms=args.hold_first_last_ms)
                    if ok:
                        print(f"  ok: {task_name}/{which}/{rollout_name}")
                        made += 1
                    else:
                        print(f"  skip (missing pngs): {task_name}/{which}/{rollout_name}")
                        skipped += 1
        print(f"done — made {made}, skipped {skipped}\n")


if __name__ == "__main__":
    main()
