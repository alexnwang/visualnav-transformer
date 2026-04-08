import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these to change which runs are plotted
# ============================================================

RUNS = {
    "waypoint_point3d": "/home/anw2067/visualnav-transformer/train/logs/cem/2026_03_27_17_19_17:waypoint_point3d_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    "waypoint": "/home/anw2067/visualnav-transformer/train/logs/cem/2026_03_27_17_14_39:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    "peva": "/home/anw2067/visualnav-transformer/train/logs/cem/2026_03_28_00_29_25:peva_cem-h8-n8-t2-v0.05-o6-N64-ds64-dist8-8",
}

METRIC = "all_xyz"
CUMULATIVE_MIN = True  # If True, each step's value is min(step_0, ..., step_i) per task
OUTPUT_PATH = "train/logs/paper_vis/graphs/mje_vs_cem_iterations.pdf"

# ============================================================


def load_mean_curve(run_dir: str, metric: str, cumulative_min: bool = False) -> np.ndarray:
    """Load accum_eval_metric_dicts.pth and return mean metric across tasks per CEM step.
    Prepends the all_xyz_init value as step 0 (no planning baseline)."""
    path = Path(run_dir) / "accum_eval_metric_dicts.pth"
    data = torch.load(path, map_location="cpu", weights_only=False)

    task_dicts = torch.load(Path(run_dir) / "accum_task_dicts.pth", map_location="cpu", weights_only=False)
    task_keys = list(data.keys())
    init_vals = np.array([task_dicts[k]["task"]["all_xyz_init"] for k in task_keys])  # (num_tasks,)

    curves = np.array([data[k][metric] for k in task_keys])  # (num_tasks, num_steps)
    curves = np.concatenate([init_vals[:, None], curves], axis=1)  # prepend init as step 0
    if cumulative_min:
        curves = np.minimum.accumulate(curves, axis=1)
    return curves.mean(axis=0)


def main():
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, run_dir in RUNS.items():
        mean_curve = load_mean_curve(run_dir, METRIC, cumulative_min=CUMULATIVE_MIN)
        steps = np.arange(len(mean_curve))
        ax.plot(steps, mean_curve, marker="o", label=label)

    ax.set_xlabel("CEM Iterations")
    ax.set_ylabel("Mean Joint Error (m)")
    # ax.set_title(f"Mean Joint Error vs. CEM Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
