import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these to change which runs are plotted
# ============================================================

LOG_ROOT = "/home/anw2067/visualnav-transformer/train/logs/cem"

# Each curve: label -> {dist_value: run_dir_name}
RUNS = {
    "waypoint_cem": {
        8:  f"{LOG_ROOT}/2026_03_27_02_19_14:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
        12: f"{LOG_ROOT}/2026_03_27_02_19_14:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist12-12",
        16: f"{LOG_ROOT}/2026_03_27_02_33_19:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist16-16",
        20: f"{LOG_ROOT}/2026_03_27_02_35_26:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist20-20",
    },
    "peva_cem": {
        8:  f"{LOG_ROOT}/2026_03_28_00_29_25:peva_cem-h8-n8-t2-v0.05-o6-N64-ds64-dist8-8",
        12: f"{LOG_ROOT}/2026_03_29_23_49_18:peva_cem-h12-n8-t2-v0.05-o6-N64-ds64-dist12-12",
        16: f"{LOG_ROOT}/2026_03_30_10_19_22:peva_cem-h16-n8-t2-v0.05-o6-N64-ds64-dist16-16",
        20: f"{LOG_ROOT}/2026_03_31_16_07_19:peva_cem-h20-n8-t2-v0.05-o6-N64-ds64-dist20-20",
    },
}

METRIC = "all_xyz"
CUMULATIVE_MIN = True  # If True, take best value across all CEM steps per task
OUTPUT_PATH = "train/logs/paper_vis/graphs/mje_vs_horizon.pdf"

# ============================================================


def load_final_mean(run_dir: str, metric: str, cumulative_min: bool = False) -> float:
    """Load accum_eval_metric_dicts.pth and return mean of the final-step metric across tasks."""
    path = Path(run_dir) / "accum_eval_metric_dicts.pth"
    data = torch.load(path, map_location="cpu", weights_only=False)
    curves = np.array([v[metric] for v in data.values()])  # (num_tasks, num_steps)
    if cumulative_min:
        curves = np.minimum.accumulate(curves, axis=1)
    return curves[:, -1].mean()


def load_init_mean(run_dir: str) -> float:
    """Load accum_task_dicts.pth and return mean all_xyz_init across tasks."""
    path = Path(run_dir) / "accum_task_dicts.pth"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return np.mean([v["task"]["all_xyz_init"] for v in data.values()])


def main():
    fig, ax = plt.subplots(figsize=(6, 4))

    # Use the first curve's runs to compute init baseline (same tasks across runs)
    first_runs = next(iter(RUNS.values()))
    init_dists = sorted(first_runs.keys())
    init_values = [load_init_mean(first_runs[d]) for d in init_dists]
    ax.plot(init_dists, init_values, marker="s", linestyle="--", color="gray", label="Initial MJE")

    for label, dist_runs in RUNS.items():
        dists = sorted(dist_runs.keys())
        values = [load_final_mean(dist_runs[d], METRIC, cumulative_min=CUMULATIVE_MIN) for d in dists]
        label = "Initial MJE" if "no planning" in label else label
        ax.plot(dists, values, marker="o", label=label)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Joint Error (m)")
    # ax.set_title(f"Mean {METRIC} vs. Time Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
