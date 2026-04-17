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
    "Lifted CEM": {
        6:  f"{LOG_ROOT}/2026_04_09_12_38_59:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist6-6",
        8:  f"{LOG_ROOT}/2026_03_27_02_19_14:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
        10: f"{LOG_ROOT}/2026_04_08_17_06_27:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist10-10",
        12: f"{LOG_ROOT}/2026_03_27_02_19_14:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist12-12",
        14: f"{LOG_ROOT}/2026_04_08_17_37_16:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist14-14",
        16: f"{LOG_ROOT}/2026_03_27_02_33_19:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist16-16",
        18: f"{LOG_ROOT}/2026_04_08_22_34_55:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist18-18",
        20: f"{LOG_ROOT}/2026_03_27_02_35_26:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist20-20",
    },
    "PEVA CEM": {
        6:  f"{LOG_ROOT}/2026_04_09_16_05_29:peva_cem-h6-n8-t2-v0.05-o6-N64-ds64-dist6-6",
        8:  f"{LOG_ROOT}/2026_03_28_00_29_25:peva_cem-h8-n8-t2-v0.05-o6-N64-ds64-dist8-8",
        10: f"{LOG_ROOT}/2026_04_08_17_25_34:peva_cem-h10-n8-t2-v0.05-o6-N64-ds64-dist10-10",
        12: f"{LOG_ROOT}/2026_03_29_23_49_18:peva_cem-h12-n8-t2-v0.05-o6-N64-ds64-dist12-12",
        14: f"{LOG_ROOT}/2026_04_08_22_31_56:peva_cem-h14-n8-t2-v0.05-o6-N64-ds64-dist14-14",
        16: f"{LOG_ROOT}/2026_03_30_10_19_22:peva_cem-h16-n8-t2-v0.05-o6-N64-ds64-dist16-16",
        18: f"{LOG_ROOT}/2026_04_08_23_30_51:peva_cem-h18-n8-t2-v0.05-o6-N64-ds64-dist18-18",
        20: f"{LOG_ROOT}/2026_03_31_16_07_19:peva_cem-h20-n8-t2-v0.05-o6-N64-ds64-dist20-20",
    },
}

METRIC = "all_xyz"
CUMULATIVE_MIN = False  # If True, take best value across all CEM steps per task
OUTPUT_PATH = "/home/anw2067/visualnav-transformer/train/logs/paper_vis/graphs/mje_vs_horizon.pdf" if CUMULATIVE_MIN else "/home/anw2067/visualnav-transformer/train/logs/paper_vis/graphs/mje_vs_horizon_nocummin.pdf"

# ============================================================


def load_final_mean_sem(run_dir: str, metric: str, cumulative_min: bool = False) -> tuple[float, float]:
    """Load accum_eval_metric_dicts.pth and return (mean, SEM) of the final-step metric across tasks."""
    path = Path(run_dir) / "accum_eval_metric_dicts.pth"
    data = torch.load(path, map_location="cpu", weights_only=False)
    curves = np.array([v[metric] for v in data.values()])  # (num_tasks, num_steps)
    if cumulative_min:
        curves = np.minimum.accumulate(curves, axis=1)
    final = curves[:, -1]
    return final.mean(), final.std() / np.sqrt(len(final))


def load_init_mean_sem(run_dir: str) -> tuple[float, float]:
    """Load accum_task_dicts.pth and return (mean, SEM) of all_xyz_init across tasks."""
    path = Path(run_dir) / "accum_task_dicts.pth"
    data = torch.load(path, map_location="cpu", weights_only=False)
    vals = np.array([v["task"]["all_xyz_init"] for v in data.values()])
    return vals.mean(), vals.std() / np.sqrt(len(vals))


def main():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(x=8, color="black", linestyle="-", alpha=0.5, label="Training timesteps")

    # Use the first curve's runs to compute init baseline (same tasks across runs)
    first_runs = next(iter(RUNS.values()))
    init_dists = sorted(first_runs.keys())
    init_means, init_sems = zip(*[load_init_mean_sem(first_runs[d]) for d in init_dists])
    init_means, init_sems = np.array(init_means), np.array(init_sems)
    ax.plot(init_dists, init_means, marker="s", linestyle="--", color="gray", label="Initial MJE")
    ax.fill_between(init_dists, init_means - init_sems, init_means + init_sems, color="gray", alpha=0.3)

    for label, dist_runs in RUNS.items():
        dists = sorted(dist_runs.keys())
        means, sems = zip(*[load_final_mean_sem(dist_runs[d], METRIC, cumulative_min=CUMULATIVE_MIN) for d in dists])
        means, sems = np.array(means), np.array(sems)
        label = "Initial MJE" if "no planning" in label else label
        line, = ax.plot(dists, means, marker="o", label=label)
        ax.fill_between(dists, means - sems, means + sems, color=line.get_color(), alpha=0.3)

    ax.set_xlabel("Goal Distance (timesteps)")
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
