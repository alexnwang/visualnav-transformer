import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these to change which runs are plotted
# ============================================================

LOG_ROOT = "/home/anw2067/visualnav-transformer/train/logs/cem"

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
CUMULATIVE_MIN = True
BUCKET_MODE = "quantile"  # "fixed" for evenly spaced, "quantile" for equal-count bins
BUCKET_WIDTH = 0.5  # meters (used when BUCKET_MODE == "fixed")
NUM_BUCKETS = 20     # (used when BUCKET_MODE == "quantile")
OUTPUT_PATH = "train/logs/paper_vis/graphs/mje_vs_init_distance.pdf"

# ============================================================


def load_all_tasks(dist_runs: dict, metric: str, cumulative_min: bool = False):
    """Load all tasks across horizon runs. Returns (init_values, final_values) arrays."""
    inits, finals = [], []
    for dist, run_dir in dist_runs.items():
        eval_data = torch.load(Path(run_dir) / "accum_eval_metric_dicts.pth", map_location="cpu", weights_only=False)
        task_data = torch.load(Path(run_dir) / "accum_task_dicts.pth", map_location="cpu", weights_only=False)
        for task_key in eval_data:
            curve = np.array(eval_data[task_key][metric])
            if cumulative_min:
                curve = np.minimum.accumulate(curve)
            finals.append(curve[-1])
            inits.append(task_data[task_key]["task"]["all_xyz_init"])
    return np.array(inits), np.array(finals)


def main():
    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute bucket edges
    first_inits, _ = load_all_tasks(next(iter(RUNS.values())), METRIC, CUMULATIVE_MIN)
    if BUCKET_MODE == "fixed":
        bin_start = np.floor(first_inits.min() / BUCKET_WIDTH) * BUCKET_WIDTH
        bin_end = np.ceil(first_inits.max() / BUCKET_WIDTH) * BUCKET_WIDTH
        bucket_edges = np.arange(bin_start, bin_end + BUCKET_WIDTH, BUCKET_WIDTH)
    else:  # quantile
        bucket_edges = np.quantile(first_inits, np.linspace(0, 1, NUM_BUCKETS + 1))
        bucket_edges[0] = -np.inf
        bucket_edges[-1] = np.inf
    num_buckets = len(bucket_edges) - 1

    for label, dist_runs in RUNS.items():
        inits, finals = load_all_tasks(dist_runs, METRIC, cumulative_min=CUMULATIVE_MIN)
        centers, means, sems = [], [], []
        for i in range(num_buckets):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (inits >= lo) & (inits < hi if i < num_buckets - 1 else inits <= hi)
            if mask.sum() == 0:
                continue
            centers.append(0.5 * (lo + hi))
            bucket_vals = finals[mask]
            means.append(bucket_vals.mean())
            sems.append(bucket_vals.std() / np.sqrt(len(bucket_vals)))
        centers, means, sems = np.array(centers), np.array(means), np.array(sems)
        line, = ax.plot(centers, means, marker="o", label=label, markersize=4)
        ax.fill_between(centers, means - sems, means + sems, color=line.get_color(), alpha=0.3)

    # Identity line: y = x (no improvement from planning)
    xlims = ax.get_xlim()
    line_x = np.linspace(xlims[0], xlims[1], 50)
    ax.plot(line_x, line_x, linestyle="--", color="gray", alpha=0.5, label="Initial MJE")

    ax.set_xlabel("Initial Mean Joint Error (m)")
    ax.set_ylabel("Mean Joint Error (m)")
    # ax.set_title(f"{METRIC} vs. Initial Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
