import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these to change which runs are plotted
# ============================================================

LOG_ROOT = "/home/anw2067/visualnav-transformer/train/logs/cem"

RUNS = {
    "waypoint_point3d": f"{LOG_ROOT}/2026_03_27_17_19_17:waypoint_point3d_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    "waypoint": f"{LOG_ROOT}/2026_03_27_17_14_39:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    "peva": f"{LOG_ROOT}/2026_03_28_00_29_25:peva_cem-h8-n8-t2-v0.05-o6-N64-ds64-dist8-8",
}

# o0 runs to pull dreamsim_init from (matched by algorithm type)
INIT_RUNS = {
    "waypoint_point3d": f"{LOG_ROOT}/2026_04_02_02_49_27:waypoint_cem-h1-n1-t1-v0.3-o0-N64-ds64-dist8-8",
    "waypoint": f"{LOG_ROOT}/2026_04_02_02_49_27:waypoint_cem-h1-n1-t1-v0.3-o0-N64-ds64-dist8-8",
    "peva": f"{LOG_ROOT}/2026_04_01_04_21_38:peva_cem-h8-n1-t1-v0.05-o0-N64-ds64-dist8-8",
}

METRIC = "dreamsim"
CUMULATIVE_MIN = True
OUTPUT_PATH = "train/logs/paper_vis/graphs/dreamsim_vs_cem_iterations.pdf"

# ============================================================


def load_mean_curve(run_dir: str, init_run_dir: str, metric: str, cumulative_min: bool = False) -> np.ndarray:
    """Load dreamsim curve prepended with dreamsim_init from the o0 run.
    Only uses tasks present in both the main run and the o0 run."""
    eval_data = torch.load(Path(run_dir) / "accum_eval_metric_dicts.pth", map_location="cpu", weights_only=False)
    init_data = torch.load(Path(init_run_dir) / "accum_task_dicts.pth", map_location="cpu", weights_only=False)

    shared_tasks = sorted(set(eval_data.keys()) & set(init_data.keys()))
    init_vals = np.array([init_data[k]["task"]["dreamsim_init"] for k in shared_tasks])
    curves = np.array([eval_data[k][metric] for k in shared_tasks])  # (num_tasks, num_steps)
    curves = np.concatenate([init_vals[:, None], curves], axis=1)
    if cumulative_min:
        curves = np.minimum.accumulate(curves, axis=1)
    return curves.mean(axis=0)


def main():
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, run_dir in RUNS.items():
        mean_curve = load_mean_curve(run_dir, INIT_RUNS[label], METRIC, cumulative_min=CUMULATIVE_MIN)
        steps = np.arange(len(mean_curve))
        ax.plot(steps, mean_curve, marker="o", label=label)

    ax.set_xlabel("CEM Iterations")
    ax.set_ylabel("DreamSIM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
