import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these to change which runs are plotted
# ============================================================

LOG_ROOT = "/home/anw2067/visualnav-transformer/train/logs/cem"

METHODS = ["Lifted CEM(3D)", "Lifted CEM", "PEVA CEM"]
N_VALUES = [8, 16, 64]
MARKERS = {8: "o", 16: "s", 64: "D"}

RUNS = {
    # n=8
    ("Lifted CEM(3D)", 8): f"{LOG_ROOT}/2026_03_27_17_19_17:waypoint_point3d_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    ("Lifted CEM", 8): f"{LOG_ROOT}/2026_03_27_17_14_39:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    ("PEVA CEM", 8): f"{LOG_ROOT}/2026_03_28_00_29_25:peva_cem-h8-n8-t2-v0.05-o6-N64-ds64-dist8-8",
    # n=16
    ("Lifted CEM(3D)", 16): f"{LOG_ROOT}/2026_04_14_07_23_53:waypoint_point3d_cem-h1-n16-t4-v0.3-o6-N64-ds64-dist8-8",
    ("Lifted CEM", 16): f"{LOG_ROOT}/2026_04_14_07_23_53:waypoint_cem-h1-n16-t4-v0.3-o6-N64-ds64-dist8-8",
    ("PEVA CEM", 16): f"{LOG_ROOT}/2026_04_13_12_49_39:peva_cem-h8-n16-t2-v0.05-o6-N64-ds64-dist8-8",
    # n=64 (two ranks each, combined)
    ("Lifted CEM(3D)", 64): [
        f"{LOG_ROOT}/2026_04_14_07_25_13:waypoint_point3d_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-0:2",
        f"{LOG_ROOT}/2026_04_14_07_35_00:waypoint_point3d_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-1:2",
    ],
    ("Lifted CEM", 64): [
        f"{LOG_ROOT}/2026_04_13_12_51_51:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-0:2",
        f"{LOG_ROOT}/2026_04_13_12_53_43:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-1:2",
    ],
    ("PEVA CEM", 64): [
        f"{LOG_ROOT}/2026_04_13_12_51_51:peva_cem-h8-n64-t2-v0.05-o6-N64-ds64-dist8-8-rank:ws-0:2",
        f"{LOG_ROOT}/2026_04_14_07_53_38:peva_cem-h8-n64-t2-v0.05-o6-N64-ds64-dist8-8-rank:ws-1:2",
    ],
}

# o0 runs to pull dreamsim_init from (matched by algorithm type)
INIT_RUNS = {
    "waypoint_point3d": f"{LOG_ROOT}/2026_04_02_02_49_27:waypoint_cem-h1-n1-t1-v0.3-o0-N64-ds64-dist8-8",
    "waypoint": f"{LOG_ROOT}/2026_04_02_02_49_27:waypoint_cem-h1-n1-t1-v0.3-o0-N64-ds64-dist8-8",
    "peva": f"{LOG_ROOT}/2026_04_01_04_21_38:peva_cem-h8-n1-t1-v0.05-o0-N64-ds64-dist8-8",
}

# Map each run label to its algorithm type for init lookup
def _algo_type(label):
    if "PEVA" in label:
        return "peva"
    elif "3D" in label:
        return "waypoint_point3d"
    else:
        return "waypoint"

METRIC = "dreamsim"
CUMULATIVE_MIN = True
SHOW_SEM = False
_base = "train/logs/paper_vis/graphs/dreamsim_vs_cem_iterations"
_suffixes = ("" if CUMULATIVE_MIN else "_nocummin") + ("_sem" if SHOW_SEM else "")
OUTPUT_PATH = f"{_base}{_suffixes}.pdf"

# ============================================================


def _load_dicts(run_dirs, filename):
    """Load and merge .pth dicts from one or more run directories."""
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    merged = {}
    for d in run_dirs:
        data = torch.load(Path(d) / filename, map_location="cpu", weights_only=False)
        merged.update(data)
    return merged


def load_mean_curve(run_dirs, init_run_dir: str, metric: str, cumulative_min: bool = False) -> np.ndarray:
    """Load dreamsim curve prepended with dreamsim_init from the o0 run.
    Only uses tasks present in both the main run and the o0 run.
    run_dirs can be a single path string or a list of paths (for multi-rank runs)."""
    eval_data = _load_dicts(run_dirs, "accum_eval_metric_dicts.pth")
    init_data = torch.load(Path(init_run_dir) / "accum_task_dicts.pth", map_location="cpu", weights_only=False)

    shared_tasks = sorted(set(eval_data.keys()) & set(init_data.keys()))
    init_vals = np.array([init_data[k]["task"]["dreamsim_init"] for k in shared_tasks])
    curves = np.array([eval_data[k][metric] for k in shared_tasks])  # (num_tasks, num_steps)
    curves = np.concatenate([init_vals[:, None], curves], axis=1)
    if cumulative_min:
        curves = np.minimum.accumulate(curves, axis=1)
    means = curves.mean(axis=0)
    sems = curves.std(axis=0) / np.sqrt(len(curves))
    return means, sems


def main():
    fig, ax = plt.subplots(figsize=(6, 4))

    # Assign a consistent color per method
    cmap = plt.cm.tab10
    method_colors = {m: cmap(i) for i, m in enumerate(METHODS)}

    # Plot grouped by method, varying marker by n
    for method in METHODS:
        for n in N_VALUES:
            key = (method, n)
            if key not in RUNS:
                continue
            init_run_dir = INIT_RUNS[_algo_type(method)]
            means, sems = load_mean_curve(RUNS[key], init_run_dir, METRIC, cumulative_min=CUMULATIVE_MIN)
            steps = np.arange(len(means))
            color = method_colors[method]
            ax.plot(steps, means, marker=MARKERS[n], color=color,
                    markersize=5, markeredgecolor="black", markeredgewidth=0.5)
            if SHOW_SEM:
                ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.15)

    # Combined legend: method → color line, n → marker shape
    from matplotlib.lines import Line2D
    legend_handles, legend_labels = [], []
    for method in METHODS:
        legend_handles.append(Line2D([], [], color=method_colors[method], linewidth=2))
        legend_labels.append(method)
    for n in N_VALUES:
        legend_handles.append(Line2D([], [], color="black", marker=MARKERS[n], linestyle="None",
                                     markersize=5, markeredgecolor="black", markeredgewidth=0.5))
        legend_labels.append(f"n={n}")
    ax.legend(legend_handles, legend_labels, fontsize="small")

    ax.set_xlabel("CEM Iterations")
    ax.set_ylabel("DreamSIM")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
