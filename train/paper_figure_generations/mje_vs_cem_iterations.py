import argparse
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
    ("Lifted CEM(3D)", 8): f"{LOG_ROOT}/2026_04_15_19_35_50:waypoint_point3d_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    ("Lifted CEM", 8): [
        f"{LOG_ROOT}/2026_04_15_19_35_26:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
        # f"{LOG_ROOT}/2026_04_19_01_17_06:waypoint_cem-h1-n8-t4-v0.3-o6-N64-ds64-dist8-8",
    ],
    ("PEVA CEM", 8): f"{LOG_ROOT}/2026_04_15_19_36_50:peva_cem-h8-n8-t2-v0.05-o6-N64-ds64-dist8-8",
    # n=16
    ("Lifted CEM(3D)", 16): f"{LOG_ROOT}/2026_04_15_20_15_01:waypoint_point3d_cem-h1-n16-t4-v0.3-o6-N64-ds64-dist8-8",
    ("Lifted CEM", 16): [
        f"{LOG_ROOT}/2026_04_15_20_08_54:waypoint_cem-h1-n16-t4-v0.3-o6-N64-ds64-dist8-8",
        # f"{LOG_ROOT}/2026_04_19_01_24_00:waypoint_cem-h1-n16-t4-v0.3-o6-N64-ds64-dist8-8",
    ],
    ("PEVA CEM", 16): f"{LOG_ROOT}/2026_04_18_10_49_07:COMBINED-peva_cem-h8-n16-t2-v0.05-o6-N64-ds64-dist8-8",
    # n=64 (four ranks each, combined)
    ("Lifted CEM(3D)", 64): [
        f"{LOG_ROOT}/2026_04_15_20_54_34:waypoint_point3d_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-0:4",
        f"{LOG_ROOT}/2026_04_15_21_14_10:waypoint_point3d_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-1:4",
        f"{LOG_ROOT}/2026_04_16_19_17_27:waypoint_point3d_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-2:4",
        f"{LOG_ROOT}/2026_04_17_12_09_59:waypoint_point3d_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-3:4",
    ],
    ("Lifted CEM", 64): [
        f"{LOG_ROOT}/2026_04_15_20_18_33:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-0:4",
        f"{LOG_ROOT}/2026_04_15_21_06_50:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-1:4",
        f"{LOG_ROOT}/2026_04_15_21_23_35:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-2:4",
        f"{LOG_ROOT}/2026_04_17_12_10_00:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-3:4",
        # f"{LOG_ROOT}/2026_04_19_02_06_14:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-0:2",
        # f"{LOG_ROOT}/2026_04_19_02_41_35:waypoint_cem-h1-n64-t4-v0.3-o6-N64-ds64-dist8-8-rank:ws-1:2",
    ],
    ("PEVA CEM", 64): [
        f"{LOG_ROOT}/2026_04_15_21_05_52:peva_cem-h8-n64-t2-v0.05-o6-N64-ds64-dist8-8-rank:ws-0:4",
        f"{LOG_ROOT}/2026_04_15_21_15_06:peva_cem-h8-n64-t2-v0.05-o6-N64-ds64-dist8-8-rank:ws-1:4",
        f"{LOG_ROOT}/2026_04_16_20_19_27:peva_cem-h8-n64-t2-v0.05-o6-N64-ds64-dist8-8-rank:ws-2:4",
        f"{LOG_ROOT}/2026_04_17_12_09_55:peva_cem-h8-n64-t2-v0.05-o6-N64-ds64-dist8-8-rank:ws-3:4",
    ],
}

METRIC = "all_xyz"
CUMULATIVE_MIN = True  # If True, each step's value is min(step_0, ..., step_i) per task
SHOW_SEM = True
OUTPUT_BASE = "train/logs/figures/paper/paper_vis/graphs/mje_vs_cem_iterations"

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


def load_mean_curve(run_dirs, metric: str, cumulative_min: bool = False) -> np.ndarray:
    """Load accum_eval_metric_dicts.pth and return mean metric across tasks per CEM step.
    Prepends the all_xyz_init value as step 0 (no planning baseline). cumulative_min
    is applied only across CEM steps — the init step is shown as-is and excluded from the min.
    run_dirs can be a single path string or a list of paths (for multi-rank runs)."""
    data = _load_dicts(run_dirs, "accum_eval_metric_dicts.pth")
    task_dicts = _load_dicts(run_dirs, "accum_task_dicts.pth")

    task_keys = sorted(data.keys())
    init_vals = np.array([task_dicts[k]["task"]["all_xyz_init"] for k in task_keys])  # (num_tasks,)

    cem_curves = np.array([data[k][metric] for k in task_keys])  # (num_tasks, num_steps)
    if cumulative_min:
        cem_curves = np.minimum.accumulate(cem_curves, axis=1)
    curves = np.concatenate([init_vals[:, None], cem_curves], axis=1)  # prepend init as step 0
    means = curves.mean(axis=0)
    sems = curves.std(axis=0) / np.sqrt(len(curves))
    return means, sems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cummin", action=argparse.BooleanOptionalAction, default=CUMULATIVE_MIN,
                        help="If set, each step's value is min(step_0, ..., step_i) per task.")
    parser.add_argument("--sem", action=argparse.BooleanOptionalAction, default=SHOW_SEM,
                        help="If set, shade SEM band around each curve.")
    args = parser.parse_args()

    output_path = f"{OUTPUT_BASE}{'' if args.cummin else '_nocummin'}{'_sem' if args.sem else ''}.pdf"

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
            means, sems = load_mean_curve(RUNS[key], METRIC, cumulative_min=args.cummin)
            steps = np.arange(len(means))
            color = method_colors[method]
            ax.plot(steps, means, marker=MARKERS[n], color=color,
                    markersize=5, markeredgecolor="black", markeredgewidth=0.5)
            if args.sem:
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
    ax.set_ylabel("Mean Joint Error (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
