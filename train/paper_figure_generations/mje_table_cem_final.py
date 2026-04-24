import csv
import numpy as np
from pathlib import Path

from mje_vs_cem_iterations import (
    RUNS,
    METHODS,
    N_VALUES,
    _load_dicts,
)

METRICS = ["intermediate_xyz", "leaf_xyz", "all_xyz"]
OUTPUT_CSV = "train/logs/paper_vis/tables/mje_table_cem_final.csv"


def cummin_final_means(run_dirs, metrics):
    data = _load_dicts(run_dirs, "accum_eval_metric_dicts.pth")
    task_keys = sorted(data.keys())
    out = {}
    for m in metrics:
        curves = np.array([data[k][m] for k in task_keys])  # (num_tasks, num_steps)
        cummin = np.minimum.accumulate(curves, axis=1)
        out[m] = float(cummin[:, -1].mean())
    return out, len(task_keys)


def main():
    rows = []
    for n in N_VALUES:
        for method in METHODS:
            key = (method, n)
            if key not in RUNS:
                continue
            vals, num_tasks = cummin_final_means(RUNS[key], METRICS)
            rows.append((method, n, num_tasks, vals))

    header = ["Method", "n", "num_tasks"] + METRICS
    widths = [max(len(str(r[0])) for r in rows + [(header[0],)]), 4, 10, 14, 10, 10]

    fmt_header = f"{header[0]:<{widths[0]}}  {header[1]:<{widths[1]}}  {header[2]:<{widths[2]}}  " \
                 f"{header[3]:<{widths[3]}}  {header[4]:<{widths[4]}}  {header[5]:<{widths[5]}}"
    print(fmt_header)
    print("-" * len(fmt_header))
    for method, n, num_tasks, vals in rows:
        print(f"{method:<{widths[0]}}  {n:<{widths[1]}}  {num_tasks:<{widths[2]}}  "
              f"{vals['intermediate_xyz']:<{widths[3]}.4f}  "
              f"{vals['leaf_xyz']:<{widths[4]}.4f}  "
              f"{vals['all_xyz']:<{widths[5]}.4f}")

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for method, n, num_tasks, vals in rows:
            w.writerow([method, n, num_tasks,
                        f"{vals['intermediate_xyz']:.6f}",
                        f"{vals['leaf_xyz']:.6f}",
                        f"{vals['all_xyz']:.6f}"])
    print(f"\nSaved CSV to {out_path}")


if __name__ == "__main__":
    main()
