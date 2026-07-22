"""
Compare evaluation results across model versions.

Usage:
    python experiments/compare_eval.py \
        --dirs results_eval/base_pi05 results_eval/ft_dagger_r0

Reads all *_agg.csv files in each directory and prints a comparison table
grouped by suite/level/mode, with model versions as columns.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from collections import defaultdict


METRICS = ["car_pct", "tsr_pct", "collision_rate_pct", "ets", "mean_cbf_activation_rate"]
METRIC_LABELS = {
    "car_pct":                 "CAR ↑",
    "tsr_pct":                 "TSR ↑",
    "collision_rate_pct":      "Coll ↓",
    "ets":                     "ETS ↓",
    "mean_cbf_activation_rate":"CBF Rate ↓",
}


def load_dir(d: Path) -> dict[str, dict]:
    """Load all agg CSVs in a directory. Returns {scene_mode_key: row_dict}."""
    results = {}
    for csv_path in sorted(d.glob("*_agg.csv")):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            key = f"{row['scene']}_{row['mode']}"
            results[key] = row
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True,
                   help="Result directories to compare (one per model version)")
    p.add_argument("--metrics", nargs="+", default=METRICS,
                   help="Metrics to display")
    args = p.parse_args()

    dirs = [Path(d) for d in args.dirs]
    model_tags = [d.name for d in dirs]
    data = {tag: load_dir(d) for tag, d in zip(model_tags, dirs)}

    # Collect all scene×mode keys
    all_keys = sorted(set(k for d in data.values() for k in d))

    col_w = 10
    tag_w = max(len(t) for t in model_tags)
    key_w = max(len(k) for k in all_keys) if all_keys else 40

    for metric in args.metrics:
        label = METRIC_LABELS.get(metric, metric)
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        header = f"  {'Scene / Mode':<{key_w}}"
        for tag in model_tags:
            header += f"  {tag:>{col_w}}"
        print(header)
        print(f"  {'-'*key_w}" + f"  {'-'*col_w}" * len(model_tags))

        for key in all_keys:
            row_str = f"  {key:<{key_w}}"
            for tag in model_tags:
                val = data[tag].get(key, {}).get(metric, "N/A")
                try:
                    row_str += f"  {float(val):>{col_w}.2f}"
                except (ValueError, TypeError):
                    row_str += f"  {'N/A':>{col_w}}"
            print(row_str)

    # Summary averages
    print(f"\n{'='*80}")
    print("  AVERAGES")
    print(f"{'='*80}")
    header = f"  {'Metric':<30}"
    for tag in model_tags:
        header += f"  {tag:>{col_w}}"
    print(header)
    print(f"  {'-'*30}" + f"  {'-'*col_w}" * len(model_tags))

    for metric in args.metrics:
        label = METRIC_LABELS.get(metric, metric)
        row_str = f"  {label:<30}"
        for tag in model_tags:
            vals = []
            for key in all_keys:
                v = data[tag].get(key, {}).get(metric)
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            avg = sum(vals) / len(vals) if vals else None
            row_str += f"  {avg:>{col_w}.2f}" if avg is not None else f"  {'N/A':>{col_w}}"
        print(row_str)
    print()


if __name__ == "__main__":
    main()
