#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


RUNS = [
    ("Baseline", "cnn_baseline"),
    ("TDCF cap90", "cnn_tdcf_cap90"),
    ("TDCF cap80", "cnn_tdcf_cap80"),
    ("TDCF cap100", "cnn_tdcf_cap100"),
    ("Fixed match90", "cnn_fixed_match90"),
    ("Random match90", "cnn_random_match90"),
]


def parse_args():
    p = argparse.ArgumentParser("summarize_cifar100_ablations")
    p.add_argument("--results_root", type=Path, default=Path("./results/cifar100_ablations"))
    p.add_argument("--write_markdown", action="store_true")
    return p.parse_args()


def load_run(results_path: Path) -> Optional[Dict[str, object]]:
    if not results_path.exists():
        return None

    with results_path.open() as f:
        data = json.load(f)

    if "tdcf" in data:
        hist = data["tdcf"]
        io_hist = hist.get("io_ratio", hist.get("approx_ratio", [1.0]))
    elif "baseline" in data:
        hist = data["baseline"]
        io_hist = [1.0]
    else:
        hist = data
        io_hist = [1.0]

    return {
        "best_test": float(max(hist["test_acc"])),
        "final_test": float(hist["test_acc"][-1]),
        "avg_io": float(np.mean(io_hist)),
        "avg_saved": float((1.0 - np.mean(io_hist)) * 100.0),
        "path": str(results_path.parent),
    }


def format_table(rows):
    header = "| Run | Best Test | Final Test | Avg I/O | Avg Saved |"
    sep = "|---|---:|---:|---:|---:|"
    body = [
        f"| {row['name']} | {row['best_test']:.4f} | {row['final_test']:.4f} | "
        f"{row['avg_io']:.3f} | {row['avg_saved']:.1f}% |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def main():
    args = parse_args()
    rows = []

    for name, run_dir in RUNS:
        run = load_run(args.results_root / run_dir / "results.json")
        if run is None:
            continue
        run["name"] = name
        rows.append(run)

    table = format_table(rows)
    print(table)

    if args.write_markdown:
        out_path = args.results_root / "summary.md"
        out_path.write_text(table + "\n")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
