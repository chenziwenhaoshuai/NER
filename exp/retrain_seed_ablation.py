"""Retrain all NER neural branches across random seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full neural-module seed ablation.")
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/retrain_seed_ablation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands: list[list[str]] = []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed{seed}"
        summary_file = (
            seed_dir
            / "neural_router_v7"
            / "summary"
            / "neural_router_v7_pair_metrics.csv"
        )
        if args.skip_existing and summary_file.exists():
            continue
        commands.append(
            [
                sys.executable,
                str(ROOT / "exp/train_component_ablation.py"),
                "--exp-dir",
                str(args.exp_dir),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(seed_dir),
                "--seed",
                str(seed),
                "--batch-size",
                str(args.batch_size),
                "--score-batch-size",
                str(args.score_batch_size),
            ]
        )

    for command in commands:
        print("+", subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.dry_run:
        return

    pair_rows = []
    for seed in args.seeds:
        summary_file = (
            args.output_dir
            / f"seed{seed}"
            / "neural_router_v7"
            / "summary"
            / "neural_router_v7_pair_metrics.csv"
        )
        if not summary_file.exists():
            raise FileNotFoundError(summary_file)
        current = pd.read_csv(summary_file)
        current = current[current["variant"] == "Neural Router v7"].copy()
        current.insert(0, "seed", seed)
        pair_rows.append(current)

    pair = pd.concat(pair_rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair.to_csv(args.output_dir / "seed_ablation_pair_metrics.csv", index=False)
    metric_columns = ["pa_f1", "event_f1", "range_f1", "fpr"]
    by_seed = pair.groupby("seed", as_index=False)[metric_columns].mean(numeric_only=True)
    by_seed.to_csv(args.output_dir / "seed_ablation_by_seed.csv", index=False)
    overall = by_seed[metric_columns].agg(["mean", "std"]).transpose().reset_index()
    overall.columns = ["metric", "mean", "std"]
    overall.to_csv(args.output_dir / "seed_ablation_summary.csv", index=False)
    print(by_seed.to_string(index=False))
    print("\nAcross-seed summary")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
