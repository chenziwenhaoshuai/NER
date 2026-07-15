from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    "Baseline",
    "+ Temporal calibration",
    "Prior + NMS",
    "SelfNet only",
    "Geometry ConvAE only",
    "Augmented ConvAE only",
    "Neural-only mixed",
    "Neural Router v7",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the component ablation table.")
    parser.add_argument("--pair-metrics", type=Path, default=ROOT / "results/main/pair_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ablation")
    args = parser.parse_args()
    if not args.pair_metrics.exists():
        raise FileNotFoundError("Run `python reproduce.py` first.")
    pair = pd.read_csv(args.pair_metrics)
    rows = []
    for variant in VARIANTS:
        current = pair[pair["variant"] == variant]
        rows.append(
            {
                "variant": variant,
                "pa_precision": current["pa_precision"].mean() * 100,
                "pa_recall": current["pa_recall"].mean() * 100,
                "pa_f1": current["pa_f1"].mean() * 100,
                "point_f1": current["point_f1"].mean() * 100,
                "event_f1": current["event_f1"].mean() * 100,
                "range_f1": current["range_f1"].mean() * 100,
                "fpr": current["fpr"].mean() * 100,
            }
        )
    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "component_ablation.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

