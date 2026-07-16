"""Create paper-ready comparison tables for MoE-router experiments.

The script expects one or more experiment directories with a
``summary/overall_metrics.csv`` file and, for the final method, optional
``summary/pair_metrics.csv``, ``summary/seed_summary.csv``, and
``summary/temperature_selection.csv`` files.  It does not depend on internal
absolute paths, so the same command can be used inside the released repository
or in the full experiment workspace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "pa_precision",
    "pa_recall",
    "pa_f1",
    "point_f1",
    "event_f1",
    "range_f1",
    "strict_event_f1",
    "false_events_per_100k",
    "fpr",
]


def read_overall(path: Path, method: str, component: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    row = pd.read_csv(path).iloc[0].to_dict()
    row["method"] = method
    row["component"] = component
    return row


def parse_method_spec(spec: str) -> tuple[str, str, Path]:
    """Parse METHOD|COMPONENT|DIR command-line specs."""

    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(
            "Each --method entry must be formatted as 'method name|component|experiment_dir'."
        )
    return parts[0], parts[1], Path(parts[2])


def percentify(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for metric in metrics:
        if metric in out.columns and metric != "false_events_per_100k":
            out[metric] *= 100.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MoE comparison and ablation tables.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=(
            "Method spec formatted as 'method name|component|experiment_dir'. "
            "The directory must contain summary/overall_metrics.csv. Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help=(
            "Directory of the final MoE run. If provided, dataset averages, "
            "seed stability, and temperature-selection tables are generated when files exist."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_ablation_tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = DEFAULT_METRICS

    rows = []
    for spec in args.method:
        method, component, directory = parse_method_spec(spec)
        rows.append(read_overall(directory / "summary" / "overall_metrics.csv", method, component))
    if rows:
        comparison = pd.DataFrame(rows)
        cols = ["method", "component"] + [metric for metric in metrics if metric in comparison.columns]
        percentify(comparison[cols], [m for m in metrics if m != "false_events_per_100k"]).to_csv(
            args.output_dir / "table_moe_comparison.csv", index=False
        )
        print("\nMoE comparison")
        print(percentify(comparison[cols], [m for m in metrics if m != "false_events_per_100k"]).to_string(index=False))

    if args.final_dir is None:
        return

    final_dir = args.final_dir
    pair_path = final_dir / "summary" / "pair_metrics.csv"
    if pair_path.exists():
        pair = pd.read_csv(pair_path)
        dataset = (
            pair.groupby("dataset", as_index=False)[[m for m in metrics if m in pair.columns]]
            .mean(numeric_only=True)
            .sort_values("dataset")
        )
        dataset = percentify(dataset, metrics)
        dataset.to_csv(args.output_dir / "table_moe_dataset_average.csv", index=False)
        print("\nFinal MoE dataset averages")
        print(dataset.to_string(index=False))

    seed_path = final_dir / "summary" / "seed_summary.csv"
    if seed_path.exists():
        seeds = pd.read_csv(seed_path)
        seed_metrics = [m for m in ["pa_f1", "event_f1", "range_f1", "point_f1"] if m in seeds.columns]
        seed_summary = seeds[seed_metrics].agg(["mean", "std"])
        seed_summary.to_csv(args.output_dir / "table_moe_seed_stability.csv")
        print("\nFinal MoE seed stability")
        print(seed_summary.to_string())

    selection_path = final_dir / "summary" / "temperature_selection.csv"
    if selection_path.exists():
        selection = pd.read_csv(selection_path)
        selection.to_csv(args.output_dir / "table_moe_temperature_selection.csv", index=False)
        print("\nFinal MoE temperature selection")
        print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
