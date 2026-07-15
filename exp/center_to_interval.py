from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ner.metrics import evaluate


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


def expand_centers(length: int, centers: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for center in centers.astype(np.int64):
        start = max(0, int(center) - radius)
        end = min(length, int(center) + radius + 1)
        mask[start:end] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse rescued-center conversion to finite alarm intervals."
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/v7")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/center_to_interval")
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5, 10, 20, 50, 100],
    )
    args = parser.parse_args()
    if not (args.artifact_dir / "manifest.csv").exists():
        raise FileNotFoundError("Run `python reproduce.py` first.")

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            saved = np.load(
                args.artifact_dir
                / "predictions"
                / f"{dataset}_{model}_seed2021_predictions.npz"
            )
            gt = saved["gt"].astype(bool)
            temporal = saved["temporal_pred"].astype(bool)
            centers = saved["selected_peaks"].astype(np.int64)
            for radius in args.radii:
                prediction = temporal | expand_centers(len(gt), centers, radius)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "radius": radius,
                        "selected_centers": len(centers),
                        **evaluate(prediction, gt),
                    }
                )

    raw = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "center_to_interval_pair_metrics.csv", index=False)
    metrics = [
        "pa_f1",
        "event_f1",
        "range_precision",
        "range_recall",
        "range_f1",
        "fpr",
        "false_events_per_100k",
    ]
    summary = raw.groupby("radius", as_index=False)[metrics].mean(numeric_only=True)
    summary.to_csv(args.output_dir / "center_to_interval_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(4.2, 2.7))
    for metric, label, color, marker in [
        ("pa_f1", "PA-F1", "#0072B2", "o"),
        ("event_f1", "Event F1", "#D55E00", "s"),
        ("range_f1", "Range F1", "#7B61A8", "^"),
    ]:
        axis.plot(
            summary["radius"].to_numpy(dtype=float),
            100.0 * summary[metric].to_numpy(dtype=float),
            marker=marker,
            linewidth=1.4,
            markersize=4,
            label=label,
            color=color,
        )
    axis.set_xlabel("Interval expansion radius")
    axis.set_ylabel("Metric (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "center_to_interval.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "center_to_interval.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
