from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ner.metrics import evaluate
from ner.router import rank01, route_score, select_from_candidate_score


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]
DEFAULTS = {
    "low_density": 2.5,
    "high_density": 10.0,
    "compact_budget": 9,
    "compact_candidates": 50,
}
SWEEPS = {
    "low_density": [1.5, 2.5, 4.0],
    "high_density": [8.0, 10.0, 12.0],
    "compact_budget": [7, 9, 12],
    "compact_candidates": [40, 50, 64],
}


def settings() -> list[tuple[str, float, dict[str, float | int]]]:
    output: list[tuple[str, float, dict[str, float | int]]] = []
    for parameter, values in SWEEPS.items():
        for value in values:
            current = dict(DEFAULTS)
            current[parameter] = value
            output.append((parameter, float(value), current))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-factor-at-a-time sensitivity of the legacy deterministic routing rule."
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/v35")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/router_sensitivity")
    args = parser.parse_args()
    if not (args.artifact_dir / "manifest.csv").exists():
        raise FileNotFoundError("Run `python reproduce.py` first.")

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            prediction_file = np.load(
                args.artifact_dir
                / "predictions"
                / f"{dataset}_{model}_seed2021_predictions.npz"
            )
            score_file = np.load(
                args.artifact_dir
                / "candidate_scores"
                / f"{dataset}_{model}_candidate_scores.npz"
            )
            gt = prediction_file["gt"].astype(bool)
            temporal = prediction_file["temporal_pred"].astype(bool)
            candidates = score_file["candidates"].astype(np.int64)
            budget = int(score_file["budget"][0])
            density = len(candidates) / max(float(budget), 1.0)
            inputs = [
                rank01(score_file["prior"]),
                rank01(score_file["self_event_score"]),
                rank01(score_file["geometry_ae_score"]),
                rank01(score_file["augmented_ae_score"]),
            ]
            for parameter, value, current in settings():
                score, route = route_score(
                    *inputs,
                    density=density,
                    budget=budget,
                    candidate_count=len(candidates),
                    low_density=float(current["low_density"]),
                    high_density=float(current["high_density"]),
                    compact_budget=int(current["compact_budget"]),
                    compact_candidates=int(current["compact_candidates"]),
                )
                prediction = select_from_candidate_score(
                    temporal, candidates, score, min(budget, len(candidates))
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "parameter": parameter,
                        "value": value,
                        "route": route,
                        **evaluate(prediction, gt),
                    }
                )

    raw = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "router_sensitivity_pair_metrics.csv", index=False)
    summary = (
        raw.groupby(["parameter", "value"], as_index=False)[
            ["pa_f1", "event_f1", "range_f1", "fpr"]
        ]
        .mean()
    )
    route_counts = (
        raw.groupby(["parameter", "value", "route"], as_index=False)
        .size()
        .rename(columns={"size": "pair_count"})
    )
    summary.to_csv(args.output_dir / "router_sensitivity_summary.csv", index=False)
    route_counts.to_csv(args.output_dir / "router_sensitivity_route_counts.csv", index=False)
    print(summary.to_string(index=False))
    print("\nRoute counts")
    print(route_counts.to_string(index=False))


if __name__ == "__main__":
    main()


