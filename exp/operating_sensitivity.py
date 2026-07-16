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
from ner.router import select_from_candidate_score


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


def select_with_gap(
    temporal: np.ndarray,
    candidates: np.ndarray,
    score: np.ndarray,
    budget: int,
    minimum_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = temporal.astype(bool).copy()
    chosen: list[int] = []
    for position in np.argsort(score)[::-1]:
        candidate = int(candidates[position])
        if minimum_gap and any(abs(candidate - previous) <= minimum_gap for previous in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= budget:
            break
    chosen_array = np.asarray(chosen, dtype=np.int64)
    prediction[chosen_array] = True
    return prediction, chosen_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget and spacing sensitivity on frozen candidates.")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/v35")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sensitivity")
    args = parser.parse_args()
    if not (args.artifact_dir / "manifest.csv").exists():
        raise FileNotFoundError("Run `python reproduce.py` first.")
    budget_multipliers = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    spacing_multipliers = [1.0, 2.0, 4.0, 8.0]
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            prediction_file = np.load(
                args.artifact_dir / "predictions" / f"{dataset}_{model}_seed2021_predictions.npz"
            )
            score_file = np.load(
                args.artifact_dir / "candidate_scores" / f"{dataset}_{model}_candidate_scores.npz"
            )
            gt = prediction_file["gt"].astype(bool)
            temporal = prediction_file["temporal_pred"].astype(bool)
            candidates = score_file["candidates"].astype(np.int64)
            score = score_file["router_score"].astype(np.float64)
            budget = int(score_file["budget"][0])
            gaps = np.diff(np.sort(candidates))
            base_gap = max(int(np.min(gaps)) - 1, 0) if len(gaps) else 0
            for multiplier in budget_multipliers:
                current_budget = int(round(budget * multiplier))
                if multiplier > 0 and current_budget == 0:
                    current_budget = 1
                pred = select_from_candidate_score(temporal, candidates, score, current_budget)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "experiment": "budget",
                        "multiplier": multiplier,
                        **evaluate(pred, gt),
                    }
                )
            for multiplier in spacing_multipliers:
                pred, _ = select_with_gap(
                    temporal, candidates, score, budget, int(round(base_gap * multiplier))
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "experiment": "spacing",
                        "multiplier": multiplier,
                        **evaluate(pred, gt),
                    }
                )
    raw = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "operating_sensitivity_pair_metrics.csv", index=False)
    summary = (
        raw.groupby(["experiment", "multiplier"], as_index=False)[
            ["pa_f1", "event_f1", "range_f1", "false_events_per_100k"]
        ]
        .mean()
    )
    summary.to_csv(args.output_dir / "operating_sensitivity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for axis, experiment, title in zip(
        axes, ["budget", "spacing"], ["Alert budget", "Candidate spacing"]
    ):
        current = summary[summary["experiment"] == experiment]
        for metric, label, color in [
            ("pa_f1", "PA-F1", "#0072B2"),
            ("event_f1", "Event F1", "#D55E00"),
            ("range_f1", "Range F1", "#7B61A8"),
        ]:
            axis.plot(
                current["multiplier"].to_numpy(dtype=float),
                100 * current[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=1.4,
                label=label,
                color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("Multiplier")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "operating_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "operating_sensitivity.png", dpi=250, bbox_inches="tight")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

