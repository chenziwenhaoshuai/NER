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
from ner.router import rank01, route_score, select_from_candidate_score


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


def perturb(values: np.ndarray, candidates: np.ndarray, family: str, severity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = rank01(values)
    if family == "clean":
        return clean
    if family == "noise":
        return clean + rng.normal(0.0, severity, len(clean))
    if family == "drift":
        output = clean.copy()
        output[np.argsort(candidates)] += np.linspace(-severity, severity, len(clean))
        return output
    if family == "dropout":
        output = clean.copy()
        output[rng.random(len(output)) < severity] = 0.5
        return output
    raise ValueError(family)


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-evidence robustness analysis.")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/v7")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evidence_stress")
    args = parser.parse_args()
    grid = {
        "clean": [0.0],
        "noise": [0.02, 0.05, 0.10, 0.20],
        "drift": [0.02, 0.05, 0.10, 0.20],
        "dropout": [0.01, 0.05, 0.10, 0.20],
    }
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(MODELS):
            prediction_file = np.load(
                args.artifact_dir / "predictions" / f"{dataset}_{model}_seed2021_predictions.npz"
            )
            score_file = np.load(
                args.artifact_dir / "candidate_scores" / f"{dataset}_{model}_candidate_scores.npz"
            )
            gt = prediction_file["gt"].astype(bool)
            temporal = prediction_file["temporal_pred"].astype(bool)
            candidates = score_file["candidates"].astype(np.int64)
            budget = int(score_file["budget"][0])
            density = len(candidates) / max(float(budget), 1.0)
            sources = [
                score_file["prior"],
                score_file["self_event_score"],
                score_file["geometry_ae_score"],
                score_file["augmented_ae_score"],
            ]
            for family, severities in grid.items():
                for severity in severities:
                    for repeat in range(3):
                        seed = 20210 + 1000 * dataset_index + 100 * model_index + repeat
                        perturbed = [
                            perturb(source, candidates, family, severity, seed + 13 * index)
                            for index, source in enumerate(sources)
                        ]
                        score, route = route_score(
                            *perturbed,
                            density=density,
                            budget=budget,
                            candidate_count=len(candidates),
                        )
                        pred = select_from_candidate_score(temporal, candidates, score, budget)
                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "family": family,
                                "severity": severity,
                                "repeat": repeat,
                                "route": route,
                                **evaluate(pred, gt),
                            }
                        )
    raw = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "evidence_stress_pair_metrics.csv", index=False)
    summary = raw.groupby(["family", "severity"], as_index=False)[
        ["pa_f1", "event_f1", "range_f1", "fpr"]
    ].agg(["mean", "std"])
    summary.columns = [
        "_".join(item).strip("_") if isinstance(item, tuple) else item for item in summary.columns
    ]
    summary.to_csv(args.output_dir / "evidence_stress_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for axis, family in zip(axes, ["noise", "drift", "dropout"]):
        current = raw[raw["family"] == family].groupby("severity", as_index=False)["event_f1"].mean()
        axis.plot(
            current["severity"].to_numpy(dtype=float),
            100 * current["event_f1"].to_numpy(dtype=float),
            marker="o",
            color="#0072B2",
        )
        axis.set_title(family.capitalize())
        axis.set_xlabel("Perturbation")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].set_ylabel("Event F1 (%)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "evidence_stress.pdf", bbox_inches="tight")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
