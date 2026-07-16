from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ner.metrics import evaluate


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


def random_nms(
    base: np.ndarray, budget: int, radius: int, rng: np.random.Generator
) -> np.ndarray:
    prediction = base.astype(bool).copy()
    eligible = np.flatnonzero(~prediction)
    rng.shuffle(eligible)
    suppressed = np.zeros(len(prediction), dtype=bool)
    chosen: list[int] = []
    for index in eligible:
        if suppressed[index]:
            continue
        chosen.append(int(index))
        suppressed[max(0, index - radius) : min(len(prediction), index + radius + 1)] = True
        if len(chosen) >= budget:
            break
    if chosen:
        prediction[np.asarray(chosen, dtype=np.int64)] = True
    return prediction


def infer_radius(dataset: str) -> int:
    return 100 if dataset == "MSL" else 2500


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-budget random insertion guardrail.")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/v35")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/random_guardrail")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if not (args.artifact_dir / "manifest.csv").exists():
        raise FileNotFoundError("Run `python reproduce.py` first.")
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(MODELS):
            prediction_file = np.load(
                args.artifact_dir / "predictions" / f"{dataset}_{model}_seed2021_predictions.npz"
            )
            score_file = np.load(
                args.artifact_dir / "candidate_scores" / f"{dataset}_{model}_candidate_scores.npz"
            )
            gt = prediction_file["gt"].astype(bool)
            budget = int(score_file["budget"][0])
            radius = infer_radius(dataset)
            deterministic = {
                "Baseline": prediction_file["baseline_pred"].astype(bool),
                "+ Temporal calibration": prediction_file["temporal_pred"].astype(bool),
                "Prior + NMS": prediction_file["prior_nms_pred"].astype(bool),
                "Neural Event Rescue": prediction_file["final_pred"].astype(bool),
            }
            for method, pred in deterministic.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": -1,
                        **evaluate(pred, gt),
                    }
                )
            for seed in range(args.seeds):
                base_seed = 10000 * dataset_index + 100 * model_index + seed
                for method, base_key in [
                    ("Baseline + Random", "baseline_pred"),
                    ("Temporal + Random", "temporal_pred"),
                ]:
                    pred = random_nms(
                        prediction_file[base_key].astype(bool),
                        budget,
                        radius,
                        np.random.default_rng(base_seed),
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "seed": seed,
                            **evaluate(pred, gt),
                        }
                    )
    raw = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "random_insertion_pair_seed_metrics.csv", index=False)
    summary = raw.groupby("method", as_index=False)[
        ["pa_f1", "point_f1", "event_f1", "range_f1", "strict_event_f1", "fpr"]
    ].mean()
    for column in ["pa_f1", "point_f1", "event_f1", "range_f1", "strict_event_f1", "fpr"]:
        summary[column] *= 100
    summary.to_csv(args.output_dir / "random_insertion_summary.csv", index=False)
    print(summary.sort_values("pa_f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

