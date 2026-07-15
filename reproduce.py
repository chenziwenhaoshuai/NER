from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ner.io import ensure_artifacts
from ner.metrics import evaluate


ROOT = Path(__file__).resolve().parent
DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]
RELEASE_URL = (
    "https://github.com/chenziwenhaoshuai/NER/releases/download/v1.0/"
    "ner_v7_reproduction_artifacts.zip"
)
ARTIFACT_SHA256 = "e885f4087382257c9f6b7c7f66a8d40929e45d396039dac408d46b3a5b492f76"
EXPECTED = {
    "Baseline": {"pa_f1": 80.96960696018658, "event_f1": 29.484028191065175, "range_f1": 16.760069516068903},
    "Neural Router v7": {"pa_f1": 83.32966426706636, "event_f1": 32.690501656194066, "range_f1": 17.52668014993204},
}


PREDICTION_KEYS = {
    "Baseline": "baseline_pred",
    "+ Temporal calibration": "temporal_pred",
    "Prior + NMS": "prior_nms_pred",
    "SelfNet only": "self_only_pred",
    "Geometry ConvAE only": "geometry_ae_pred",
    "Augmented ConvAE only": "augmented_ae_pred",
    "Neural-only mixed": "neural_mix_pred",
    "Neural Router v7": "final_pred",
}


def load_artifacts(no_download: bool) -> Path:
    existing = ROOT / "artifacts" / "v7"
    if no_download:
        if not (existing / "manifest.csv").exists():
            raise FileNotFoundError(
                "artifacts/v7 is missing. Run without --no-download or place the release archive locally."
            )
        return existing
    return ensure_artifacts(ROOT, RELEASE_URL, ARTIFACT_SHA256)


def reproduce(artifact_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            path = artifact_root / "predictions" / f"{dataset}_{model}_seed2021_predictions.npz"
            saved = np.load(path)
            gt = saved["gt"].astype(bool).reshape(-1)
            for variant, key in PREDICTION_KEYS.items():
                metrics = evaluate(saved[key].astype(bool).reshape(-1), gt)
                rows.append({"dataset": dataset, "model": model, "variant": variant, **metrics})
    pair = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair.to_csv(output_dir / "pair_metrics.csv", index=False)
    metrics = [
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
    overall = pair.groupby("variant", as_index=False)[metrics].mean(numeric_only=True)
    for column in metrics:
        if column != "false_events_per_100k":
            overall[column] *= 100.0
    overall = overall.sort_values(["pa_f1", "event_f1", "range_f1"], ascending=False)
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    dataset = pair.groupby(["dataset", "variant"], as_index=False)[metrics].mean(numeric_only=True)
    dataset.to_csv(output_dir / "dataset_metrics_fraction.csv", index=False)
    model = pair.groupby(["model", "variant"], as_index=False)[metrics].mean(numeric_only=True)
    model.to_csv(output_dir / "backbone_metrics_fraction.csv", index=False)
    return overall


def verify(overall: pd.DataFrame, tolerance: float) -> None:
    indexed = overall.set_index("variant")
    failures = []
    for variant, expected_metrics in EXPECTED.items():
        for metric, expected in expected_metrics.items():
            actual = float(indexed.loc[variant, metric])
            if abs(actual - expected) > tolerance:
                failures.append(
                    {"variant": variant, "metric": metric, "expected": expected, "actual": actual}
                )
    if failures:
        raise RuntimeError("Reproduction mismatch:\n" + json.dumps(failures, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the NER paper's main 5x5 results.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "main")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    artifact_root = load_artifacts(args.no_download)
    overall = reproduce(artifact_root, args.output_dir)
    verify(overall, args.tolerance)
    print(overall.to_string(index=False))
    print("\nReproduction check passed.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
