"""Regenerate candidate pools and retrain neural scorers across NMS radii."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from experiment_augmented_convae_candidate_scorer import (
    make_augmented_series,
    make_windows as make_augmented_windows,
    sample_normal_centers,
    score_ae as score_augmented_ae,
    train_ae as train_augmented_ae,
)
from experiment_convae_candidate_scorer import (
    make_feature_series,
    make_windows as make_geometry_windows,
    score_windows_ae,
    train_ae as train_geometry_ae,
)
from experiment_self_trained_event_ranker import (
    build_augmented_series,
    score_model,
    train_self_ranker,
)
from ner.metrics import evaluate
from ner.router import rank01, route_score, select_from_candidate_score
from run_dtpp_from_at_scores import load_dataset, nms_order


DEFAULT_PAIRS = (
    "MSL:AnomalyTransformer,SMD:AnomalyTransformer,"
    "PSM:Autoformer,PSM:KANAD,SWaT:TimesNet"
)


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for item in text.split(","):
        dataset, model = item.strip().split(":", 1)
        pairs.append((dataset, model))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Candidate-spacing ablation with regenerated pools and trained scorers."
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500, 1000, 2500, 5000],
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--self-epochs", type=int, default=30)
    parser.add_argument("--geometry-epochs", type=int, default=8)
    parser.add_argument("--augmented-epochs", type=int, default=4)
    parser.add_argument("--geometry-windows", type=int, default=40000)
    parser.add_argument("--augmented-windows", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=2048)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.output_dir / "candidate_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    geometry_cache: dict[str, tuple[torch.nn.Module, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for dataset, model in parse_pairs(args.pairs):
        source = np.load(
            args.exp_dir / "rescue" / f"{dataset}_{model}_seed2021_predictions.npz"
        )
        gt = source["gt"].astype(bool).reshape(-1)
        baseline = source["baseline_pred"].astype(bool).reshape(-1)
        temporal = source["temporal_pred"].astype(bool).reshape(-1)
        prior = source["event_prior"].astype(np.float64).reshape(-1)
        train_energy = source["train_energy"].astype(np.float64).reshape(-1)
        test_energy = source["test_energy"].astype(np.float64).reshape(-1)
        selected = source["selected_peaks"].astype(np.int64).reshape(-1)
        budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

        if dataset not in data_cache:
            train_z, test_z_full, labels = load_dataset(
                args.data_root / dataset, dataset, 10**18
            )
            data_cache[dataset] = (
                train_z.astype(np.float32),
                test_z_full[: len(gt)].astype(np.float32),
                labels[: len(gt)].astype(bool),
            )
        train_z, test_z, labels = data_cache[dataset]
        if not np.array_equal(labels, gt):
            raise ValueError(f"label mismatch for {dataset}/{model}")

        self_series, self_diff_scale = build_augmented_series(
            test_z, train_energy, test_energy, prior
        )
        self_model, pseudo = train_self_ranker(
            aug=self_series,
            diff_scale=self_diff_scale,
            temporal=temporal,
            prior=prior,
            window=args.window,
            seed=args.seed,
            device=device,
            epochs=args.self_epochs,
            batch_size=args.batch_size,
            hidden=96,
            max_pos=5000,
            neg_multiplier=2.0,
            boundary_radius=96,
            feature_mode="geometry",
        )

        if dataset not in geometry_cache:
            geometry_train = make_feature_series(train_z, train_z, mode="geometry")
            geometry_test = make_feature_series(test_z, train_z, mode="geometry")
            geometry_model = train_geometry_ae(
                train_z=geometry_train,
                window=args.window,
                n_windows=args.geometry_windows,
                seed=args.seed,
                device=device,
                epochs=args.geometry_epochs,
                batch_size=args.batch_size,
                hidden=64,
                bottleneck=32,
                noise_std=0.03,
            )
            geometry_cache[dataset] = (geometry_model, geometry_test)
        geometry_model, geometry_test = geometry_cache[dataset]

        augmented_series = make_augmented_series(
            train_z, test_z, train_energy, test_energy, prior
        )
        augmented_centers = sample_normal_centers(
            temporal=temporal,
            prior=prior,
            window=args.window,
            n_windows=args.augmented_windows,
            seed=args.seed,
            exclusion_radius=32,
            prior_quantile=0.70,
        )
        augmented_model = train_augmented_ae(
            feature_series=augmented_series,
            train_centers=augmented_centers,
            window=args.window,
            seed=args.seed,
            device=device,
            epochs=args.augmented_epochs,
            batch_size=max(128, args.batch_size // 2),
            hidden=96,
            bottleneck=48,
            noise_std=0.02,
        )

        baseline_metrics = evaluate(baseline, gt)
        for radius in args.radii:
            candidates = nms_order(prior, radius=radius)
            candidates = candidates[(prior[candidates] > 0) & (~temporal[candidates])]
            candidates = candidates[:30000].astype(np.int64)
            prior_score = prior[candidates]
            self_score = score_model(
                self_model,
                self_series,
                self_diff_scale,
                candidates,
                window=args.window,
                feature_mode="geometry",
                device=device,
                batch_size=args.score_batch_size,
            )
            geometry_windows = (
                make_geometry_windows(geometry_test, candidates, args.window)
                if len(candidates)
                else np.empty((0, args.window, geometry_test.shape[1]), dtype=np.float32)
            )
            geometry_score = score_windows_ae(
                geometry_model, geometry_windows, device, args.score_batch_size
            )
            augmented_windows = (
                make_augmented_windows(augmented_series, candidates, args.window)
                if len(candidates)
                else np.empty((0, args.window, augmented_series.shape[1]), dtype=np.float32)
            )
            augmented_score = score_augmented_ae(
                augmented_model, augmented_windows, device, args.score_batch_size
            )
            density = len(candidates) / max(float(budget), 1.0)
            router_score, route = route_score(
                rank01(prior_score),
                rank01(self_score),
                rank01(geometry_score),
                rank01(augmented_score),
                density=density,
                budget=budget,
                candidate_count=len(candidates),
            )
            final = select_from_candidate_score(
                temporal, candidates, router_score, min(budget, len(candidates))
            )
            metrics = evaluate(final, gt)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": args.seed,
                    "nms_radius": radius,
                    "route": route,
                    "budget": budget,
                    "candidate_count": len(candidates),
                    "candidate_density": density,
                    "baseline_pa_f1": baseline_metrics["pa_f1"],
                    "ner_pa_f1": metrics["pa_f1"],
                    "delta_pa_f1": metrics["pa_f1"] - baseline_metrics["pa_f1"],
                    "baseline_event_f1": baseline_metrics["event_f1"],
                    "ner_event_f1": metrics["event_f1"],
                    "delta_event_f1": metrics["event_f1"] - baseline_metrics["event_f1"],
                    "baseline_range_f1": baseline_metrics["range_f1"],
                    "ner_range_f1": metrics["range_f1"],
                    "delta_range_f1": metrics["range_f1"] - baseline_metrics["range_f1"],
                    "ner_fpr": metrics["fpr"],
                    "false_events_per_100k": metrics["false_events_per_100k"],
                    **pseudo,
                }
            )
            np.savez_compressed(
                score_dir / f"{dataset}_{model}_radius{radius}.npz",
                candidates=candidates.astype(np.int32),
                prior_score=prior_score.astype(np.float32),
                self_score=self_score.astype(np.float32),
                geometry_score=geometry_score.astype(np.float32),
                augmented_score=augmented_score.astype(np.float32),
                router_score=router_score.astype(np.float32),
            )
            print(
                f"{dataset}/{model} radius={radius} candidates={len(candidates)} "
                f"route={route} PA-F1={metrics['pa_f1']:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "spacing_ablation_pair_metrics.csv", index=False)
    summary = frame.groupby("nms_radius", as_index=False)[
        [
            "baseline_pa_f1",
            "ner_pa_f1",
            "delta_pa_f1",
            "ner_event_f1",
            "delta_event_f1",
            "ner_range_f1",
            "delta_range_f1",
            "ner_fpr",
            "false_events_per_100k",
            "candidate_count",
        ]
    ].mean(numeric_only=True)
    summary.to_csv(args.output_dir / "spacing_ablation_summary.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
