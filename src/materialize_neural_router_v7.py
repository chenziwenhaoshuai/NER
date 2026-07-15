"""Materialize Neural Router v7.

v7 is the neural-dominant version of the router.  The candidate pool and
budget are unchanged, but the final event decision is controlled by neural
evidence:

  * compact / low-density candidate sets use Geometry ConvAE;
  * high-density candidate sets use Augmented ConvAE;
  * middle-density sets use a neural-dominant rank mixture.

The event prior is kept only as a weak stabilizer in the mixed route, not as
the main ranking signal.  Labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuse_neural_candidate_scores import DATASETS, MODELS, evaluate, point_adjust, rank01, select_from_candidate_score


DATASET_ORDER = {d: i for i, d in enumerate(DATASETS)}
MODEL_ORDER = {m: i for i, m in enumerate(MODELS)}


def find_score_file(root: Path, dataset: str, model: str) -> Path:
    matches = list(root.glob(f"**/candidate_scores/{dataset}_{model}_candidate_scores.npz"))
    if not matches:
        raise FileNotFoundError(f"missing candidate score for {dataset}/{model} under {root}")
    return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]


def route_score(
    prior_rank: np.ndarray,
    self_rank: np.ndarray,
    geom_rank: np.ndarray,
    aug_rank: np.ndarray,
    density: float,
    budget: int,
    candidate_count: int,
    low_density: float,
    high_density: float,
    compact_budget: int,
    compact_candidates: int,
) -> tuple[np.ndarray, str]:
    if density <= low_density or (budget <= compact_budget and candidate_count <= compact_candidates):
        return geom_rank, "geometry_ae_compact"
    if density >= high_density:
        return aug_rank, "augmented_ae_dense"
    # Neural-dominant mixed route.  Prior is only a weak stabilizer.
    score = 0.25 * prior_rank + 0.5 * self_rank + 2.0 * geom_rank + 1.0 * aug_rank
    return score, "mixed_neural_dominant"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=Path(
            "experiments/baseline_transfer/v83/full_comparable_baselines_20260713/"
            "paper_msl_profile_ratio1_budget144_radius100"
        ),
    )
    parser.add_argument("--self_dir", type=Path, default=None)
    parser.add_argument("--geometry_ae_dir", type=Path, default=None)
    parser.add_argument("--augmented_ae_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--low_density", type=float, default=2.5)
    parser.add_argument("--high_density", type=float, default=10.0)
    parser.add_argument("--compact_budget", type=int, default=9)
    parser.add_argument("--compact_candidates", type=int, default=50)
    args = parser.parse_args()

    self_dir = args.self_dir or args.exp_dir / "self_event_ranker_all25_e30"
    geometry_ae_dir = args.geometry_ae_dir or args.exp_dir / "convae_geometry_all25_scores_e8"
    augmented_ae_dir = args.augmented_ae_dir or args.exp_dir / "augmented_convae_all25_e4_w12000_remote"
    out_dir = args.output_dir or args.exp_dir / "neural_router_v7_neural_dominant"
    pred_dir = out_dir / "predictions"
    score_dir = out_dir / "candidate_scores"
    summary_dir = out_dir / "summary"
    for p in [pred_dir, score_dir, summary_dir]:
        p.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            z = np.load(args.exp_dir / "rescue" / f"{dataset}_{model}_seed2021_predictions.npz")
            gt = z["gt"].astype(bool).reshape(-1)
            baseline = z["baseline_pred"].astype(bool).reshape(-1)
            temporal = z["temporal_pred"].astype(bool).reshape(-1)
            current = z["final_pred"].astype(bool).reshape(-1)
            prior = z["event_prior"].astype(np.float64).reshape(-1)
            candidates = z["candidate_indices"].astype(np.int64).reshape(-1)
            candidates = candidates[(candidates >= 0) & (candidates < len(gt))]
            candidates = candidates[~temporal[candidates]]
            candidates = np.unique(candidates)
            selected = z["selected_peaks"].astype(np.int64).reshape(-1) if "selected_peaks" in z.files else np.array([], dtype=np.int64)
            budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

            self_z = np.load(find_score_file(self_dir, dataset, model))
            geom_z = np.load(find_score_file(geometry_ae_dir, dataset, model))
            aug_z = np.load(find_score_file(augmented_ae_dir, dataset, model))
            for name, arr in [
                ("SelfNet", self_z["candidates"]),
                ("Geometry ConvAE", geom_z["candidates"]),
                ("Augmented ConvAE", aug_z["candidates"]),
            ]:
                if not np.array_equal(candidates, arr.astype(np.int64).reshape(-1)):
                    raise ValueError(f"candidate mismatch for {dataset}/{model}: {name}")

            prior_c = prior[candidates]
            self_score = self_z["self_event_score"].astype(np.float64).reshape(-1)
            geom_score = geom_z["center_event_score"].astype(np.float64).reshape(-1)
            aug_score = aug_z["augmented_convae_score"].astype(np.float64).reshape(-1)
            density = len(candidates) / max(float(budget), 1.0)
            router_score, route = route_score(
                rank01(prior_c),
                rank01(self_score),
                rank01(geom_score),
                rank01(aug_score),
                density=density,
                budget=budget,
                candidate_count=len(candidates),
                low_density=args.low_density,
                high_density=args.high_density,
                compact_budget=args.compact_budget,
                compact_candidates=args.compact_candidates,
            )
            final = select_from_candidate_score(temporal, candidates, router_score, budget)
            final_selected = candidates[np.argsort(router_score)[::-1]][:budget] if len(candidates) else np.array([], dtype=np.int64)

            prior_pred = select_from_candidate_score(temporal, candidates, prior_c, budget)
            geom_pred = select_from_candidate_score(temporal, candidates, geom_score, budget)
            aug_pred = select_from_candidate_score(temporal, candidates, aug_score, budget)
            self_pred = select_from_candidate_score(temporal, candidates, self_score, budget)
            neural_mix_score = 0.5 * rank01(self_score) + 2.0 * rank01(geom_score) + rank01(aug_score)
            neural_mix_pred = select_from_candidate_score(temporal, candidates, neural_mix_score, budget)

            np.savez_compressed(
                pred_dir / f"{dataset}_{model}_seed2021_predictions.npz",
                gt=gt.astype(np.int8),
                baseline_pred=baseline.astype(np.int8),
                temporal_pred=temporal.astype(np.int8),
                prior_nms_pred=prior_pred.astype(np.int8),
                current_ner_pred=current.astype(np.int8),
                self_only_pred=self_pred.astype(np.int8),
                geometry_ae_pred=geom_pred.astype(np.int8),
                augmented_ae_pred=aug_pred.astype(np.int8),
                neural_mix_pred=neural_mix_pred.astype(np.int8),
                final_pred=final.astype(np.int8),
                final_point_adjust_pred=point_adjust(final, gt).astype(np.int8),
                event_prior=prior.astype(np.float32),
                candidate_indices=candidates.astype(np.int32),
                selected_peaks=final_selected.astype(np.int32),
                router_score_candidate=router_score.astype(np.float32),
                route=np.array([route]),
                candidate_density=np.array([density], dtype=np.float32),
            )
            np.savez_compressed(
                score_dir / f"{dataset}_{model}_candidate_scores.npz",
                candidates=candidates.astype(np.int32),
                prior=prior_c.astype(np.float32),
                self_event_score=self_score.astype(np.float32),
                geometry_ae_score=geom_score.astype(np.float32),
                augmented_ae_score=aug_score.astype(np.float32),
                router_score=router_score.astype(np.float32),
                budget=np.array([budget], dtype=np.int32),
                route=np.array([route]),
                candidate_density=np.array([density], dtype=np.float32),
            )

            variants = {
                "Baseline": baseline,
                "+ Temporal calibration": temporal,
                "Prior + NMS": prior_pred,
                "Current NER": current,
                "SelfNet only": self_pred,
                "Geometry ConvAE only": geom_pred,
                "Augmented ConvAE only": aug_pred,
                "Neural-only mixed": neural_mix_pred,
                "Neural Router v7": final,
            }
            for variant, pred in variants.items():
                row = {
                    "dataset": dataset,
                    "model": model,
                    "variant": variant,
                    "budget": budget,
                    "candidate_count": len(candidates),
                    "candidate_density": density,
                    "route": route if variant == "Neural Router v7" else "",
                }
                row.update(evaluate(pred, gt))
                rows.append(row)

    pair = pd.DataFrame(rows)
    pair["_dataset_order"] = pair["dataset"].map(DATASET_ORDER)
    pair["_model_order"] = pair["model"].map(MODEL_ORDER)
    pair = pair.sort_values(["_dataset_order", "_model_order", "variant"]).drop(columns=["_dataset_order", "_model_order"])
    pair.to_csv(summary_dir / "neural_router_v7_pair_metrics.csv", index=False)
    avg_cols = ["pa_precision", "pa_recall", "pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]
    overall = pair.groupby("variant", as_index=False)[avg_cols].mean(numeric_only=True)
    dataset_avg = pair.groupby(["dataset", "variant"], as_index=False)[avg_cols].mean(numeric_only=True)
    model_avg = pair.groupby(["model", "variant"], as_index=False)[avg_cols].mean(numeric_only=True)
    route_summary = pair[pair["variant"] == "Neural Router v7"][
        ["dataset", "model", "budget", "candidate_count", "candidate_density", "route"]
    ].copy()
    for df in [overall, dataset_avg, model_avg]:
        for c in avg_cols:
            df[c] *= 100.0
    overall = overall.sort_values(["pa_f1", "event_f1", "range_f1"], ascending=False)
    dataset_avg["_dataset_order"] = dataset_avg["dataset"].map(DATASET_ORDER)
    dataset_avg = dataset_avg.sort_values(["_dataset_order", "variant"]).drop(columns="_dataset_order")
    model_avg["_model_order"] = model_avg["model"].map(MODEL_ORDER)
    model_avg = model_avg.sort_values(["_model_order", "variant"]).drop(columns="_model_order")
    overall.to_csv(summary_dir / "neural_router_v7_overall_metrics.csv", index=False)
    dataset_avg.to_csv(summary_dir / "neural_router_v7_dataset_average.csv", index=False)
    model_avg.to_csv(summary_dir / "neural_router_v7_model_average.csv", index=False)
    route_summary.to_csv(summary_dir / "neural_router_v7_routes.csv", index=False)
    config = {
        "method": "Neural Router v7",
        "route_rule": "compact/low density: Geometry ConvAE; high density: Augmented ConvAE; middle: neural-dominant rank mixture",
        "mixed_formula": "0.25*rank(prior)+0.5*rank(SelfNet)+2*rank(GeometryConvAE)+rank(AugmentedConvAE)",
        "low_density": args.low_density,
        "high_density": args.high_density,
        "compact_budget": args.compact_budget,
        "compact_candidates": args.compact_candidates,
        "exp_dir": str(args.exp_dir),
        "self_dir": str(self_dir),
        "geometry_ae_dir": str(geometry_ae_dir),
        "augmented_ae_dir": str(augmented_ae_dir),
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    print(route_summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
