"""Fuse saved neural candidate scores without retraining.

Consumes:
  - paper profile prediction NPZs
  - SelfNet candidate score NPZs
  - ConvAE candidate score NPZs

Evaluates fixed rank-sum mixtures over all 25 pairs.  This is an offline
architecture-search diagnostic: if a fixed neural mixture beats Prior+NMS
substantially and consistently, it becomes a candidate module for the final
pipeline; otherwise we keep redesigning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, v in enumerate(mask.astype(bool)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def point_adjust(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = pred.astype(bool).copy()
    for s, e in segments(gt):
        if out[s : e + 1].any():
            out[s : e + 1] = True
    return out


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def binary_f1(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return f1_from_counts(tp, fp, fn)


def event_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_segs = segments(pred)
    gt_segs = segments(gt)
    matched_gt = set()
    matched_pred = 0
    gi = 0
    for ps, pe in pred_segs:
        while gi < len(gt_segs) and gt_segs[gi][1] < ps:
            gi += 1
        scan = gi
        while scan < len(gt_segs) and gt_segs[scan][0] <= pe:
            matched_gt.add(scan)
            matched_pred += 1
            break
    p = matched_pred / len(pred_segs) if pred_segs else 0.0
    r = len(matched_gt) / len(gt_segs) if gt_segs else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def range_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_segs = segments(pred)
    gt_segs = segments(gt)
    gt_prefix = np.concatenate([[0], np.cumsum(gt.astype(np.int64))])
    pred_prefix = np.concatenate([[0], np.cumsum(pred.astype(np.int64))])
    p = float(np.mean([(gt_prefix[e + 1] - gt_prefix[s]) / max(e - s + 1, 1) for s, e in pred_segs])) if pred_segs else 0.0
    r = float(np.mean([(pred_prefix[e + 1] - pred_prefix[s]) / max(e - s + 1, 1) for s, e in gt_segs])) if gt_segs else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pa = point_adjust(pred, gt)
    _, _, pf1 = binary_f1(pred, gt)
    pap, par, paf1 = binary_f1(pa, gt)
    fp = int(np.logical_and(pred.astype(bool), ~gt.astype(bool)).sum())
    tn = int(np.logical_and(~pred.astype(bool), ~gt.astype(bool)).sum())
    return {
        "pa_precision": pap,
        "pa_recall": par,
        "pa_f1": paf1,
        "point_f1": pf1,
        "event_f1": event_f1(pred, gt),
        "range_f1": range_f1(pred, gt),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "pred_points": int(pred.astype(bool).sum()),
    }


def rank01(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True) if len(values) > 1 else 1.0
    return ranks


def select_from_candidate_score(
    temporal: np.ndarray, candidates: np.ndarray, candidate_score: np.ndarray, budget: int
) -> np.ndarray:
    pred = temporal.astype(bool).copy()
    if len(candidates) and budget > 0:
        ordered = candidates[np.argsort(candidate_score)[::-1]]
        pred[ordered[:budget]] = True
    return pred


def find_score_file(root: Path, dataset: str, model: str) -> Path:
    matches = list(root.glob(f"**/candidate_scores/{dataset}_{model}_candidate_scores.npz"))
    if not matches:
        raise FileNotFoundError(f"missing candidate score for {dataset}/{model} under {root}")
    if len(matches) > 1:
        # Prefer all25/e30 merged tree paths over probes.
        matches = sorted(matches, key=lambda p: (len(str(p)), str(p)))
    return matches[0]


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
    parser.add_argument("--convae_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Evaluate a small fixed set of fusions.")
    args = parser.parse_args()

    self_dir = args.self_dir or args.exp_dir / "self_event_ranker_all25_e30"
    convae_dir = args.convae_dir or args.exp_dir / "convae_geometry_all25"
    out_dir = args.output_dir or args.exp_dir / "neural_score_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    combo_specs = []
    if args.quick:
        combo_specs = [
            ("Rank Self only", (0.0, 1.0, 0.0)),
            ("Rank AE only", (0.0, 0.0, 1.0)),
            ("Rank prior+2Self", (1.0, 2.0, 0.0)),
            ("Rank prior+2AE", (1.0, 0.0, 2.0)),
            ("Rank prior+2Self+2AE", (1.0, 2.0, 2.0)),
            ("Rank prior+4Self+2AE", (1.0, 4.0, 2.0)),
            ("Rank prior+2Self+4AE", (1.0, 2.0, 4.0)),
        ]
    else:
        weights = [0.25, 0.5, 1.0, 2.0, 4.0]
        for ws in weights:
            combo_specs.append((f"Rank prior+{ws:g}Self", (1.0, ws, 0.0)))
        for wa in weights:
            combo_specs.append((f"Rank prior+{wa:g}AE", (1.0, 0.0, wa)))
        for ws in weights:
            for wa in weights:
                combo_specs.append((f"Rank prior+{ws:g}Self+{wa:g}AE", (1.0, ws, wa)))
        for ws in weights:
            for wa in weights:
                combo_specs.append((f"Rank {ws:g}Self+{wa:g}AE", (0.0, ws, wa)))

    for dataset in DATASETS:
        for model in MODELS:
            pred_path = args.exp_dir / "rescue" / f"{dataset}_{model}_seed2021_predictions.npz"
            z = np.load(pred_path)
            gt = z["gt"].astype(bool).reshape(-1)
            temporal = z["temporal_pred"].astype(bool).reshape(-1)
            baseline = z["baseline_pred"].astype(bool).reshape(-1)
            current = z["final_pred"].astype(bool).reshape(-1)
            prior = z["event_prior"].astype(np.float64).reshape(-1)
            candidates = z["candidate_indices"].astype(np.int64).reshape(-1)
            candidates = candidates[(candidates >= 0) & (candidates < len(gt))]
            candidates = candidates[~temporal[candidates]]
            candidates = np.unique(candidates)
            selected = z["selected_peaks"].astype(np.int64).reshape(-1) if "selected_peaks" in z.files else np.array([], dtype=np.int64)
            budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

            self_z = np.load(find_score_file(self_dir, dataset, model))
            ae_z = np.load(find_score_file(convae_dir, dataset, model))
            self_candidates = self_z["candidates"].astype(np.int64).reshape(-1)
            ae_candidates = ae_z["candidates"].astype(np.int64).reshape(-1)
            if not np.array_equal(candidates, self_candidates) or not np.array_equal(candidates, ae_candidates):
                raise ValueError(f"candidate mismatch for {dataset}/{model}")
            self_score = self_z["self_event_score"].astype(np.float64).reshape(-1)
            ae_score = ae_z["center_event_score"].astype(np.float64).reshape(-1) if "center_event_score" in ae_z.files else ae_z["prior"].astype(np.float64).reshape(-1)
            if "center_event_score" not in ae_z.files:
                # ConvAE script stores ae score implicitly only in pair metrics in
                # older runs.  Fail loudly rather than silently fusing the wrong signal.
                raise KeyError(f"{find_score_file(convae_dir, dataset, model)} lacks center_event_score")
            prior_c = prior[candidates]
            r_prior = rank01(prior_c)
            r_self = rank01(self_score)
            r_ae = rank01(ae_score)

            variants = {
                "Baseline": baseline,
                "+ Temporal calibration": temporal,
                "Prior + NMS": select_from_candidate_score(temporal, candidates, prior_c, budget),
                "Current NER": current,
            }
            for name, (wp, ws, wa) in combo_specs:
                score = wp * r_prior + ws * r_self + wa * r_ae
                variants[name] = select_from_candidate_score(temporal, candidates, score, budget)

            for variant, pred in variants.items():
                row = {"dataset": dataset, "model": model, "variant": variant, "budget": budget, "candidate_count": len(candidates)}
                row.update(evaluate(pred, gt))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "neural_score_fusion_pair_metrics.csv", index=False)
    summary = df.groupby("variant", as_index=False)[["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]].mean(numeric_only=True)
    for c in ["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]:
        summary[c] *= 100
    summary = summary.sort_values(["event_f1", "range_f1", "pa_f1"], ascending=False)
    summary.to_csv(out_dir / "neural_score_fusion_summary.csv", index=False)
    print(summary.head(30).to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
