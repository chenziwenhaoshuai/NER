"""Apply Neural Event Rescue to any reconstruction-score anomaly baseline.

Input contract:
  - score NPZ with at_thre_flat_* arrays by default, or explicit score axis
  - dataset arrays in the Anomaly-Transformer/TSLib format

The script is baseline-agnostic: it never assumes the baseline architecture.
It evaluates:
  1. baseline raw prediction under point-adjust
  2. + Temporal calibration
  3. + Learnable rescue
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_neural_event_rescue_v8_smd import (
    empirical_tail_rank,
    make_windows_from_centers,
    neural_event_prior,
    score_windows,
    train_head,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(dtpp, pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float | int]:
    gt_segs = dtpp.segments(gt)
    pa = dtpp.point_adjust(pred.astype(bool), gt_segs)
    out = {f"{prefix}_pred_points": int(pred.astype(bool).sum()), f"{prefix}_pa_pred_points": int(pa.sum())}
    for k, v in dtpp.binary_metrics(pa, gt, prefix).items():
        out[k] = v
    for k, v in dtpp.event_metrics(pred.astype(bool), gt).items():
        out[f"{prefix}_{k}"] = v
    return out


def flatten_nonoverlap(x: np.ndarray, win_size: int) -> np.ndarray:
    n_windows = (len(x) - win_size) // win_size + 1
    if n_windows <= 0:
        raise ValueError(f"Cannot make non-overlap windows: len={len(x)}, win_size={win_size}")
    return np.concatenate([x[i * win_size : i * win_size + win_size] for i in range(n_windows)], axis=0)


def align_baseline_arrays(raw, train_z, test_z_full, labels_full, score_axis: str, win_size: int):
    if score_axis == "at_thre_flat":
        required = [
            "at_thre_flat_train_energy",
            "at_thre_flat_test_energy",
            "at_thre_flat_raw_pred",
            "at_thre_flat_gt",
        ]
        missing = [k for k in required if k not in raw]
        if missing:
            raise KeyError(f"score_axis=at_thre_flat requires missing arrays: {missing}")
        npz_gt = raw["at_thre_flat_gt"].astype(int).reshape(-1)
        test_energy = raw["at_thre_flat_test_energy"].astype(np.float64).reshape(-1)
        raw_pred = raw["at_thre_flat_raw_pred"].astype(bool).reshape(-1)
        train_energy = raw["at_thre_flat_train_energy"].astype(np.float64).reshape(-1)
        test_z = flatten_nonoverlap(test_z_full, win_size)
        labels = flatten_nonoverlap(labels_full.reshape(-1, 1), win_size).reshape(-1).astype(int)
    elif score_axis == "timeline":
        required = ["timeline_train_energy", "timeline_test_energy", "timeline_raw_pred", "timeline_gt"]
        missing = [k for k in required if k not in raw]
        if missing:
            raise KeyError(f"score_axis=timeline requires missing arrays: {missing}")
        npz_gt = raw["timeline_gt"].astype(int).reshape(-1)
        test_energy = raw["timeline_test_energy"].astype(np.float64).reshape(-1)
        raw_pred = raw["timeline_raw_pred"].astype(bool).reshape(-1)
        train_energy = raw["timeline_train_energy"].astype(np.float64).reshape(-1)
        test_z = test_z_full
        labels = labels_full.astype(int).reshape(-1)
    elif score_axis == "legacy":
        npz_gt = raw["gt"].astype(int).reshape(-1)
        test_energy = raw["test_energy"].astype(np.float64).reshape(-1)
        raw_pred = raw["raw_pred"].astype(bool).reshape(-1)
        train_energy = raw["train_energy"].astype(np.float64).reshape(-1)
        test_z = test_z_full
        labels = labels_full.astype(int).reshape(-1)
    else:
        raise ValueError(f"Unknown score_axis: {score_axis}")

    n = min(len(npz_gt), len(test_energy), len(raw_pred), len(test_z), len(labels))
    gt = labels[:n].astype(int)
    npz_gt = npz_gt[:n].astype(int)
    return train_energy, test_energy[:n], raw_pred[:n], gt, npz_gt, test_z[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["SMAP", "MSL", "SMD", "PSM", "SWaT"])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--score_npz", required=True, type=Path)
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--score_axis", choices=["at_thre_flat", "timeline", "legacy"], default="at_thre_flat")
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--n_pos", type=int, default=40000)
    parser.add_argument("--n_neg", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--score_batch_size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--candidate_pool", type=int, default=30000)
    parser.add_argument("--nms_radius", type=int, default=2500)
    parser.add_argument("--budget", type=int, default=-1, help="If <0, use ceil(n * budget_ratio).")
    parser.add_argument("--budget_ratio", type=float, default=2e-5)
    parser.add_argument("--min_budget", type=int, default=2)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtpp = load_module(SCRIPT_DIR / "run_dtpp_from_at_scores.py", "dtpp_any_baseline")
    v61 = load_module(SCRIPT_DIR / "adaptive_dtpp_v61_detector.py", "v61_any_baseline")
    detector = v61.AdaptiveDTPPv61Detector()
    raw = np.load(args.score_npz)

    train_z, test_z_full, labels_full = dtpp.load_dataset(args.data_path, args.dataset, 10**18)
    train_energy, test_energy, base_pred, gt, npz_gt, test_z = align_baseline_arrays(
        raw, train_z, test_z_full, labels_full, args.score_axis, args.win_size
    )
    label_match = bool(np.array_equal(gt.astype(int), npz_gt.astype(int)))
    if not label_match:
        raise ValueError(
            f"Dataset labels do not match NPZ labels under score_axis={args.score_axis}; "
            f"gt_points={int(gt.sum())}, npz_gt_points={int(npz_gt.sum())}, n={len(gt)}"
        )

    temporal_result = detector.predict(train_z, test_z, base_pred, train_energy, test_energy)
    temporal_pred = temporal_result.final_pred.astype(bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model = train_head(
        train_z=train_z,
        window=args.window,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
    )
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    energy_rank = empirical_tail_rank(train_energy, test_energy).reshape(-1, 1)
    pred_channel = temporal_pred.astype(np.float32).reshape(-1, 1)
    extra = np.concatenate([energy_rank, pred_channel], axis=1)
    prior = neural_event_prior(train_z, test_z, train_energy, test_energy)
    prior_peaks = dtpp.nms_order(prior, radius=args.nms_radius)
    candidate_mask = ~temporal_pred
    candidate_indices = prior_peaks[candidate_mask[prior_peaks]][: min(args.candidate_pool, int(candidate_mask.sum()))]
    if len(candidate_indices):
        windows = make_windows_from_centers(test_z, candidate_indices, args.window, extra_channels=extra)
        proba = score_windows(model, windows, device=device, batch_size=args.score_batch_size)
    else:
        proba = np.array([], dtype=np.float64)
    full_score = np.zeros(len(gt), dtype=np.float64)
    full_score[candidate_indices] = prior[candidate_indices] * ((0.5 + 0.5 * proba) ** args.prior_weight)
    budget = args.budget
    if budget < 0:
        budget = max(args.min_budget, int(np.ceil(len(gt) * args.budget_ratio)))
    selected = candidate_indices[np.argsort(full_score[candidate_indices])[::-1]][:budget]
    final_pred = temporal_pred.copy()
    final_pred[selected] = True
    infer_seconds = time.perf_counter() - t1
    peak_cuda_mb = float("nan")
    if device.type == "cuda":
        peak_cuda_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    row: dict[str, float | int | str | bool] = {
        "dataset": args.dataset,
        "baseline": args.baseline,
        "score_npz": str(args.score_npz),
        "seed": args.seed,
        "device": str(device),
        "n_points": int(len(gt)),
        "score_axis": args.score_axis,
        "win_size": args.win_size,
        "gt_points": int(gt.sum()),
        "npz_gt_points": int(npz_gt.sum()),
        "label_match": bool(label_match),
        "window": args.window,
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "epochs": args.epochs,
        "candidate_pool": args.candidate_pool,
        "nms_radius": args.nms_radius,
        "budget": int(budget),
        "selected_count": int(len(selected)),
        "selected_indices": ";".join(map(str, selected.tolist())),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(infer_seconds),
        "total_seconds": float(train_seconds + infer_seconds),
        "peak_cuda_memory_mb": peak_cuda_mb,
        "candidate_count": int(len(candidate_indices)),
        "scored_windows": int(len(candidate_indices)),
        "train_points": int(len(train_z)),
    }
    row.update(evaluate(dtpp, base_pred, gt, "baseline"))
    row.update(evaluate(dtpp, temporal_pred, gt, "temporal"))
    row.update(evaluate(dtpp, final_pred, gt, "ours"))
    row["temporal_delta_f1_pct"] = (row["temporal_f1"] - row["baseline_f1"]) * 100.0
    row["ours_delta_f1_pct"] = (row["ours_f1"] - row["baseline_f1"]) * 100.0
    row["ours_vs_temporal_delta_f1_pct"] = (row["ours_f1"] - row["temporal_f1"]) * 100.0

    run_id = f"{args.dataset}_{args.baseline}_seed{args.seed}"
    np.savez_compressed(
        args.output_dir / f"{run_id}_predictions.npz",
        gt=gt.astype(np.int8),
        baseline_pred=base_pred.astype(np.int8),
        temporal_pred=temporal_pred.astype(np.int8),
        final_pred=final_pred.astype(np.int8),
        final_point_adjust_pred=dtpp.point_adjust(final_pred, dtpp.segments(gt)).astype(np.int8),
        train_energy=train_energy.astype(np.float32),
        test_energy=test_energy.astype(np.float32),
        event_prior=prior.astype(np.float32),
        neural_rescue_score=full_score.astype(np.float32),
        candidate_indices=candidate_indices.astype(np.int32),
        selected_peaks=selected.astype(np.int32),
    )
    pd.DataFrame([row]).to_csv(args.output_dir / f"{run_id}_summary.csv", index=False)
    (args.output_dir / f"{run_id}_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame([row])[[
        "dataset",
        "baseline",
        "baseline_f1",
        "temporal_f1",
        "ours_f1",
        "temporal_delta_f1_pct",
        "ours_delta_f1_pct",
        "ours_vs_temporal_delta_f1_pct",
        "budget",
        "selected_count",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
