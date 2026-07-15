"""Dual-path Temporal Peak Proposer (DTPP) on top of saved AT scores.

This script is intentionally dataset-generic for NASA-style Anomaly Transformer
preprocessed arrays:

  DATASET_train.npy
  DATASET_test.npy
  DATASET_test_label.npy

It preserves a raw AT detector from a saved score NPZ and adds sparse temporal
difference peak proposals:

  final_pred = raw_at_pred OR NMS(train-calibrated diff peaks)

Labels are used only for audit metrics, not for selecting the train-calibrated
threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
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


def point_adjust(pred: np.ndarray, gt_segs: list[tuple[int, int]]) -> np.ndarray:
    pred = pred.astype(bool)
    out = pred.copy()
    for s, e in gt_segs:
        if pred[s : e + 1].any():
            out[s : e + 1] = True
    return out


def binary_metrics(pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        gt.astype(int), pred.astype(int), average="binary", zero_division=0
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(gt.astype(int), pred.astype(int))),
        f"{prefix}_precision": float(p),
        f"{prefix}_recall": float(r),
        f"{prefix}_f1": float(f1),
    }


def event_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred_segs = segments(pred)
    gt_segs = segments(gt)
    matched_gt = set()
    tp_pred = 0
    for ps, pe in pred_segs:
        for i, (gs, ge) in enumerate(gt_segs):
            if pe >= gs and ps <= ge:
                matched_gt.add(i)
                tp_pred += 1
                break
    ep = tp_pred / len(pred_segs) if pred_segs else 0.0
    er = len(matched_gt) / len(gt_segs) if gt_segs else 0.0
    ef = 2 * ep * er / (ep + er) if ep + er else 0.0
    return {
        "event_precision": float(ep),
        "event_recall": float(er),
        "event_f1": float(ef),
        "event_pred_count": int(len(pred_segs)),
        "event_gt_count": int(len(gt_segs)),
        "event_matched_gt_count": int(len(matched_gt)),
    }


def nms_order(score: np.ndarray, radius: int, max_peaks: int | None = None) -> np.ndarray:
    n = len(score)
    selected = []
    suppressed = np.zeros(n, dtype=bool)
    for idx in np.argsort(score)[::-1]:
        if suppressed[idx]:
            continue
        selected.append(idx)
        if max_peaks is not None and len(selected) >= max_peaks:
            break
        suppressed[max(0, idx - radius) : min(n, idx + radius + 1)] = True
    return np.array(selected, dtype=np.int64)


def load_dataset(data_path: Path, dataset: str, n_test: int):
    if dataset == "PSM":
        train = pd.read_csv(data_path / "train.csv").values[:, 1:]
        test = pd.read_csv(data_path / "test.csv").values[:, 1:]
        labels = pd.read_csv(data_path / "test_label.csv").values[:, 1:]
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
    else:
        train = np.load(data_path / f"{dataset}_train.npy")
        test = np.load(data_path / f"{dataset}_test.npy")
        labels = np.load(data_path / f"{dataset}_test_label.npy")
    labels = labels.astype(int).reshape(-1)[:n_test]
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train).astype(np.float64)
    test_z = scaler.transform(test).astype(np.float64)[:n_test]
    return train_z, test_z, labels


def diff_scores(train_z: np.ndarray, test_z: np.ndarray):
    train_diff = np.diff(train_z, axis=0, prepend=train_z[:1])
    test_diff = np.diff(test_z, axis=0, prepend=test_z[:1])
    diff_std = train_diff.std(axis=0) + 1e-6
    train_score = np.max(np.abs(train_diff / diff_std), axis=1)
    test_score = np.max(np.abs(test_diff / diff_std), axis=1)
    return train_score, test_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--at_scores", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--radius", default=2500, type=int)
    parser.add_argument("--train_quantile", default=99.0, type=float)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    raw = np.load(args.at_scores)
    gt = raw["gt"].astype(int).reshape(-1)
    n_test = len(gt)
    raw_pred = raw["raw_pred"].astype(bool).reshape(-1)
    raw_score = raw["test_energy"].astype(np.float64).reshape(-1)

    train_z, test_z, labels = load_dataset(args.data_path, args.dataset, n_test)
    if not np.array_equal(labels.astype(int), gt.astype(int)):
        raise ValueError("dataset labels do not match labels saved in AT score NPZ")

    train_diff_score, test_diff_score = diff_scores(train_z, test_z)
    peaks = nms_order(test_diff_score, radius=args.radius, max_peaks=args.topk)
    if args.topk is None:
        threshold = float(np.percentile(train_diff_score, args.train_quantile))
        chosen = peaks[test_diff_score[peaks] >= threshold]
        selector = "train_quantile"
    else:
        threshold = None
        chosen = peaks[: args.topk]
        selector = "topk"

    final_pred = raw_pred.copy()
    final_pred[chosen] = True
    pa_pred = point_adjust(final_pred, segments(gt))

    metrics: dict[str, float | int | str | None] = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "seed": args.seed,
        "base": str(args.at_scores),
        "method": "DTPP",
        "selector": selector,
        "radius": args.radius,
        "train_quantile": args.train_quantile if args.topk is None else None,
        "train_threshold": threshold,
        "topk": args.topk,
        "selected_peaks": int(len(chosen)),
        "n_points": int(len(gt)),
        "positive_points": int(gt.sum()),
        "raw_at_pred_points": int(raw_pred.sum()),
        "final_pred_points": int(final_pred.sum()),
        "pa_pred_points": int(pa_pred.sum()),
    }
    metrics.update(binary_metrics(final_pred, gt, "raw"))
    metrics.update(binary_metrics(pa_pred, gt, "point_adjust"))
    metrics.update(event_metrics(final_pred, gt))
    metrics["auc_pr_raw_score"] = float(average_precision_score(gt.astype(int), raw_score))
    metrics["auc_roc_raw_score"] = float(roc_auc_score(gt.astype(int), raw_score))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"{args.run_id}_predictions.npz",
        gt=gt.astype(np.int8),
        raw_at_pred=raw_pred.astype(np.int8),
        diff_score=test_diff_score.astype(np.float32),
        selected_peaks=chosen.astype(np.int32),
        final_raw_pred=final_pred.astype(np.int8),
        final_point_adjust_pred=pa_pred.astype(np.int8),
    )
    (args.output_dir / f"{args.run_id}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(args.output_dir / f"{args.run_id}_summary.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
