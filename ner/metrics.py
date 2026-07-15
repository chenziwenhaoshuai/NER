from __future__ import annotations

import numpy as np


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    values = mask.astype(bool).reshape(-1)
    if len(values) == 0:
        return []
    transitions = np.diff(values.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def point_adjust(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = pred.astype(bool).copy()
    for start, end in segments(gt):
        if out[start : end + 1].any():
            out[start : end + 1] = True
    return out


def binary_prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def event_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_segments = segments(pred)
    gt_segments = segments(gt)
    matched_gt: set[int] = set()
    matched_pred = 0
    gt_start = 0
    for pred_start, pred_end in pred_segments:
        while gt_start < len(gt_segments) and gt_segments[gt_start][1] < pred_start:
            gt_start += 1
        index = gt_start
        while index < len(gt_segments) and gt_segments[index][0] <= pred_end:
            matched_gt.add(index)
            matched_pred += 1
            break
    precision = matched_pred / len(pred_segments) if pred_segments else 0.0
    recall = len(matched_gt) / len(gt_segments) if gt_segments else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def range_prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_segments = segments(pred)
    gt_segments = segments(gt)
    gt_prefix = np.concatenate([[0], np.cumsum(gt.astype(np.int64))])
    pred_prefix = np.concatenate([[0], np.cumsum(pred.astype(np.int64))])
    precision = (
        float(
            np.mean(
                [
                    (gt_prefix[end + 1] - gt_prefix[start]) / max(end - start + 1, 1)
                    for start, end in pred_segments
                ]
            )
        )
        if pred_segments
        else 0.0
    )
    recall = (
        float(
            np.mean(
                [
                    (pred_prefix[end + 1] - pred_prefix[start]) / max(end - start + 1, 1)
                    for start, end in gt_segments
                ]
            )
        )
        if gt_segments
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def strict_event_prf(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[float, float, float, float]:
    pred_segments = segments(pred)
    gt_segments = segments(gt)
    candidates: list[tuple[int, int, int]] = []
    for pred_index, (pred_start, pred_end) in enumerate(pred_segments):
        for gt_index, (gt_start, gt_end) in enumerate(gt_segments):
            overlap = max(0, min(pred_end, gt_end) - max(pred_start, gt_start) + 1)
            if overlap:
                candidates.append((overlap, pred_index, gt_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
    tp = len(used_gt)
    fp = len(pred_segments) - len(used_pred)
    fn = len(gt_segments) - len(used_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_events_per_100k = fp / max(len(gt), 1) * 100000.0
    return precision, recall, f1, false_events_per_100k


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pa_precision, pa_recall, pa_f1 = binary_prf(point_adjust(pred, gt), gt)
    _, _, point_f1 = binary_prf(pred, gt)
    range_precision, range_recall, range_f1 = range_prf(pred, gt)
    strict_precision, strict_recall, strict_f1, false_events = strict_event_prf(pred, gt)
    fp = int(np.logical_and(pred, ~gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {
        "pa_precision": pa_precision,
        "pa_recall": pa_recall,
        "pa_f1": pa_f1,
        "point_f1": point_f1,
        "event_f1": event_f1(pred, gt),
        "range_precision": range_precision,
        "range_recall": range_recall,
        "range_f1": range_f1,
        "strict_event_precision": strict_precision,
        "strict_event_recall": strict_recall,
        "strict_event_f1": strict_f1,
        "false_events_per_100k": false_events,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "pred_points": int(pred.sum()),
    }

