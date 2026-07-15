"""Train a center-aware neural event ranker for rescue-candidate selection.

This experiment addresses a weakness of the earlier neural rescue head:
it learned whether a local window looks anomalous, but not whether the
candidate *center* is a good event decision.  That mismatch can improve
PA-F1 while fragmenting event-level decisions.

This script keeps the saved profile fixed:

  - same baseline predictions
  - same temporal calibration predictions
  - same candidate pool
  - same rescue budget

It trains one neural ranker per dataset using only normal training windows
and synthetic corruptions.  Positives contain a synthetic event centered at
the candidate position.  Hard negatives contain either no corruption or a
similar corruption away from the center, forcing the network to learn
center-localized event geometry rather than generic window abnormality.

Labels are used only for final evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CenterEventRanker(nn.Module):
    """Small TCN-style ranker with explicit center/context readout."""

    def __init__(self, channels: int, hidden: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=8),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, C]
        z = self.encoder(x.transpose(1, 2).contiguous())  # [B,H,W]
        w = z.shape[-1]
        center = w // 2
        center_z = z[:, :, center]
        local = z[:, :, max(0, center - 3) : min(w, center + 4)].amax(dim=-1)
        global_max = z.amax(dim=-1)
        global_mean = z.mean(dim=-1)
        return self.head(torch.cat([center_z, local, global_max, global_mean], dim=1)).squeeze(-1)


def make_value_windows(x: np.ndarray, centers: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    return np.stack([padded[int(c) : int(c) + window] for c in centers], axis=0).astype(np.float32)


def windows_to_features(value_windows: np.ndarray, diff_scale: np.ndarray, mode: str) -> np.ndarray:
    value = value_windows.astype(np.float32)
    if mode == "value":
        return np.clip(value, -12.0, 12.0)
    diff = np.diff(value, axis=1, prepend=value[:, :1, :]) / diff_scale.reshape(1, 1, -1)
    diff = np.clip(diff, -12.0, 12.0).astype(np.float32)
    if mode == "diff":
        return diff
    if mode == "geometry":
        return np.concatenate([np.clip(value, -12.0, 12.0), diff, np.abs(diff)], axis=2).astype(np.float32)
    raise ValueError(f"unknown feature_mode={mode}")


def corrupt_window_at(
    out: np.ndarray,
    i: int,
    rng: np.random.Generator,
    center: int,
    max_channels: int,
    force_away_from_center: bool,
) -> None:
    w, c = out.shape[1], out.shape[2]
    k = int(rng.integers(1, min(c, max_channels) + 1))
    ch = rng.choice(c, size=k, replace=False)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=k)
    mag = rng.uniform(2.0, 7.0, size=k).astype(np.float32)

    if force_away_from_center:
        left_hi = max(2, center - 10)
        right_lo = min(w - 3, center + 10)
        if rng.random() < 0.5 and left_hi > 2:
            anchor = int(rng.integers(2, left_hi))
        elif right_lo < w - 3:
            anchor = int(rng.integers(right_lo, w - 2))
        else:
            anchor = int(rng.choice([max(2, center - 12), min(w - 3, center + 12)]))
    else:
        anchor = center

    kind = int(rng.integers(0, 6))
    def sample_span(low: int, high_cap: int) -> int:
        max_len = min(high_cap, w - anchor)
        if max_len <= 1:
            return 1
        low = min(low, max_len)
        return int(rng.integers(low, max_len + 1))

    if kind == 0:  # spike
        out[i, anchor, ch] += sign * mag
    elif kind == 1:  # short burst
        span = sample_span(3, 17)
        rows = np.arange(anchor, min(w, anchor + span))
        out[i][np.ix_(rows, ch)] += sign.reshape(1, -1) * mag.reshape(1, -1)
    elif kind == 2:  # local level shift
        span = sample_span(8, 31)
        rows = np.arange(anchor, min(w, anchor + span))
        out[i][np.ix_(rows, ch)] += sign.reshape(1, -1) * rng.uniform(1.5, 5.5, size=(1, k))
    elif kind == 3:  # dropout / stuck channel
        span = sample_span(5, 25)
        rows = np.arange(anchor, min(w, anchor + span))
        out[i][np.ix_(rows, ch)] = rng.normal(0.0, 0.05, size=(len(rows), k))
    elif kind == 4:  # variance burst
        span = sample_span(5, 25)
        rows = np.arange(anchor, min(w, anchor + span))
        out[i][np.ix_(rows, ch)] += rng.normal(0.0, mag.reshape(1, -1), size=(len(rows), k))
    else:  # ramp
        span = sample_span(8, 31)
        rows = np.arange(anchor, min(w, anchor + span))
        ramp = np.linspace(0.0, 1.0, len(rows), dtype=np.float32).reshape(-1, 1)
        out[i][np.ix_(rows, ch)] += ramp * sign.reshape(1, -1) * mag.reshape(1, -1)


def build_synthetic_dataset(
    train_z: np.ndarray,
    diff_scale: np.ndarray,
    window: int,
    n_pos: int,
    n_neg: int,
    seed: int,
    max_channels: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = window // 2
    lo, hi = half, len(train_z) - half
    if hi <= lo:
        raise ValueError(f"train sequence too short for window={window}")

    pos_centers = rng.integers(lo, hi, size=n_pos)
    neg_centers = rng.integers(lo, hi, size=n_neg)
    x_pos = make_value_windows(train_z, pos_centers, window)
    x_neg = make_value_windows(train_z, neg_centers, window)

    center = window // 2
    for i in range(n_pos):
        corrupt_window_at(x_pos, i, rng, center, max_channels=max_channels, force_away_from_center=False)

    # Half of negatives are normal; half are hard off-center corruptions.
    n_hard = n_neg // 2
    for i in range(n_hard):
        corrupt_window_at(x_neg, i, rng, center, max_channels=max_channels, force_away_from_center=True)

    x = np.concatenate(
        [windows_to_features(x_pos, diff_scale, feature_mode), windows_to_features(x_neg, diff_scale, feature_mode)],
        axis=0,
    )
    y = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)], axis=0)
    order = rng.permutation(len(y))
    return x[order].astype(np.float32), y[order].astype(np.float32)


def train_ranker(
    train_z: np.ndarray,
    window: int,
    n_pos: int,
    n_neg: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden: int,
    max_channels: int,
    feature_mode: str,
) -> tuple[CenterEventRanker, np.ndarray]:
    train_diff = np.diff(train_z, axis=0, prepend=train_z[:1])
    diff_scale = (train_diff.std(axis=0) + 1e-6).astype(np.float32)
    x, y = build_synthetic_dataset(
        train_z=train_z,
        diff_scale=diff_scale,
        window=window,
        n_pos=n_pos,
        n_neg=n_neg,
        seed=seed,
        max_channels=max_channels,
        feature_mode=feature_mode,
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = CenterEventRanker(x.shape[2], hidden=hidden).to(device)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model, diff_scale


@torch.no_grad()
def score_ranker(
    model: CenterEventRanker,
    value_windows: np.ndarray,
    diff_scale: np.ndarray,
    feature_mode: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(value_windows) == 0:
        return np.array([], dtype=np.float64)
    x = windows_to_features(value_windows, diff_scale, feature_mode)
    model.eval()
    outs = []
    for s in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[s : s + batch_size]).to(device)
        outs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


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


def binary_f1(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        gt.astype(int), pred.astype(int), average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


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


def parse_pairs(text: str) -> list[tuple[str, str]]:
    if text.lower() == "all":
        return [(d, m) for d in DATASETS for m in MODELS]
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"pair must be DATASET:MODEL, got {item}")
        d, m = item.split(":", 1)
        out.append((d, m))
    return out


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
    parser.add_argument("--data_root", type=Path, default=Path("datasets/anomaly_transformer"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--pairs", default="PSM:Autoformer,SMD:Autoformer,MSL:AnomalyTransformer")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--n_pos", type=int, default=40000)
    parser.add_argument("--n_neg", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--score_batch_size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--max_channels", type=int, default=8)
    parser.add_argument("--feature_mode", choices=["value", "diff", "geometry"], default="geometry")
    args = parser.parse_args()

    out_dir = args.output_dir or args.exp_dir / "center_event_ranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = out_dir / "candidate_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    dtpp = load_module(
        Path(__file__).resolve().parent / "run_dtpp_from_at_scores.py",
        "dtpp_center_event_ranker",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = parse_pairs(args.pairs)

    models_by_dataset: dict[str, tuple[CenterEventRanker, np.ndarray]] = {}
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    lambdas = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    product_alphas = [0.25, 0.5, 1.0, 2.0]

    for dataset, model_name in pairs:
        pred_path = args.exp_dir / "rescue" / f"{dataset}_{model_name}_seed2021_predictions.npz"
        z = np.load(pred_path)
        gt = z["gt"].astype(bool).reshape(-1)

        if dataset not in data_cache:
            train_z, test_z_full, labels = dtpp.load_dataset(args.data_root / dataset, dataset, 10**18)
            data_cache[dataset] = (train_z.astype(np.float32), test_z_full.astype(np.float32), labels)
        train_z, test_z_full, labels = data_cache[dataset]
        test_z = test_z_full[: len(gt)]
        if not np.array_equal(labels[: len(gt)].astype(bool), gt):
            raise ValueError(f"label mismatch for {dataset}/{model_name}")

        if dataset not in models_by_dataset:
            print(f"training CenterEventRanker for {dataset} on {device}", flush=True)
            models_by_dataset[dataset] = train_ranker(
                train_z=train_z,
                window=args.window,
                n_pos=args.n_pos,
                n_neg=args.n_neg,
                seed=args.seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                max_channels=args.max_channels,
                feature_mode=args.feature_mode,
            )
        ranker, diff_scale = models_by_dataset[dataset]

        temporal = z["temporal_pred"].astype(bool).reshape(-1)
        baseline = z["baseline_pred"].astype(bool).reshape(-1)
        current_ner = z["final_pred"].astype(bool).reshape(-1)
        prior = z["event_prior"].astype(np.float64).reshape(-1)
        saved_neural = z["neural_rescue_score"].astype(np.float64).reshape(-1)
        candidates = z["candidate_indices"].astype(np.int64).reshape(-1)
        candidates = candidates[(candidates >= 0) & (candidates < len(gt))]
        candidates = candidates[~temporal[candidates]]
        candidates = np.unique(candidates)
        selected = z["selected_peaks"].astype(np.int64).reshape(-1) if "selected_peaks" in z.files else np.array([], dtype=np.int64)
        budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

        print(f"scoring {dataset}/{model_name}: candidates={len(candidates)} budget={budget}", flush=True)
        value_windows = make_value_windows(test_z, candidates, args.window) if len(candidates) else np.empty((0, args.window, test_z.shape[1]), dtype=np.float32)
        net_score = score_ranker(
            ranker,
            value_windows,
            diff_scale,
            feature_mode=args.feature_mode,
            device=device,
            batch_size=args.score_batch_size,
        )
        prior_c = prior[candidates]
        saved_c = saved_neural[candidates]
        prior_r = rank01(prior_c)
        net_r = rank01(net_score)

        np.savez_compressed(
            score_dir / f"{dataset}_{model_name}_candidate_scores.npz",
            candidates=candidates.astype(np.int32),
            prior=prior_c.astype(np.float32),
            center_event_score=net_score.astype(np.float32),
            saved_neural=saved_c.astype(np.float32),
            budget=np.array([budget], dtype=np.int32),
        )

        variants: dict[str, np.ndarray] = {
            "Baseline": baseline,
            "+ Temporal calibration": temporal,
            "Prior + NMS": select_from_candidate_score(temporal, candidates, prior_c, budget),
            "Current NER": current_ner,
            "CenterNet only": select_from_candidate_score(temporal, candidates, net_score, budget),
        }
        if len(candidates):
            for lam in lambdas:
                variants[f"RankSum prior+{lam:g}CenterNet"] = select_from_candidate_score(
                    temporal, candidates, prior_r + lam * net_r, budget
                )
            for alpha in product_alphas:
                variants[f"Prior*CenterNet^{alpha:g}"] = select_from_candidate_score(
                    temporal, candidates, prior_c * np.power(0.05 + 0.95 * net_score, alpha), budget
                )
            variants["Saved neural only"] = select_from_candidate_score(temporal, candidates, saved_c, budget)

        for variant, pred in variants.items():
            row = {
                "dataset": dataset,
                "model": model_name,
                "variant": variant,
                "budget": budget,
                "candidate_count": len(candidates),
                "device": str(device),
                "window": args.window,
                "n_pos": args.n_pos,
                "n_neg": args.n_neg,
                "epochs": args.epochs,
                "feature_mode": args.feature_mode,
            }
            row.update(evaluate(pred, gt))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "center_event_ranker_pair_metrics.csv", index=False)
    summary = df.groupby("variant", as_index=False)[["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]].mean(numeric_only=True)
    for c in ["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]:
        summary[c] *= 100.0
    summary = summary.sort_values(["event_f1", "range_f1", "pa_f1"], ascending=False)
    summary.to_csv(out_dir / "center_event_ranker_summary.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
