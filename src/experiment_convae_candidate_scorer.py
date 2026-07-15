"""Prototype a normal-window ConvAE scorer for event-candidate ranking.

This script is intentionally experimental.  It keeps the existing temporal
calibration and candidate pool fixed, trains a small convolutional autoencoder
on normal training windows, and uses reconstruction error to re-rank event
candidate centers.

Goal: test whether a genuinely neural local-structure score can improve over
the hand-crafted event prior on representative pairs before changing the full
pipeline.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support


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


class ConvWindowAE(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, bottleneck: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, bottleneck, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(bottleneck, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, C]
        z = x.transpose(1, 2).contiguous()
        y = self.net(z).transpose(1, 2).contiguous()
        return y


def make_feature_series(x: np.ndarray, train_ref: np.ndarray | None = None, mode: str = "value") -> np.ndarray:
    if mode == "value":
        return x.astype(np.float32)
    if train_ref is None:
        train_ref = x
    train_diff = np.diff(train_ref, axis=0, prepend=train_ref[:1])
    diff = np.diff(x, axis=0, prepend=x[:1])
    scale = train_diff.std(axis=0) + 1e-6
    z_diff = diff / scale[None, :]
    if mode == "diff":
        return z_diff.astype(np.float32)
    if mode == "geometry":
        return np.concatenate([x, z_diff, np.abs(z_diff)], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature mode: {mode}")


def make_windows(x: np.ndarray, centers: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    return np.stack([padded[int(c) : int(c) + window] for c in centers], axis=0).astype(np.float32)


def sample_train_windows(train_z: np.ndarray, window: int, n_windows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = window // 2
    lo, hi = half, len(train_z) - half
    if hi <= lo:
        raise ValueError(f"train sequence too short for window={window}")
    centers = rng.integers(lo, hi, size=n_windows)
    return make_windows(train_z, centers, window)


def train_ae(
    train_z: np.ndarray,
    window: int,
    n_windows: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden: int,
    bottleneck: int,
    noise_std: float,
) -> ConvWindowAE:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = sample_train_windows(train_z, window, n_windows, seed)
    ds = TensorDataset(torch.from_numpy(x))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    model = ConvWindowAE(train_z.shape[1], hidden=hidden, bottleneck=bottleneck).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            if noise_std > 0:
                xb_in = xb + noise_std * torch.randn_like(xb)
            else:
                xb_in = xb
            rec = model(xb_in)
            loss = ((rec - xb) ** 2).mean()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def score_windows_ae(model: ConvWindowAE, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    w = windows.shape[1]
    center_slice = slice(max(0, w // 2 - 3), min(w, w // 2 + 4))
    for s in range(0, len(windows), batch_size):
        xb = torch.from_numpy(windows[s : s + batch_size]).to(device)
        rec = model(xb)
        err = ((rec - xb) ** 2).detach().cpu().numpy()
        # Emphasize the candidate center; this matches event-center ranking.
        scores.append(0.5 * err.mean(axis=(1, 2)) + 0.5 * err[:, center_slice, :].mean(axis=(1, 2)))
    return np.concatenate(scores).astype(np.float64) if scores else np.array([], dtype=np.float64)


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
    p, r, f1, _ = precision_recall_fscore_support(gt.astype(int), pred.astype(int), average="binary", zero_division=0)
    return float(p), float(r), float(f1)


def overlap_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


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
    pp, pr, pf1 = binary_f1(pred, gt)
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


def select_from_candidate_score(temporal: np.ndarray, candidates: np.ndarray, candidate_score: np.ndarray, budget: int) -> np.ndarray:
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
    parser.add_argument("--train_windows", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--score_batch_size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--feature_mode", choices=["value", "diff", "geometry"], default="geometry")
    parser.add_argument("--noise_std", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = args.output_dir or args.exp_dir / "convae_candidate_scorer"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = out_dir / "candidate_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    dtpp = load_module(Path("scripts/run_dtpp_from_at_scores.py"), "dtpp_convae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = parse_pairs(args.pairs)
    models_by_dataset: dict[str, ConvWindowAE] = {}
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for dataset, model_name in pairs:
        pred_path = args.exp_dir / "rescue" / f"{dataset}_{model_name}_seed2021_predictions.npz"
        z = np.load(pred_path)
        gt = z["gt"].astype(bool).reshape(-1)
        train_z_raw, test_z_full_raw, labels = dtpp.load_dataset(args.data_root / dataset, dataset, 10**18)
        train_z = make_feature_series(train_z_raw, train_z_raw, mode=args.feature_mode)
        test_z_full = make_feature_series(test_z_full_raw, train_z_raw, mode=args.feature_mode)
        test_z = test_z_full[: len(gt)]
        if not np.array_equal(labels[: len(gt)].astype(bool), gt):
            raise ValueError(f"label mismatch for {dataset}/{model_name}")
        data_cache[dataset] = (train_z, test_z, labels[: len(gt)])
        if dataset not in models_by_dataset:
            print(f"training ConvAE for {dataset} on {device}", flush=True)
            models_by_dataset[dataset] = train_ae(
                train_z=train_z,
                window=args.window,
                n_windows=args.train_windows,
                seed=args.seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                bottleneck=args.bottleneck,
                noise_std=args.noise_std,
            )

        temporal = z["temporal_pred"].astype(bool).reshape(-1)
        baseline = z["baseline_pred"].astype(bool).reshape(-1)
        ours = z["final_pred"].astype(bool).reshape(-1)
        prior = z["event_prior"].astype(np.float64).reshape(-1)
        saved_neural = z["neural_rescue_score"].astype(np.float64).reshape(-1)
        candidates = z["candidate_indices"].astype(np.int64).reshape(-1)
        candidates = candidates[(candidates >= 0) & (candidates < len(gt))]
        candidates = candidates[~temporal[candidates]]
        candidates = np.unique(candidates)
        selected = z["selected_peaks"].astype(np.int64).reshape(-1) if "selected_peaks" in z.files else np.array([], dtype=np.int64)
        budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

        print(f"scoring {dataset}/{model_name}: candidates={len(candidates)} budget={budget}", flush=True)
        windows = make_windows(test_z, candidates, args.window) if len(candidates) else np.empty((0, args.window, test_z.shape[1]), dtype=np.float32)
        ae_score = score_windows_ae(models_by_dataset[dataset], windows, device, args.score_batch_size)
        prior_c = prior[candidates]
        saved_c = saved_neural[candidates]
        np.savez_compressed(
            score_dir / f"{dataset}_{model_name}_candidate_scores.npz",
            candidates=candidates.astype(np.int32),
            prior=prior_c.astype(np.float32),
            center_event_score=ae_score.astype(np.float32),
            saved_neural=saved_c.astype(np.float32),
            budget=np.array([budget], dtype=np.int32),
        )

        variants: dict[str, np.ndarray] = {
            "Baseline": baseline,
            "+ Temporal calibration": temporal,
            "Prior + NMS": select_from_candidate_score(temporal, candidates, prior_c, budget),
            "Current NER": ours,
            "ConvAE only": select_from_candidate_score(temporal, candidates, ae_score, budget),
        }
        if len(candidates):
            variants["RankSum prior+AE"] = select_from_candidate_score(temporal, candidates, rank01(prior_c) + rank01(ae_score), budget)
            variants["RankSum prior+2AE"] = select_from_candidate_score(temporal, candidates, rank01(prior_c) + 2.0 * rank01(ae_score), budget)
            variants["RankSum 2prior+AE"] = select_from_candidate_score(temporal, candidates, 2.0 * rank01(prior_c) + rank01(ae_score), budget)
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
                "train_windows": args.train_windows,
                "epochs": args.epochs,
            }
            row.update(evaluate(pred, gt))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "convae_candidate_scorer_pair_metrics.csv", index=False)
    summary = df.groupby("variant", as_index=False)[["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]].mean(numeric_only=True)
    for c in ["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]:
        summary[c] *= 100.0
    summary = summary.sort_values(["event_f1", "range_f1", "pa_f1"], ascending=False)
    summary.to_csv(out_dir / "convae_candidate_scorer_summary.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
