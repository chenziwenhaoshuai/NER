"""Augmented neural candidate scorer for event rescue.

This experiment is meant to make the neural module structurally meaningful,
not merely a fixed rank fusion over hand-crafted scores.

Compared with ``experiment_convae_candidate_scorer.py``, the autoencoder is
trained per detector--dataset pair on unlabeled low-evidence test windows and
its input contains both local multivariate geometry and detector-score
dynamics:

    [x, dx, |dx|, energy-rank, prior-rank, |d energy-rank|, |d prior-rank|]

The scorer is still label-free: ground-truth anomaly labels are used only for
evaluation.  Training windows are sampled outside the calibrated prediction and
from the lower-prior part of the sequence, so the network learns a local normal
score-geometry manifold for the current detector output.  Candidate windows
that reconstruct poorly are ranked as event centers.
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

from fuse_neural_candidate_scores import DATASETS, MODELS, evaluate, rank01, select_from_candidate_score
from experiment_self_trained_event_ranker import empirical_tail_rank, robust01


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ResidualConvAE(nn.Module):
    def __init__(self, channels: int, hidden: int = 96, bottleneck: int = 48, dropout: float = 0.05):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, bottleneck, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(bottleneck, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2).contiguous()
        y = self.dec(self.enc(z)).transpose(1, 2).contiguous()
        return y


def parse_pairs(text: str) -> list[tuple[str, str]]:
    if text.lower() == "all":
        return [(d, m) for d in DATASETS for m in MODELS]
    out: list[tuple[str, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"pair must be DATASET:MODEL, got {item}")
        d, m = item.split(":", 1)
        out.append((d, m))
    return out


def make_windows(x: np.ndarray, centers: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    return np.stack([padded[int(c) : int(c) + window] for c in centers], axis=0).astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0 or not mask.any():
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    for i in idx:
        out[max(0, i - radius) : min(len(mask), i + radius + 1)] = True
    return out


def make_augmented_series(
    train_z: np.ndarray,
    test_z: np.ndarray,
    train_energy: np.ndarray,
    test_energy: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    test_z = test_z.astype(np.float32)
    train_diff = np.diff(train_z.astype(np.float32), axis=0, prepend=train_z[:1].astype(np.float32))
    test_diff = np.diff(test_z, axis=0, prepend=test_z[:1])
    scale = train_diff.std(axis=0) + 1e-6
    diff = np.clip(test_diff / scale[None, :], -12.0, 12.0).astype(np.float32)
    value = np.clip(test_z, -12.0, 12.0).astype(np.float32)

    energy_rank = empirical_tail_rank(train_energy, test_energy).astype(np.float32)
    prior_rank = robust01(prior).astype(np.float32)
    energy_d = np.abs(np.diff(energy_rank, prepend=energy_rank[:1])).astype(np.float32)
    prior_d = np.abs(np.diff(prior_rank, prepend=prior_rank[:1])).astype(np.float32)
    score_channels = np.stack([energy_rank, prior_rank, energy_d, prior_d], axis=1)
    return np.concatenate([value, diff, np.abs(diff), score_channels], axis=1).astype(np.float32)


def sample_normal_centers(
    temporal: np.ndarray,
    prior: np.ndarray,
    window: int,
    n_windows: int,
    seed: int,
    exclusion_radius: int,
    prior_quantile: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(temporal)
    half = window // 2
    valid = np.zeros(n, dtype=bool)
    valid[half : n - half] = True
    base = valid & ~dilate_mask(temporal, exclusion_radius)
    if base.any():
        threshold = float(np.quantile(prior[base], prior_quantile))
        pool = np.flatnonzero(base & (prior <= threshold))
    else:
        pool = np.flatnonzero(valid & ~temporal)
    if len(pool) == 0:
        pool = np.flatnonzero(valid)
    return rng.choice(pool, size=n_windows, replace=len(pool) < n_windows).astype(np.int64)


def train_ae(
    feature_series: np.ndarray,
    train_centers: np.ndarray,
    window: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden: int,
    bottleneck: int,
    noise_std: float,
) -> ResidualConvAE:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = make_windows(feature_series, train_centers, window)
    ds = TensorDataset(torch.from_numpy(x))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    model = ResidualConvAE(feature_series.shape[1], hidden=hidden, bottleneck=bottleneck).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            xb_in = xb + noise_std * torch.randn_like(xb) if noise_std > 0 else xb
            opt.zero_grad(set_to_none=True)
            rec = model(xb_in)
            loss = torch.nn.functional.smooth_l1_loss(rec, xb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def score_ae(model: ResidualConvAE, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if len(windows) == 0:
        return np.array([], dtype=np.float64)
    model.eval()
    scores: list[np.ndarray] = []
    w = windows.shape[1]
    center_slice = slice(max(0, w // 2 - 3), min(w, w // 2 + 4))
    for s in range(0, len(windows), batch_size):
        xb = torch.from_numpy(windows[s : s + batch_size]).to(device)
        rec = model(xb)
        err = torch.nn.functional.smooth_l1_loss(rec, xb, reduction="none").detach().cpu().numpy()
        full = err.mean(axis=(1, 2))
        center = err[:, center_slice, :].mean(axis=(1, 2))
        scores.append((0.35 * full + 0.65 * center).astype(np.float64))
    return np.concatenate(scores)


def find_score_file(root: Path, dataset: str, model: str) -> Path:
    matches = list(root.glob(f"**/candidate_scores/{dataset}_{model}_candidate_scores.npz"))
    if not matches:
        raise FileNotFoundError(f"missing candidate score for {dataset}/{model} under {root}")
    return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]


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
    parser.add_argument("--self_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--pairs", default="MSL:AnomalyTransformer,SMAP:AnomalyTransformer,SWaT:AnomalyTransformer")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--train_windows", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--score_batch_size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--bottleneck", type=int, default=48)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--exclusion_radius", type=int, default=32)
    parser.add_argument("--prior_quantile", type=float, default=0.70)
    args = parser.parse_args()

    out_dir = args.output_dir or args.exp_dir / "augmented_convae_candidate_scorer"
    score_dir = out_dir / "candidate_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    self_dir = args.self_dir or args.exp_dir / "self_event_ranker_all25_e30"

    dtpp = load_module(SCRIPT_DIR / "run_dtpp_from_at_scores.py", "dtpp_aug_convae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset, model_name in parse_pairs(args.pairs):
        z = np.load(args.exp_dir / "rescue" / f"{dataset}_{model_name}_seed2021_predictions.npz")
        gt = z["gt"].astype(bool).reshape(-1)
        if dataset not in data_cache:
            train_z, test_z_full, labels = dtpp.load_dataset(args.data_root / dataset, dataset, 10**18)
            data_cache[dataset] = (train_z.astype(np.float32), test_z_full[: len(gt)].astype(np.float32), labels[: len(gt)])
        train_z, test_z, labels = data_cache[dataset]
        if not np.array_equal(labels.astype(bool), gt):
            raise ValueError(f"label mismatch for {dataset}/{model_name}")

        temporal = z["temporal_pred"].astype(bool).reshape(-1)
        baseline = z["baseline_pred"].astype(bool).reshape(-1)
        current = z["final_pred"].astype(bool).reshape(-1)
        prior = z["event_prior"].astype(np.float64).reshape(-1)
        train_energy = z["train_energy"].astype(np.float64).reshape(-1)
        test_energy = z["test_energy"].astype(np.float64).reshape(-1)
        candidates = z["candidate_indices"].astype(np.int64).reshape(-1)
        candidates = candidates[(candidates >= 0) & (candidates < len(gt))]
        candidates = candidates[~temporal[candidates]]
        candidates = np.unique(candidates)
        selected = z["selected_peaks"].astype(np.int64).reshape(-1) if "selected_peaks" in z.files else np.array([], dtype=np.int64)
        budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

        print(f"training augmented ConvAE for {dataset}/{model_name} on {device}; candidates={len(candidates)} budget={budget}", flush=True)
        feature_series = make_augmented_series(train_z, test_z, train_energy, test_energy, prior)
        train_centers = sample_normal_centers(
            temporal=temporal,
            prior=prior,
            window=args.window,
            n_windows=args.train_windows,
            seed=args.seed,
            exclusion_radius=args.exclusion_radius,
            prior_quantile=args.prior_quantile,
        )
        model = train_ae(
            feature_series=feature_series,
            train_centers=train_centers,
            window=args.window,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden=args.hidden,
            bottleneck=args.bottleneck,
            noise_std=args.noise_std,
        )
        windows = make_windows(feature_series, candidates, args.window) if len(candidates) else np.empty((0, args.window, feature_series.shape[1]), dtype=np.float32)
        aug_score = score_ae(model, windows, device, args.score_batch_size)

        self_z = np.load(find_score_file(self_dir, dataset, model_name))
        if not np.array_equal(candidates, self_z["candidates"].astype(np.int64).reshape(-1)):
            raise ValueError(f"candidate mismatch for {dataset}/{model_name}")
        self_score = self_z["self_event_score"].astype(np.float64).reshape(-1)
        prior_c = prior[candidates]
        score_fixed = rank01(prior_c) + 2.0 * rank01(self_score) + 2.0 * rank01(aug_score)
        score_aug_main = 0.25 * rank01(prior_c) + rank01(self_score) + 3.0 * rank01(aug_score)

        np.savez_compressed(
            score_dir / f"{dataset}_{model_name}_candidate_scores.npz",
            candidates=candidates.astype(np.int32),
            prior=prior_c.astype(np.float32),
            self_event_score=self_score.astype(np.float32),
            augmented_convae_score=aug_score.astype(np.float32),
            fixed_fusion_score=score_fixed.astype(np.float32),
            aug_main_score=score_aug_main.astype(np.float32),
            budget=np.array([budget], dtype=np.int32),
        )

        variants = {
            "Baseline": baseline,
            "+ Temporal calibration": temporal,
            "Current NER": current,
            "Prior + NMS": select_from_candidate_score(temporal, candidates, prior_c, budget),
            "SelfNet only": select_from_candidate_score(temporal, candidates, self_score, budget),
            "AugConvAE only": select_from_candidate_score(temporal, candidates, aug_score, budget),
            "AugConvAE fixed fusion": select_from_candidate_score(temporal, candidates, score_fixed, budget),
            "AugConvAE main fusion": select_from_candidate_score(temporal, candidates, score_aug_main, budget),
        }
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
                "prior_quantile": args.prior_quantile,
            }
            row.update(evaluate(pred, gt))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "augmented_convae_pair_metrics.csv", index=False)
    summary = df.groupby("variant", as_index=False)[["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]].mean(numeric_only=True)
    for c in ["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]:
        summary[c] *= 100.0
    summary = summary.sort_values(["event_f1", "range_f1", "pa_f1"], ascending=False)
    summary.to_csv(out_dir / "augmented_convae_summary.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
