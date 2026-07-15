"""Self-trained neural event-prototype ranker for rescue candidates.

The center-synthetic ranker is label-free but too synthetic: it learns generic
local corruptions and can miss the morphology of real detector events.  This
experiment trains a neural ranker from unlabeled test-time evidence:

  positives: centers of high-confidence events already detected by the
             baseline/temporal detector;
  negatives: low-evidence centers outside detected events, plus boundary
             negatives around detected events.

No ground-truth labels are used for training or selection.  Labels are used
only for final evaluation.
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

from experiment_center_event_ranker import (
    CenterEventRanker,
    evaluate,
    make_value_windows,
    rank01,
    select_from_candidate_score,
    segments,
    windows_to_features,
)


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


def empirical_tail_rank(train_score: np.ndarray, test_score: np.ndarray) -> np.ndarray:
    train_sorted = np.sort(train_score.astype(np.float64).reshape(-1))
    right = np.searchsorted(train_sorted, test_score.astype(np.float64).reshape(-1), side="right")
    return right.astype(np.float64) / max(float(len(train_sorted)), 1.0)


def robust01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).reshape(-1)
    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def build_augmented_series(
    test_z: np.ndarray,
    train_energy: np.ndarray,
    test_energy: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    energy_rank = empirical_tail_rank(train_energy, test_energy).reshape(-1, 1)
    prior_rank = robust01(prior).reshape(-1, 1)
    aug = np.concatenate([test_z.astype(np.float32), energy_rank.astype(np.float32), prior_rank.astype(np.float32)], axis=1)
    diff = np.diff(aug, axis=0, prepend=aug[:1])
    diff_scale = (diff.std(axis=0) + 1e-6).astype(np.float32)
    return aug, diff_scale


def choose_positive_centers(mask: np.ndarray, score: np.ndarray, max_pos: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    for s, e in segments(mask):
        if e < s:
            continue
        local = np.arange(s, e + 1)
        out.append(int(local[np.argmax(score[local])]))
    out = np.array(out, dtype=np.int64)
    if len(out) > max_pos:
        weights = score[out].astype(np.float64) + 1e-6
        weights = weights / weights.sum()
        out = rng.choice(out, size=max_pos, replace=False, p=weights)
    return np.unique(out)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool).copy()
    out = np.zeros_like(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    for i in idx:
        out[max(0, i - radius) : min(len(mask), i + radius + 1)] = True
    return out


def choose_negative_centers(
    temporal: np.ndarray,
    prior: np.ndarray,
    n_neg: int,
    window: int,
    rng: np.random.Generator,
    boundary_radius: int,
) -> np.ndarray:
    n = len(temporal)
    half = window // 2
    valid = np.zeros(n, dtype=bool)
    valid[half : n - half] = True
    far = valid & ~dilate_mask(temporal, boundary_radius)
    low_thr = np.quantile(prior[far] if far.any() else prior[valid], 0.60)
    easy_pool = np.flatnonzero(far & (prior <= low_thr))

    boundary = np.zeros(n, dtype=bool)
    for s, e in segments(temporal):
        boundary[max(half, s - boundary_radius) : max(half, s)] = True
        boundary[min(n - half, e + 1) : min(n - half, e + 1 + boundary_radius)] = True
    hard_pool = np.flatnonzero(boundary & valid & ~temporal)

    pools = []
    if len(easy_pool):
        pools.append(rng.choice(easy_pool, size=max(1, n_neg // 2), replace=len(easy_pool) < max(1, n_neg // 2)))
    if len(hard_pool):
        pools.append(rng.choice(hard_pool, size=max(1, n_neg - sum(len(p) for p in pools)), replace=len(hard_pool) < max(1, n_neg - sum(len(p) for p in pools))))
    if not pools:
        fallback = np.flatnonzero(valid & ~temporal)
        pools.append(rng.choice(fallback, size=n_neg, replace=len(fallback) < n_neg))
    neg = np.concatenate(pools)
    if len(neg) > n_neg:
        neg = rng.choice(neg, size=n_neg, replace=False)
    elif len(neg) < n_neg:
        fallback = np.flatnonzero(valid & ~temporal)
        neg = np.concatenate([neg, rng.choice(fallback, size=n_neg - len(neg), replace=len(fallback) < n_neg - len(neg))])
    return neg.astype(np.int64)


def train_self_ranker(
    aug: np.ndarray,
    diff_scale: np.ndarray,
    temporal: np.ndarray,
    prior: np.ndarray,
    window: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden: int,
    max_pos: int,
    neg_multiplier: float,
    boundary_radius: int,
    feature_mode: str,
) -> tuple[CenterEventRanker, dict[str, int]]:
    rng = np.random.default_rng(seed)
    pos = choose_positive_centers(temporal, prior, max_pos=max_pos, rng=rng)
    if len(pos) < 4:
        # Degenerate detector: fall back to top temporal points, or top prior if
        # temporal is empty.  This still does not use labels.
        source = np.flatnonzero(temporal)
        if len(source) == 0:
            source = np.argsort(prior)[::-1][: max(8, max_pos)]
        pos = source[np.argsort(prior[source])[::-1][: max(4, min(max_pos, len(source)))]]
    n_neg = max(8, int(np.ceil(len(pos) * neg_multiplier)))
    neg = choose_negative_centers(temporal, prior, n_neg=n_neg, window=window, rng=rng, boundary_radius=boundary_radius)

    x_pos = windows_to_features(make_value_windows(aug, pos, window), diff_scale, feature_mode)
    x_neg = windows_to_features(make_value_windows(aug, neg, window), diff_scale, feature_mode)
    x = np.concatenate([x_pos, x_neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(x_pos), dtype=np.float32), np.zeros(len(x_neg), dtype=np.float32)])
    order = rng.permutation(len(y))
    x = x[order]
    y = y[order]

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = CenterEventRanker(x.shape[2], hidden=hidden).to(device)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    pos_weight = torch.tensor([max(1.0, len(x_neg) / max(len(x_pos), 1))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model, {"pseudo_pos": int(len(x_pos)), "pseudo_neg": int(len(x_neg))}


@torch.no_grad()
def score_model(
    model: CenterEventRanker,
    aug: np.ndarray,
    diff_scale: np.ndarray,
    candidates: np.ndarray,
    window: int,
    feature_mode: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(candidates) == 0:
        return np.array([], dtype=np.float64)
    x = windows_to_features(make_value_windows(aug, candidates, window), diff_scale, feature_mode)
    model.eval()
    outs = []
    for s in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[s : s + batch_size]).to(device)
        outs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


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
    parser.add_argument("--pairs", default="MSL:AnomalyTransformer,SMD:Autoformer,SWaT:AnomalyTransformer")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--window", type=int, default=65)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--score_batch_size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--max_pos", type=int, default=5000)
    parser.add_argument("--neg_multiplier", type=float, default=2.0)
    parser.add_argument("--boundary_radius", type=int, default=96)
    parser.add_argument("--feature_mode", choices=["value", "diff", "geometry"], default="geometry")
    args = parser.parse_args()

    out_dir = args.output_dir or args.exp_dir / "self_trained_event_ranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = out_dir / "candidate_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    dtpp = load_module(Path("scripts/run_dtpp_from_at_scores.py"), "dtpp_self_event_ranker")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = parse_pairs(args.pairs)
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
        _train_z, test_z_full, labels = data_cache[dataset]
        test_z = test_z_full[: len(gt)]
        if not np.array_equal(labels[: len(gt)].astype(bool), gt):
            raise ValueError(f"label mismatch for {dataset}/{model_name}")

        temporal = z["temporal_pred"].astype(bool).reshape(-1)
        baseline = z["baseline_pred"].astype(bool).reshape(-1)
        current_ner = z["final_pred"].astype(bool).reshape(-1)
        prior = z["event_prior"].astype(np.float64).reshape(-1)
        saved_neural = z["neural_rescue_score"].astype(np.float64).reshape(-1)
        train_energy = z["train_energy"].astype(np.float64).reshape(-1)
        test_energy = z["test_energy"].astype(np.float64).reshape(-1)
        candidates = z["candidate_indices"].astype(np.int64).reshape(-1)
        candidates = candidates[(candidates >= 0) & (candidates < len(gt))]
        candidates = candidates[~temporal[candidates]]
        candidates = np.unique(candidates)
        selected = z["selected_peaks"].astype(np.int64).reshape(-1) if "selected_peaks" in z.files else np.array([], dtype=np.int64)
        budget = len(selected) if len(selected) else max(2, int(np.ceil(2e-5 * len(gt))))

        aug, diff_scale = build_augmented_series(test_z, train_energy, test_energy[: len(gt)], prior)
        print(f"training self-ranker for {dataset}/{model_name} on {device}; candidates={len(candidates)} budget={budget}", flush=True)
        model, pseudo = train_self_ranker(
            aug=aug,
            diff_scale=diff_scale,
            temporal=temporal,
            prior=prior,
            window=args.window,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden=args.hidden,
            max_pos=args.max_pos,
            neg_multiplier=args.neg_multiplier,
            boundary_radius=args.boundary_radius,
            feature_mode=args.feature_mode,
        )
        net_score = score_model(
            model,
            aug,
            diff_scale,
            candidates,
            window=args.window,
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
            self_event_score=net_score.astype(np.float32),
            saved_neural=saved_c.astype(np.float32),
            budget=np.array([budget], dtype=np.int32),
        )

        variants: dict[str, np.ndarray] = {
            "Baseline": baseline,
            "+ Temporal calibration": temporal,
            "Prior + NMS": select_from_candidate_score(temporal, candidates, prior_c, budget),
            "Current NER": current_ner,
            "SelfNet only": select_from_candidate_score(temporal, candidates, net_score, budget),
        }
        if len(candidates):
            for lam in lambdas:
                variants[f"RankSum prior+{lam:g}SelfNet"] = select_from_candidate_score(
                    temporal, candidates, prior_r + lam * net_r, budget
                )
            for alpha in product_alphas:
                variants[f"Prior*SelfNet^{alpha:g}"] = select_from_candidate_score(
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
                "epochs": args.epochs,
                "feature_mode": args.feature_mode,
                **pseudo,
            }
            row.update(evaluate(pred, gt))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "self_trained_event_ranker_pair_metrics.csv", index=False)
    summary = df.groupby("variant", as_index=False)[["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]].mean(numeric_only=True)
    for c in ["pa_f1", "point_f1", "event_f1", "range_f1", "fpr"]:
        summary[c] *= 100.0
    summary = summary.sort_values(["event_f1", "range_f1", "pa_f1"], ascending=False)
    summary.to_csv(out_dir / "self_trained_event_ranker_summary.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
