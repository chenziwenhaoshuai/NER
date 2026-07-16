"""Counterfactual-temperature MoE for Neural Event Rescue.

This variant keeps the clean MoE interpretation while avoiding hand-written
route/fusion rules:

1. Learn a dataset-level expert prior from label-free counterfactual
   event-over-normal ranking.
2. Calibrate the prior temperature with counterfactual diagnostics only.  The
   default selector minimizes counterfactual ranking loss, but rejects
   temperatures that collapse synthetic event retrieval far below the best
   attainable AP.  This prevents over-sharpening when several experts provide
   complementary evidence.
3. Rank deployment candidates by the calibrated expert mixture.

The script does not use anomaly labels, v7 routes, v7 fused scores,
candidate-density rules, or dataset identity as a gate feature. Labels are
loaded only after predictions are materialized, for metric reporting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Keep --help usable in the lightweight runtime.
    torch = None

    class _MissingNN:
        class Module:
            pass

    nn = _MissingNN()


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from ner.metrics import evaluate  # noqa: E402
from ner.router import rank01, select_from_candidate_score  # noqa: E402


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]


EXPERTS = ["self", "geometry", "augmented"]
SCORE_KEYS = ["self_event_score", "geometry_ae_score", "augmented_ae_score"]
METRICS = [
    "pa_precision",
    "pa_recall",
    "pa_f1",
    "point_f1",
    "event_f1",
    "range_f1",
    "strict_event_f1",
    "false_events_per_100k",
    "fpr",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_probe(root: Path, dataset: str, model: str) -> Path:
    matches = list(root.glob(f"gpu*/scores/{dataset}_{model}_counterfactual_scores.npz"))
    if len(matches) != 1:
        raise FileNotFoundError((root, dataset, model, matches))
    return matches[0]


def rank_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    saved = np.load(path)
    normal = np.stack([saved[f"{expert}_normal"] for expert in EXPERTS], axis=1).astype(float)
    event = np.stack([saved[f"{expert}_event"] for expert in EXPERTS], axis=1).astype(float)
    combined = np.concatenate([normal, event], axis=0)
    ranks = np.stack([rank01(combined[:, i]) for i in range(3)], axis=1).astype(np.float32)
    return ranks[: len(normal)], ranks[len(normal) :]


class PriorGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3))
        self.raw_scale = nn.Parameter(torch.tensor(1.0))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def margin(self, delta: torch.Tensor) -> torch.Tensor:
        scale = torch.nn.functional.softplus(self.raw_scale) + 1e-3
        return scale * (delta * self.weights().unsqueeze(0)).sum(dim=1)


def train_prior(
    deltas: list[np.ndarray],
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    margin: float,
    robust_temperature: float,
    entropy_weight: float,
) -> tuple[PriorGate, list[dict[str, float]]]:
    set_seed(seed)
    model = PriorGate().to(device)
    tensors = [torch.from_numpy(delta.astype(np.float32)).to(device) for delta in deltas]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = torch.stack(
            [
                torch.nn.functional.softplus(margin - model.margin(delta)).mean()
                for delta in tensors
            ]
        )
        robust = robust_temperature * (
            torch.logsumexp(losses / robust_temperature, dim=0) - np.log(len(tensors))
        )
        weights = model.weights()
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum()
        loss = robust + entropy_weight * entropy
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "robust_loss": float(robust.detach()),
                "worst_loss": float(losses.max().detach()),
                "weight_self": float(weights[0].detach()),
                "weight_geometry": float(weights[1].detach()),
                "weight_augmented": float(weights[2].detach()),
            }
        )
    return model, history


def temperature_weights(base_weight: np.ndarray, gamma: float) -> np.ndarray:
    w = np.power(np.clip(base_weight, 1e-12, None), gamma)
    return w / w.sum()


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = labels[order].astype(bool)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    precision_at_hit = np.cumsum(y)[y] / (np.flatnonzero(y) + 1)
    return float(precision_at_hit.sum() / positives)


def topk_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    k = positives
    pred = np.zeros_like(labels, dtype=bool)
    pred[np.argsort(scores)[::-1][:k]] = True
    tp = int(np.logical_and(pred, labels).sum())
    precision = tp / k if k else 0.0
    recall = tp / positives if positives else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def temperature_curve(
    paired_ranks: list[tuple[np.ndarray, np.ndarray]],
    base_weight: np.ndarray,
    gammas: list[float],
    margin: float,
) -> pd.DataFrame:
    rows = []
    for gamma in gammas:
        weight = temperature_weights(base_weight, gamma)
        margins = []
        aps = []
        f1s = []
        accs = []
        for normal, event in paired_ranks:
            delta = event - normal
            margin_values = delta @ weight
            margins.append(margin_values)
            combined_scores = np.concatenate([normal @ weight, event @ weight])
            labels = np.concatenate([np.zeros(len(normal), dtype=bool), np.ones(len(event), dtype=bool)])
            aps.append(average_precision(combined_scores, labels))
            f1s.append(topk_f1(combined_scores, labels))
            accs.append(float((margin_values > 0).mean()))
        all_margins = np.concatenate(margins)
        rows.append(
            {
                "gamma": gamma,
                "loss": float(np.logaddexp(0.0, margin - all_margins).mean()),
                "accuracy": float(np.mean(accs)),
                "margin_mean": float(all_margins.mean()),
                "margin_p10": float(np.quantile(all_margins, 0.10)),
                "average_precision": float(np.mean(aps)),
                "topk_f1": float(np.mean(f1s)),
                "weight_self": float(weight[0]),
                "weight_geometry": float(weight[1]),
                "weight_augmented": float(weight[2]),
            }
        )
    return pd.DataFrame(rows)


def select_temperature(
    curve: pd.DataFrame,
    ap_drop_tolerance: float,
    loss_tolerance: float,
    prefer_sharp_ap: float,
    sharp_gamma: float,
    selector_mode: str,
) -> tuple[float, str]:
    best_loss = float(curve["loss"].min())
    best_ap = float(curve["average_precision"].max())
    if selector_mode == "ap_then_loss":
        # First keep temperatures on the synthetic-retrieval AP plateau, then
        # choose the sharpest temperature still on the counterfactual-loss
        # plateau.  Low-AP settings use the same ranking-loss plateau with a
        # global minimum sharpness, because AP is not discriminative there.
        if best_ap < prefer_sharp_ap:
            eligible = curve[
                (curve["loss"] <= best_loss + loss_tolerance)
                & (curve["gamma"] >= sharp_gamma)
            ]
            if eligible.empty:
                eligible = curve[curve["loss"] <= best_loss + loss_tolerance]
            return float(eligible["gamma"].min()), "low_ap_loss_plateau"
        ap_eligible = curve[curve["average_precision"] >= best_ap - ap_drop_tolerance]
        best_ap_loss = float(ap_eligible["loss"].min())
        eligible = ap_eligible[ap_eligible["loss"] <= best_ap_loss + loss_tolerance]
        return float(eligible["gamma"].max()), "ap_then_loss_plateau"
    if selector_mode != "loss_then_ap":
        raise ValueError(f"unknown selector_mode={selector_mode}")
    # If synthetic retrieval is weak, use the robust ranking loss: the AP
    # signal is too noisy to justify softening.
    if best_ap < prefer_sharp_ap:
        eligible = curve[curve["loss"] <= best_loss + loss_tolerance]
        chosen = float(eligible["gamma"].max())
        return chosen, "loss_plateau_sharp_low_ap"
    # Otherwise preserve complementary experts by selecting the smallest
    # temperature that is near-optimal in loss and does not collapse AP.
    eligible = curve[
        (curve["loss"] <= best_loss + loss_tolerance)
        & (curve["average_precision"] >= best_ap - ap_drop_tolerance)
    ]
    if eligible.empty:
        eligible = curve[curve["average_precision"] >= best_ap - ap_drop_tolerance]
        reason = "ap_guard"
    else:
        reason = "loss_ap_guard"
    chosen = float(eligible["gamma"].min())
    if chosen > sharp_gamma and best_ap >= prefer_sharp_ap:
        # Prevent pathological over-sharpening when AP remains high but the
        # loss is monotonically margin-seeking; this is still a single global
        # operating prior, not a dataset-specific rule.
        softer = eligible[eligible["gamma"] <= sharp_gamma]
        if not softer.empty:
            chosen = float(softer["gamma"].max())
            reason += "_capped"
    return chosen, reason


def resolve_storage(profile: Path) -> Path:
    """Return a directory with candidate_scores/ and predictions/.

    The full internal experiment tree stores the frozen router under
    ``neural_router_v7_neural_dominant``.  The public artifact package stores
    the same schema directly under ``artifacts/v35``.  Supporting both keeps the
    experiment script useful for exact paper reproduction and for external
    users who only download the compact release artifacts.
    """

    direct_ok = (profile / "candidate_scores").is_dir() and (profile / "predictions").is_dir()
    if direct_ok:
        return profile
    nested = profile / "neural_router_v7_neural_dominant"
    nested_ok = (nested / "candidate_scores").is_dir() and (nested / "predictions").is_dir()
    if nested_ok:
        return nested
    raise FileNotFoundError(
        "profile_dir must contain candidate_scores/ and predictions/, either "
        "directly or under neural_router_v7_neural_dominant/."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v35",
        help=(
            "Frozen prediction/candidate-score root. Accepts either the public "
            "artifacts/v35 schema or the full internal paper-profile directory."
        ),
    )
    parser.add_argument("--strong_probe_dir", type=Path, required=True)
    parser.add_argument("--weak_probe_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--robust_temperature", type=float, default=0.03)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--ap_drop_tolerance", type=float, default=0.0015)
    parser.add_argument("--loss_tolerance", type=float, default=0.002)
    parser.add_argument("--prefer_sharp_ap", type=float, default=0.98)
    parser.add_argument("--sharp_gamma", type=float, default=4.0)
    parser.add_argument(
        "--selector_mode",
        choices=["loss_then_ap", "ap_then_loss"],
        default="ap_then_loss",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for the counterfactual-temperature MoE experiment. "
            "Install it with `pip install -r requirements-train.txt`."
        )

    profile = args.profile_dir.resolve()
    storage = resolve_storage(profile)
    roots = [args.strong_probe_dir.resolve(), args.weak_probe_dir.resolve()]
    output = args.output_dir.resolve()
    for name in ["checkpoints", "candidate_scores", "predictions", "summary", "training"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    prior_weights: dict[str, np.ndarray] = {}
    chosen_temperatures: dict[str, float] = {}
    temp_rows = []

    for dataset in DATASETS:
        paired = [rank_pair(find_probe(root, dataset, model)) for root in roots for model in MODELS]
        deltas = [event - normal for normal, event in paired]
        prior, history = train_prior(
            deltas,
            device,
            args.seed,
            args.epochs,
            args.lr,
            args.margin,
            args.robust_temperature,
            args.entropy_weight,
        )
        base_weight = prior.weights().detach().cpu().numpy()
        prior_weights[dataset] = base_weight
        curve = temperature_curve(paired, base_weight, args.gammas, args.margin)
        gamma, reason = select_temperature(
            curve,
            args.ap_drop_tolerance,
            args.loss_tolerance,
            args.prefer_sharp_ap,
            args.sharp_gamma,
            args.selector_mode,
        )
        chosen_temperatures[dataset] = gamma
        curve.insert(0, "dataset", dataset)
        curve["selected"] = curve["gamma"].eq(gamma)
        curve.to_csv(output / "training" / f"{dataset}_temperature_curve.csv", index=False)
        pd.DataFrame(history).to_csv(output / "training" / f"{dataset}_prior_training.csv", index=False)
        temp_rows.append(
            {
                "dataset": dataset,
                "selected_gamma": gamma,
                "selection_reason": reason,
                "weight_self": base_weight[0],
                "weight_geometry": base_weight[1],
                "weight_augmented": base_weight[2],
                "best_loss": curve["loss"].min(),
                "best_ap": curve["average_precision"].max(),
            }
        )
        torch.save(
            {"state_dict": prior.state_dict(), "experts": EXPERTS, "selected_gamma": gamma},
            output / "checkpoints" / f"{dataset}_prior_gate.pt",
        )

    rows = []
    gate_rows = []
    for dataset in DATASETS:
        gamma = chosen_temperatures[dataset]
        weight = temperature_weights(prior_weights[dataset], gamma)
        for model in MODELS:
            score_file = np.load(storage / "candidate_scores" / f"{dataset}_{model}_candidate_scores.npz")
            pred_file = np.load(storage / "predictions" / f"{dataset}_{model}_seed2021_predictions.npz")
            raw = np.stack([score_file[key] for key in SCORE_KEYS], axis=1).astype(float)
            ranks = np.stack([rank01(raw[:, i]) for i in range(3)], axis=1)
            candidates = score_file["candidates"].astype(np.int64)
            temporal = pred_file["temporal_pred"].astype(bool)
            budget = int(score_file["budget"][0])
            score = ranks @ weight
            prediction = select_from_candidate_score(temporal, candidates, score, budget)
            np.savez_compressed(
                output / "candidate_scores" / f"{dataset}_{model}_candidate_scores.npz",
                candidates=candidates.astype(np.int32),
                expert_rank=ranks.astype(np.float32),
                moe_score=score.astype(np.float32),
                expert_weights=np.tile(weight, (len(candidates), 1)).astype(np.float32),
                selected_gamma=np.asarray([gamma], dtype=np.float32),
                budget=np.asarray([budget], dtype=np.int32),
            )
            np.savez_compressed(
                output / "predictions" / f"{dataset}_{model}_seed2021_predictions.npz",
                temporal_pred=temporal.astype(np.int8),
                candidate_indices=candidates.astype(np.int32),
                final_pred=prediction.astype(np.int8),
            )
            row = {"dataset": dataset, "model": model, "variant": "Counterfactual-temperature MoE"}
            row.update(evaluate(prediction, pred_file["gt"].astype(bool)))
            rows.append(row)
            gate_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "selected_gamma": gamma,
                    "weight_self": weight[0],
                    "weight_geometry": weight[1],
                    "weight_augmented": weight[2],
                }
            )

    pair = pd.DataFrame(rows)
    pair.to_csv(output / "summary" / "pair_metrics.csv", index=False)
    overall = pair.groupby("variant", as_index=False)[METRICS].mean()
    for metric in METRICS:
        overall[metric] *= 100.0
    overall.to_csv(output / "summary" / "overall_metrics.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output / "summary" / "gate_weights.csv", index=False)
    pd.DataFrame(temp_rows).to_csv(output / "summary" / "temperature_selection.csv", index=False)
    (output / "config.json").write_text(
        json.dumps(
            {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "device_resolved": str(device),
                "forbidden_inputs": [
                    "ground-truth anomaly labels",
                    "v7 route",
                    "v7 fused score",
                    "candidate density",
                    "dataset identity as a gate feature",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))
    print(pd.DataFrame(temp_rows).to_string(index=False))


if __name__ == "__main__":
    main()

