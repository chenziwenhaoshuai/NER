from __future__ import annotations

import numpy as np


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values)) if len(values) > 1 else 1.0
    return ranks


def route_score(
    prior_rank: np.ndarray,
    self_rank: np.ndarray,
    geometry_rank: np.ndarray,
    augmented_rank: np.ndarray,
    density: float,
    budget: int,
    candidate_count: int,
    low_density: float = 2.5,
    high_density: float = 10.0,
    compact_budget: int = 9,
    compact_candidates: int = 50,
) -> tuple[np.ndarray, str]:
    if density <= low_density or (
        budget <= compact_budget and candidate_count <= compact_candidates
    ):
        return geometry_rank, "geometry_ae_compact"
    if density >= high_density:
        return augmented_rank, "augmented_ae_dense"
    score = (
        0.25 * prior_rank
        + 0.5 * self_rank
        + 2.0 * geometry_rank
        + augmented_rank
    )
    return score, "mixed_neural_dominant"


def select_from_candidate_score(
    temporal: np.ndarray,
    candidates: np.ndarray,
    candidate_score: np.ndarray,
    budget: int,
) -> np.ndarray:
    prediction = temporal.astype(bool).copy()
    if len(candidates) and budget > 0:
        order = np.argsort(candidate_score)[::-1]
        prediction[candidates[order[:budget]]] = True
    return prediction

