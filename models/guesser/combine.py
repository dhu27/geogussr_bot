"""Multi-view combining and the expected-score guess (PLAN.md 7.2, steps 3-4)."""

from __future__ import annotations

import numpy as np

from core.scoring import haversine_km, score

_EPS = 1e-12


def combine_views(per_view_probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Independent-evidence combination:
    log p(k | v_1:t) = sum_i log p(k | v_i) - (t - 1) log p(k) + const.

    Args:
        per_view_probs: (t, K) per-view cell probabilities.
        prior: (K,) training-set cell prior p(k).
    Returns:
        (K,) normalised probabilities.
    """
    p = np.atleast_2d(per_view_probs)
    t = p.shape[0]
    logp = np.log(p + _EPS).sum(0) - (t - 1) * np.log(prior + _EPS)
    logp -= logp.max()
    out = np.exp(logp)
    return out / out.sum()


def expected_score_matrix(candidates: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(A, K) matrix S(d(a, c_k)) for candidate guesses a and cell centroids c_k (lat/lon deg)."""
    d = haversine_km(
        candidates[:, None, 0], candidates[:, None, 1], centroids[None, :, 0], centroids[None, :, 1]
    )
    return score(d)


def expected_score_guess(probs: np.ndarray, candidates: np.ndarray, s_matrix: np.ndarray):
    """argmax_a sum_k p_k S(d(a, c_k)). Returns ((lat, lon), expected score)."""
    ev = s_matrix @ probs
    i = int(np.argmax(ev))
    return (float(candidates[i, 0]), float(candidates[i, 1])), float(ev[i])
