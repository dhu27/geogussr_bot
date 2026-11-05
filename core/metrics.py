"""Reported metrics (PLAN.md 6.5): median error, mean score, share within thresholds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.scoring import score

THRESHOLDS_KM: tuple[float, ...] = (1, 25, 200, 750, 2500)


def summarize(errors_km: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Metrics for one set of guesses. Keys: n, median_km, mean_score, within_<t>km."""
    e = np.asarray(errors_km, dtype=np.float64)
    if e.size == 0:
        return {"n": 0}
    out: dict[str, float] = {
        "n": int(e.size),
        "median_km": float(np.median(e)),
        "mean_score": float(np.mean(score(e))),
    }
    for t in THRESHOLDS_KM:
        out[f"within_{t:g}km"] = float(np.mean(e <= t))
    return out


def summarize_by_group(
    errors_km: Sequence[float] | np.ndarray, groups: Sequence[str]
) -> dict[str, dict[str, float]]:
    """``{"overall": ..., <group>: ...}``, e.g. grouped by continent."""
    e = np.asarray(errors_km, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    if e.shape != g.shape:
        raise ValueError("errors_km and groups must have the same length")
    out = {"overall": summarize(e)}
    for name in sorted(set(g.tolist())):
        out[str(name)] = summarize(e[g == name])
    return out
