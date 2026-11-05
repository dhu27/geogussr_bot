"""Distance and score (PLAN.md 6.5). Vectorised: scalars or numpy arrays both work."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
MAX_SCORE = 5000.0
SCORE_SCALE_KM = 1492.7  # community approximation of GeoGuessr's world map


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees."""
    p1, l1, p2, l2 = (np.deg2rad(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(d) if np.ndim(d) == 0 else d


def score(distance_km):
    """GeoGuessr-style score in [0, 5000]: S(d) = 5000 * exp(-d / 1492.7)."""
    s = MAX_SCORE * np.exp(-np.asarray(distance_km, dtype=np.float64) / SCORE_SCALE_KM)
    return float(s) if np.ndim(s) == 0 else s
