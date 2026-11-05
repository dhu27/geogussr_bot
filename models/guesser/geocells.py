"""Geocells: cluster training locations into K cells. PLACEHOLDER (M0): spherical k-means.

M1/M2 may swap in admin-boundary or semantic geocells (as in PIGEON); the output stays the
same: (K, 2) centroids in degrees plus a cell label per training image.
"""

from __future__ import annotations

import numpy as np


def latlon_to_xyz(lat, lon) -> np.ndarray:
    p, lam = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.stack([np.cos(p) * np.cos(lam), np.cos(p) * np.sin(lam), np.sin(p)], axis=-1)


def xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return np.stack([lat, lon], axis=-1)


def fit_geocells(
    lat: np.ndarray, lon: np.ndarray, k: int, iters: int = 50, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means. Returns (centroids (K, 2) lat/lon degrees, labels (n,))."""
    x = latlon_to_xyz(lat, lon)
    k = min(k, len(np.unique(x, axis=0)))
    rng = np.random.default_rng(seed)
    # k-means++ init
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(1 - x @ np.stack(centers).T, axis=1).clip(0)
        centers.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.stack(centers)
    for _ in range(iters):
        labels = np.argmax(x @ c.T, axis=1)
        new = np.stack([x[labels == j].sum(0) if np.any(labels == j) else c[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        if np.allclose(new, c):
            break
        c = new
    labels = np.argmax(x @ c.T, axis=1)
    return xyz_to_latlon(c), labels
