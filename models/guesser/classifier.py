"""Geocell guesser implementing core.interfaces.Guesser.

PLACEHOLDER (M0): the classifier is nearest-cell-mean, p(k | v) = softmax(T * cos(v, mu_k)).
M1/M2 replaces ``fit``/``_per_view_probs`` with a trained classifier head; the multi-view
combination and expected-score guess stay as they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.guesser.combine import combine_views, expected_score_guess, expected_score_matrix
from models.guesser.geocells import fit_geocells, latlon_to_xyz, xyz_to_latlon


def _candidates(centroids: np.ndarray) -> np.ndarray:
    """Cell centroids plus great-circle midpoints of every pair, so a split belief can guess
    "in between" (PLAN.md 7.2 step 4)."""
    xyz = latlon_to_xyz(centroids[:, 0], centroids[:, 1])
    i, j = np.triu_indices(len(xyz), k=1)
    mids = xyz[i] + xyz[j]
    ok = np.linalg.norm(mids, axis=1) > 1e-6  # antipodal pairs have no midpoint
    return np.concatenate([centroids, xyz_to_latlon(mids[ok])]) if ok.any() else centroids


class CellGuesser:
    def __init__(
        self,
        centroids: np.ndarray,
        cell_means: np.ndarray,
        prior: np.ndarray,
        temperature: float = 20.0,
    ):
        self.centroids = np.asarray(centroids, np.float64)  # (K, 2)
        self.cell_means = np.asarray(cell_means, np.float32)  # (K, D), L2-normalised
        self.prior = np.asarray(prior, np.float64)  # (K,)
        self.temperature = float(temperature)
        self.candidates = _candidates(self.centroids)
        self._s = expected_score_matrix(self.candidates, self.centroids)

    @property
    def n_cells(self) -> int:
        return len(self.centroids)

    # --- training -------------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        view_embs: np.ndarray,
        view_lat: np.ndarray,
        view_lon: np.ndarray,
        *,
        n_cells: int,
        temperature: float = 20.0,
        iters: int = 50,
        seed: int = 0,
    ):
        centroids, labels = fit_geocells(view_lat, view_lon, n_cells, iters, seed)
        k = len(centroids)
        e = view_embs.astype(np.float32)
        means = np.stack(
            [
                e[labels == j].mean(0) if np.any(labels == j) else np.zeros(e.shape[1])
                for j in range(k)
            ]
        )
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-8)
        prior = (np.bincount(labels, minlength=k) + 1.0) / (len(labels) + k)
        return cls(centroids, means, prior, temperature)

    # --- Guesser protocol -----------------------------------------------------------------
    def _per_view_probs(self, view_embs: np.ndarray) -> np.ndarray:
        logits = self.temperature * (
            np.atleast_2d(view_embs).astype(np.float32) @ self.cell_means.T
        )
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def cell_probs(self, view_embs: np.ndarray) -> np.ndarray:
        return combine_views(self._per_view_probs(view_embs), self.prior)

    def guess(self, probs: np.ndarray) -> tuple[float, float]:
        return expected_score_guess(probs, self.candidates, self._s)[0]

    # --- persistence (weights live outside git: ground rule 3) ----------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            centroids=self.centroids,
            cell_means=self.cell_means,
            prior=self.prior,
            temperature=self.temperature,
        )

    @classmethod
    def load(cls, path: str | Path) -> CellGuesser:
        z = np.load(path)
        return cls(z["centroids"], z["cell_means"], z["prior"], float(z["temperature"]))
