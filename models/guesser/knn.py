"""Nearest-neighbour retrieval over geocells. Owner: Daniel Hu.

No training loop: the model stores every TRAIN view embedding with its geocell. For a view, take its
``k`` most similar training views (cosine similarity), weight each by exp(similarity / temperature),
and add the weights up per cell. Mixing in a little of the cell prior (``smoothing``) keeps every
cell above zero, which multi-view combining needs (it multiplies per-view probabilities).

Same geocells as the baseline (spherical k-means, same seed), so the comparison isolates the head.
The index is ~207,000 x 768; it is saved as float16 (~320 MB) and held as float32 in memory.
"""

from __future__ import annotations

import numpy as np

from models.guesser.classifier import _GeocellGuesser, _prior
from models.guesser.geocells import fit_geocells
from models.guesser.registry import register

K_MAX = 1000  # neighbours kept per view in the cache, so k can be tuned without re-searching
CHUNK = 2048  # views searched per matrix multiply
CACHE_LIMIT = 200_000


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@register("knn")
class KNNGuesser(_GeocellGuesser):
    def __init__(self, centroids, prior, index, labels, k=50, temperature=0.05, smoothing=0.2):
        super().__init__(centroids, prior)
        self.index = _unit(index)  # (N, D) training view embeddings, unit length
        self.labels = np.asarray(labels, np.int64)  # (N,) their geocells
        self.k, self.temperature, self.smoothing = int(k), float(temperature), float(smoothing)
        self._cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def _neighbours(self, x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each (unit) view: indices and similarities of its K_MAX nearest training views,
        most similar first. Cached per view, since evaluation sees each view twice."""
        keys = [row.tobytes() for row in x]
        todo = [i for i, key in enumerate(keys) if key not in self._cache]
        if len(self._cache) + len(todo) > CACHE_LIMIT:
            self._cache.clear()
            todo = list(range(len(x)))
        kk = min(max(K_MAX, self.k), len(self.index))
        for s in range(0, len(todo), CHUNK):
            rows = todo[s : s + CHUNK]
            sims = x[rows] @ self.index.T  # (m, N)
            top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            top_sims = np.take_along_axis(sims, top, axis=1)
            order = np.argsort(-top_sims, axis=1)
            top, top_sims = (np.take_along_axis(a, order, axis=1) for a in (top, top_sims))
            for j, i in enumerate(rows):
                self._cache[keys[i]] = (top[j], top_sims[j])
        return [self._cache[key] for key in keys]

    def warm(self, view_embs: np.ndarray) -> None:
        """Search many views in one batch (much faster than one at a time); used for tuning."""
        self._neighbours(_unit(view_embs))

    def _per_view_probs(self, view_embs):
        """(t, 768) view embeddings -> (t, K) cell probabilities, each row summing to 1."""
        out = np.empty((len(np.atleast_2d(view_embs)), self.n_cells))
        for row, (idx, sims) in zip(out, self._neighbours(_unit(view_embs)), strict=True):
            idx, sims = idx[: self.k], sims[: self.k]
            w = np.exp((sims - sims[0]) / self.temperature)
            hist = np.bincount(self.labels[idx], weights=w, minlength=self.n_cells)
            row[:] = (1 - self.smoothing) * hist / hist.sum() + self.smoothing * self.prior
        return out

    @classmethod
    def fit(cls, x, lat, lon, *, n_cells=256, k=50, temperature=0.05, smoothing=0.2):
        """x: (n_views, 768) TRAIN-split view embeddings; lat, lon: (n_views,) their locations."""
        centroids, labels = fit_geocells(lat, lon, n_cells, 50, 0)  # same cells as the baseline
        return cls(centroids, _prior(labels, len(centroids)), x, labels, k, temperature, smoothing)

    def _arrays(self):
        return {
            "index": self.index.astype(np.float16),
            "labels": self.labels.astype(np.int16),
            "params": np.array([self.k, self.temperature, self.smoothing], np.float64),
        }

    @classmethod
    def _from_arrays(cls, a):
        k, temperature, smoothing = a["params"]
        return cls(a["centroids"], a["prior"], a["index"], a["labels"], k, temperature, smoothing)
