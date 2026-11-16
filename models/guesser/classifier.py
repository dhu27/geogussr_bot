"""Geocell guessers implementing core.interfaces.Guesser.

Two heads share the geocells, the multi-view combination and the expected-score guess
(PLAN.md 7.2):

* ``CellGuesser`` (M1): a softmax classifier trained on frozen StreetCLIP view embeddings,
  p(k | v) = softmax(W^T standardise(v) + b). Trained in numpy (full-batch Adam with L2), so
  training needs no torch and inference runs anywhere (CI, the harness).
* ``NearestMeanGuesser`` (the M0 placeholder, kept as a baseline):
  p(k | v) = softmax(T * cos(v, mean embedding of cell k)).

``CellGuesser.load(path)`` returns whichever kind was saved. Weights live under runs/, never in
git (ground rule 3).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from models.guesser.combine import combine_views, expected_score_guess, expected_score_matrix
from models.guesser.geocells import fit_geocells, latlon_to_xyz, xyz_to_latlon


def _candidates(centroids: np.ndarray) -> np.ndarray:
    """Cell centroids plus great-circle midpoints of every pair, so a belief spread over several
    cells can guess "in between" (PLAN.md 7.2 step 4)."""
    xyz = latlon_to_xyz(centroids[:, 0], centroids[:, 1])
    i, j = np.triu_indices(len(xyz), k=1)
    mids = xyz[i] + xyz[j]
    ok = np.linalg.norm(mids, axis=1) > 1e-6  # antipodal pairs have no midpoint
    return np.concatenate([centroids, xyz_to_latlon(mids[ok])]) if ok.any() else centroids


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def _prior(labels: np.ndarray, k: int) -> np.ndarray:
    return (np.bincount(labels, minlength=k) + 1.0) / (len(labels) + k)  # add-one smoothing


class _GeocellGuesser:
    """Shared machinery. Subclasses implement ``_per_view_probs`` and their arrays for save."""

    kind = ""

    def __init__(self, centroids: np.ndarray, prior: np.ndarray):
        self.centroids = np.asarray(centroids, np.float64)  # (K, 2) lat, lon
        self.prior = np.asarray(prior, np.float64)  # (K,) training cell frequencies
        self.candidates = _candidates(self.centroids)
        self._s = expected_score_matrix(self.candidates, self.centroids)

    @property
    def n_cells(self) -> int:
        return len(self.centroids)

    def _per_view_probs(self, view_embs: np.ndarray) -> np.ndarray:  # (t, D) -> (t, K)
        raise NotImplementedError

    # --- Guesser protocol -----------------------------------------------------------------
    def cell_probs(self, view_embs: np.ndarray) -> np.ndarray:
        return combine_views(self._per_view_probs(view_embs), self.prior)

    def guess(self, probs: np.ndarray) -> tuple[float, float]:
        return expected_score_guess(probs, self.candidates, self._s)[0]

    # --- persistence ----------------------------------------------------------------------
    def _arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, kind=self.kind, centroids=self.centroids, prior=self.prior, **self._arrays())

    @classmethod
    def load(cls, path: str | Path) -> _GeocellGuesser:
        """Load any saved geocell guesser (files from before M1 hold a nearest-mean one)."""
        with np.load(path) as z:
            kind = str(z["kind"]) if "kind" in z.files else NearestMeanGuesser.kind
            arrays = {k: z[k] for k in z.files if k != "kind"}
        kinds = {c.kind: c for c in (CellGuesser, NearestMeanGuesser)}
        if kind not in kinds:
            raise ValueError(f"{path}: unknown guesser kind {kind!r}")
        return kinds[kind]._from_arrays(arrays)


class CellGuesser(_GeocellGuesser):
    """Trained softmax head over geocells (M1)."""

    kind = "softmax"

    def __init__(self, centroids, weights, bias, mean, scale, prior):
        super().__init__(centroids, prior)
        self.weights = np.asarray(weights, np.float32)  # (D, K)
        self.bias = np.asarray(bias, np.float32)  # (K,)
        self.mean = np.asarray(mean, np.float32)  # (D,) training embedding mean
        self.scale = np.asarray(scale, np.float32)  # (D,) training embedding std

    def _per_view_probs(self, view_embs: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(view_embs).astype(np.float32) - self.mean) / self.scale
        return _softmax(x @ self.weights + self.bias)

    @classmethod
    def fit(
        cls,
        view_embs: np.ndarray,
        view_lat: np.ndarray,
        view_lon: np.ndarray,
        *,
        n_cells: int,
        l2: float = 1e-3,
        epochs: int = 300,
        lr: float = 0.01,
        iters: int = 50,
        seed: int = 0,
        log: Callable[[str], None] | None = None,
    ) -> CellGuesser:
        """Cluster the training locations into ``n_cells`` geocells, then train the head on
        (view embedding -> its panorama's cell). Pass only the TRAIN split."""
        centroids, labels = fit_geocells(view_lat, view_lon, n_cells, iters, seed)
        k = len(centroids)
        x = view_embs.astype(np.float32)
        mean, scale = x.mean(0), x.std(0) + 1e-6
        w, b = train_softmax((x - mean) / scale, labels, k, l2=l2, epochs=epochs, lr=lr, log=log)
        return cls(centroids, w, b, mean, scale, _prior(labels, k))

    def _arrays(self):
        return {"weights": self.weights, "bias": self.bias, "mean": self.mean, "scale": self.scale}

    @classmethod
    def _from_arrays(cls, a):
        return cls(a["centroids"], a["weights"], a["bias"], a["mean"], a["scale"], a["prior"])


class NearestMeanGuesser(_GeocellGuesser):
    """The M0 placeholder, kept as a baseline: softmax(T * cosine to each cell's mean)."""

    kind = "nearest_mean"

    def __init__(self, centroids, cell_means, prior, temperature: float = 20.0):
        super().__init__(centroids, prior)
        self.cell_means = np.asarray(cell_means, np.float32)  # (K, D), L2-normalised
        self.temperature = float(temperature)

    def _per_view_probs(self, view_embs: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(view_embs).astype(np.float32)
        return _softmax(self.temperature * (v @ self.cell_means.T))

    @classmethod
    def fit(cls, view_embs, view_lat, view_lon, *, n_cells, temperature=20.0, iters=50, seed=0):
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
        return cls(centroids, means, _prior(labels, k), temperature)

    def _arrays(self):
        return {"cell_means": self.cell_means, "temperature": np.float64(self.temperature)}

    @classmethod
    def _from_arrays(cls, a):
        return cls(a["centroids"], a["cell_means"], a["prior"], float(a["temperature"]))


def train_softmax(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    l2: float = 1e-3,
    epochs: int = 300,
    lr: float = 0.01,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression, full-batch Adam, loss = mean cross-entropy +
    (l2 / 2) * ||W||^2. Returns (W (D, K), b (K,)). The bias starts at the log class prior."""
    n, d = x.shape
    x = x.astype(np.float32)
    onehot = np.zeros((n, k), np.float32)
    onehot[np.arange(n), y] = 1.0
    w = np.zeros((d, k), np.float32)
    b = np.log(_prior(y, k)).astype(np.float32)
    params = [w, b]
    moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]  # Adam first/second moments
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        p = _softmax(x @ w + b)
        g = (p - onehot) / n
        grads = [x.T @ g + l2 * w, g.sum(0)]
        for param, grad, (m, v) in zip(params, grads, moments, strict=True):
            m *= beta1
            m += (1 - beta1) * grad
            v *= beta2
            v += (1 - beta2) * grad * grad
            param -= lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
        if log and (t == 1 or t % 100 == 0 or t == epochs):
            loss = -np.log(p[np.arange(n), y] + 1e-12).mean() + 0.5 * l2 * float((w * w).sum())
            log(f"    epoch {t}: loss {loss:.4f}, train acc {(p.argmax(1) == y).mean():.3f}")
    return w, b
