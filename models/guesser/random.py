"""Random baseline guesser for harness evaluations."""

from __future__ import annotations

import numpy as np


class RandomGuesser:
    """Choose a uniformly random training location for each submitted guess."""

    def __init__(self, locations: np.ndarray, seed: int = 0):
        locations = np.asarray(locations, dtype=np.float64)
        if locations.ndim != 2 or locations.shape[1] != 2 or len(locations) == 0:
            raise ValueError("locations must be a non-empty (n, 2) array")
        if not np.isfinite(locations).all():
            raise ValueError("locations must contain only finite values")
        self.locations = locations
        self.rng = np.random.default_rng(seed)
        self._selected = 0

    def cell_probs(self, view_embs: np.ndarray) -> np.ndarray:
        """Return a uniform one-cell belief while selecting a fresh random location."""
        np.asarray(view_embs)
        self._selected = int(self.rng.integers(len(self.locations)))
        return np.ones(1, dtype=np.float64)

    def guess(self, probs: np.ndarray) -> tuple[float, float]:
        if np.asarray(probs).shape != (1,):
            raise ValueError("RandomGuesser expects the one-cell probability output")
        lat, lon = self.locations[self._selected]
        return float(lat), float(lon)
