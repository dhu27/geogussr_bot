"""Offline single-panorama environment over precomputed embeddings (every step is a table
lookup). Follows the Gymnasium API without depending on it yet.

PLACEHOLDER (M0): turning only; FORWARD/BACK are no-ops until the Phase 2 graph env (M5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Action, Guesser, Observation
from core.scoring import haversine_km, score
from core.view_spec import HEADING_STEP_DEG, N_HEADINGS


class LookAroundEnv:
    def __init__(
        self,
        images: pd.DataFrame,
        views: pd.DataFrame,
        embeddings: np.ndarray,
        guesser: Guesser,
        *,
        step_cost: float = 0.0,
        max_steps: int = 32,
    ):
        self.images = images.set_index("image_id")
        self.guesser = guesser
        self.step_cost = step_cost
        self.max_steps = max_steps
        self.emb = embeddings
        # rows[image_id] -> (N_HEADINGS,) embedding row indices ordered by heading
        v = views.sort_values(["image_id", "heading_deg"])
        self.rows = {iid: g["row"].to_numpy() for iid, g in v.groupby("image_id")}
        self.image_ids = sorted(set(self.rows) & set(self.images.index))
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None, image_id: str | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.image_id = image_id or self.image_ids[self.rng.integers(len(self.image_ids))]
        self.start = int(self.rng.integers(N_HEADINGS))
        self.offset = 0  # heading index relative to start
        self.t = 0
        self.seen: list[int] = [self.offset]
        return self._obs(), {"image_id": self.image_id}

    def _obs(self) -> Observation:
        idx = [(self.start + o) % N_HEADINGS for o in self.seen]
        embs = self.emb[self.rows[self.image_id][idx]].astype(np.float32)
        return Observation(embs, np.array(self.seen, float) * HEADING_STEP_DEG, self.t)

    def step(self, action: Action):
        self.t += 1
        if action == Action.GUESS:
            lat, lon = self.guesser.guess(self.guesser.cell_probs(self._obs().view_embs))
            true = self.images.loc[self.image_id]
            d = haversine_km(lat, lon, true["lat"], true["lon"])
            reward = score(d) - self.step_cost * self.t
            info = {
                "distance_km": d,
                "score": score(d),
                "guess": (lat, lon),
                "n_views": len(self.seen),
            }
            return self._obs(), reward, True, False, info
        if action in (Action.TURN_L, Action.TURN_R):
            self.offset += 1 if action == Action.TURN_R else -1
            self.seen.append(self.offset)
        truncated = self.t >= self.max_steps
        return self._obs(), -self.step_cost if not truncated else 0.0, False, truncated, {}
