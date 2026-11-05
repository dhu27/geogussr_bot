"""Fixed strategies implementing core.interfaces.Policy. Phase 1 uses LookAround; PPO (M7)
must beat these (decision point G2)."""

from __future__ import annotations

import numpy as np

from core.interfaces import Action, Observation


class LookAroundPolicy:
    """Turn right until ``n_views`` views have been seen, then guess. n_views=1: guess at once."""

    def __init__(self, n_views: int = 8):
        self.n_views = n_views

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        return Action.GUESS if len(obs.view_embs) >= self.n_views else Action.TURN_R


class RandomPolicy:
    """Uniform over turns, guessing with probability ``p_guess`` each step (or at ``max_steps``)."""

    def __init__(self, p_guess: float = 0.2, max_steps: int = 16, seed: int = 0):
        self.p_guess, self.max_steps = p_guess, max_steps
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        if obs.step >= self.max_steps or self.rng.random() < self.p_guess:
            return Action.GUESS
        return Action(self.rng.choice([Action.TURN_L, Action.TURN_R]))


def make_policy(cfg: dict):
    if cfg.get("policy", "look_around") == "look_around":
        return LookAroundPolicy(cfg.get("n_views", 8))
    if cfg["policy"] == "random":
        return RandomPolicy(seed=cfg.get("seed", 0))
    raise ValueError(f"unknown policy {cfg['policy']!r}")
