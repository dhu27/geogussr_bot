"""Python interfaces every module codes against (PLAN.md 6.3).

Ground rule 1: everything passed into these is derived from pixels only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    dim: int

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """(VIEW_SIZE, VIEW_SIZE, 3) uint8 RGB views -> (n, dim) float32, L2-normalised."""
        ...


@runtime_checkable
class Guesser(Protocol):
    def cell_probs(self, view_embs: np.ndarray) -> np.ndarray:
        """(t, D) embeddings of every view seen so far -> (K,) combined cell probabilities."""
        ...

    def guess(self, probs: np.ndarray) -> tuple[float, float]:
        """(K,) cell probabilities -> (lat, lon) to submit."""
        ...


class Action(IntEnum):
    TURN_L = 0  # -45 degrees
    TURN_R = 1  # +45 degrees
    FORWARD = 2  # Phase 2: move to the neighbor closest to the current heading
    BACK = 3  # Phase 2: return to the previous node
    GUESS = 4


@dataclass
class Observation:
    """What a policy sees at step t. Headings are relative to the round's start heading so the
    policy cannot read absolute orientation from metadata."""

    view_embs: np.ndarray  # (t, D), every view seen so far, oldest first
    rel_headings_deg: np.ndarray  # (t,), heading of each view relative to the start
    step: int = 0
    extras: dict = field(default_factory=dict)  # Phase 2: e.g. whether FORWARD is possible


@runtime_checkable
class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: Observation) -> Action: ...
