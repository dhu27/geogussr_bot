"""Play rounds: harness observations -> encoder -> policy/guesser -> harness actions."""

from __future__ import annotations

import time

import numpy as np

from core.interfaces import Action, Encoder, Guesser, Observation, Policy
from core.scoring import haversine_km
from core.view_spec import HEADING_STEP_DEG
from harness.harness import GameHarness


def play_round(
    harness: GameHarness, encoder: Encoder, guesser: Guesser, policy: Policy, max_steps: int = 32
) -> dict:
    """Play the round currently on screen. Returns a log record (state read for logging only)."""
    t0 = time.monotonic()
    policy.reset()
    embs = [encoder.embed([harness.observe()])[0]]
    rel = [0.0]
    step = 0
    while True:
        obs = Observation(np.stack(embs), np.array(rel), step)
        action = policy.act(obs) if step < max_steps else Action.GUESS
        if action == Action.GUESS:
            break
        harness.act(action)
        rel.append(rel[-1] + (HEADING_STEP_DEG if action == Action.TURN_R else -HEADING_STEP_DEG))
        embs.append(encoder.embed([harness.observe()])[0])
        step += 1

    lat, lon = guesser.guess(guesser.cell_probs(np.stack(embs)))
    placed = harness.place_guess(lat, lon)
    result = harness.submit()
    state = harness.game_state()
    return {
        "round_id": state["round_id"],
        "n_views": len(embs),
        "model_guess": [lat, lon],
        "placed_guess": [placed["lat"], placed["lon"]],
        "click_error_km": haversine_km(lat, lon, placed["lat"], placed["lon"]),
        **result,
        "seconds": time.monotonic() - t0,
    }
