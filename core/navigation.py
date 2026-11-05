"""Neighbor selection for the FORWARD action (PLAN.md 7.3). Used by both the Gym env and the
game so they always move to the same panorama."""

from __future__ import annotations

from collections.abc import Sequence

from core.view_spec import signed_angle_diff

MAX_FORWARD_ANGLE_DEG = 67.5  # wider than this off the current heading is not "forward"


def select_forward_neighbor(
    heading_deg: float,
    neighbors: Sequence[tuple[str, float]],
    max_angle_deg: float = MAX_FORWARD_ANGLE_DEG,
) -> str | None:
    """Pick the neighbor whose bearing is closest to ``heading_deg``.

    Args:
        neighbors: (node_id, bearing_deg) pairs.

    Returns:
        The chosen node_id, or None if no neighbor lies within ``max_angle_deg``. Ties go to
        the lexicographically smallest node_id so the choice is deterministic.
    """
    best: tuple[float, str] | None = None
    for node_id, bearing in neighbors:
        diff = abs(signed_angle_diff(bearing, heading_deg))
        if diff <= max_angle_deg and (best is None or (diff, node_id) < best):
            best = (diff, node_id)
    return None if best is None else best[1]
