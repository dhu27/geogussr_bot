"""Crop-vs-screenshot consistency check (PLAN.md M2, risk "training crops differ from what the
game renders"). Reads ``window.__game`` to know the true pano/heading: this is evaluation code,
not the model's input path.

Threshold: require cosine >= 0.995, not the 0.95 in PLAN.md. StreetCLIP embeddings sit close
together. Measured on 8 pilot panoramas (Nov 2026): the same view scored >= 0.9999, the crop
45 degrees off up to 0.983, a different panorama 0.63-0.84. So 0.95 would pass a wrong heading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.interfaces import Encoder
from core.views import crop_view, load_pano
from harness.harness import GameHarness


def crop_vs_screenshot(
    harness: GameHarness, encoder: Encoder, images: pd.DataFrame, root: str | Path
) -> dict:
    """Cosine similarity between the harness screenshot and the canonical crop of the same view."""
    shot = harness.observe()
    state = harness.game_state()
    image_id = Path(state["pano_url"]).stem
    row = images.set_index("image_id").loc[image_id]
    crop = crop_view(load_pano(Path(root) / row["path"]), state["heading"], row["compass_angle"])
    a, b = encoder.embed([shot, crop])
    return {
        "image_id": image_id,
        "heading": state["heading"],
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "mean_abs_pixel_diff": float(np.abs(shot.astype(int) - crop.astype(int)).mean()),
        "screenshot": shot,
        "crop": crop,
    }
