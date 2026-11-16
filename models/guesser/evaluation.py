"""Offline evaluation of any Guesser on one split of an embedded dataset.

Two settings per panorama:
* 1 view: each of its 8 views guessed on its own (8 rows), i.e. the average single-view result.
* 8 views: all 8 combined through ``guesser.cell_probs`` (the look-around agent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Guesser
from core.metrics import summarize_by_group
from core.scoring import haversine_km
from core.view_spec import N_HEADINGS


def view_matrix(views: pd.DataFrame, emb: np.ndarray, image_ids: list[str]) -> np.ndarray:
    """(n_panos, 8, D) embeddings, headings in order, for ``image_ids``."""
    v = views[views["image_id"].isin(image_ids)].sort_values(["image_id", "heading_deg"])
    rows = v.groupby("image_id")["row"].apply(np.asarray)
    missing = [i for i in image_ids if i not in rows.index or len(rows[i]) != N_HEADINGS]
    if missing:
        raise ValueError(f"{len(missing)} panoramas lack {N_HEADINGS} views, e.g. {missing[:3]}")
    return np.stack([emb[rows[i]] for i in image_ids]).astype(np.float32)


def predict(
    guesser: Guesser, images: pd.DataFrame, views: pd.DataFrame, emb: np.ndarray, split: str
) -> pd.DataFrame:
    """One row per guess: image_id, continent, n_views, pred_lat, pred_lon, distance_km."""
    panos = images[(images["split"] == split) & images["is_pano"]].reset_index(drop=True)
    x = view_matrix(views, emb, panos["image_id"].tolist())
    rows = []
    for p, img in enumerate(panos.itertuples()):
        settings = [(1, x[p, h : h + 1]) for h in range(N_HEADINGS)] + [(N_HEADINGS, x[p])]
        for n_views, embs in settings:
            lat, lon = guesser.guess(guesser.cell_probs(embs))
            rows.append(
                (
                    img.image_id,
                    img.continent,
                    n_views,
                    lat,
                    lon,
                    haversine_km(lat, lon, img.lat, img.lon),
                )
            )
    return pd.DataFrame(
        rows, columns=["image_id", "continent", "n_views", "pred_lat", "pred_lon", "distance_km"]
    )


def summarize(pred: pd.DataFrame) -> dict[str, dict]:
    """{"1_view": {overall, <continent>...}, "8_views": {...}} from core.metrics."""
    out = {}
    for n, g in pred.groupby("n_views"):
        out[f"{n}_view" if n == 1 else f"{n}_views"] = summarize_by_group(
            g["distance_km"].to_numpy(), g["continent"].tolist()
        )
    return out


class LargestCellGuesser:
    """Baseline: always guess the centre of the most common training cell."""

    def __init__(self, lat: float, lon: float):
        self.lat, self.lon = float(lat), float(lon)

    def cell_probs(self, view_embs: np.ndarray) -> np.ndarray:
        return np.ones(1)

    def guess(self, probs: np.ndarray) -> tuple[float, float]:
        return self.lat, self.lon
