"""Crop every panorama at the standard headings and embed the views (one-time pass)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.interfaces import Encoder
from core.schemas import view_id
from core.view_spec import HEADINGS_DEG
from core.views import crop_all_headings, load_pano
from data.dataset import write_views


def embed_dataset(
    images: pd.DataFrame, root: str | Path, encoder: Encoder, out_dir: str | Path | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Write views.parquet + embeddings.npy for every panorama in ``images``.

    ``root`` is where ``images.path`` resolves; outputs go to ``out_dir`` (default: ``root``).
    """
    root = Path(root)
    rows, embs = [], []
    for img in images[images["is_pano"]].itertuples():
        crops = crop_all_headings(load_pano(root / img.path), img.compass_angle)
        embs.append(encoder.embed(crops))
        for hd in HEADINGS_DEG:
            rows.append(
                {
                    "view_id": view_id(img.image_id, hd),
                    "image_id": img.image_id,
                    "heading_deg": hd,
                    "row": len(rows),
                }
            )
    views = pd.DataFrame(rows)
    emb = (
        np.concatenate(embs).astype(np.float16) if embs else np.zeros((0, encoder.dim), np.float16)
    )
    write_views(views, emb, out_dir or root)
    return views, emb
