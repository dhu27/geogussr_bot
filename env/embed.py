"""Crop every panorama at the standard headings and embed the views (one-time pass).

Resumable: panoramas are processed in chunks, and each finished chunk is saved under
``<out_dir>/.embed_parts/``. Re-running the same command after an interruption reuses the saved
chunks and only embeds the rest. A chunk is reused only if it holds the same images embedded by
the same encoder. Once every chunk is done they are joined into views.parquet + embeddings.npy
and the parts folder is removed.

StreetCLIP runs at about 8 views/s on an M3 Pro (Nov 2026): roughly 70 min for the 4,058-panorama
pilot and 14 h for ~400,000 views, so the full set is a Colab/GPU job or an overnight run.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from core.interfaces import Encoder
from core.schemas import view_id
from core.view_spec import HEADINGS_DEG
from core.views import crop_all_headings, load_pano
from data.dataset import write_views

PARTS_DIR = ".embed_parts"
CHUNK_PANOS = 50  # 400 views: under a minute of StreetCLIP on a laptop, ~135 MB of crops


def encoder_tag(encoder: Encoder) -> str:
    """Identifies which encoder made a saved chunk, so a chunk from another one is not reused."""
    return f"{type(encoder).__name__}:{getattr(encoder, 'model_name', '')}:{encoder.dim}"


def _load_part(path: Path, image_ids: list[str], tag: str) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as z:
            if z["tag"].item() != tag or z["image_ids"].tolist() != image_ids:
                return None
            emb = z["emb"]
    except (OSError, ValueError, KeyError):  # a half-written or foreign file: redo the chunk
        return None
    return emb if emb.shape == (len(image_ids) * len(HEADINGS_DEG), emb.shape[1]) else None


def _save_part(path: Path, image_ids: list[str], tag: str, emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.savez(f, image_ids=np.array(image_ids), tag=np.array(tag), emb=emb)
    tmp.replace(path)  # atomic: a chunk file is either complete or absent


def embed_dataset(
    images: pd.DataFrame,
    root: str | Path,
    encoder: Encoder,
    out_dir: str | Path | None = None,
    *,
    chunk_panos: int = CHUNK_PANOS,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Write views.parquet + embeddings.npy for every panorama in ``images``.

    ``root`` is where ``images.path`` resolves; outputs go to ``out_dir`` (default: ``root``).
    Views are ordered as the panoramas appear in ``images``, headings in HEADINGS_DEG order.
    """
    root = Path(root)
    out = Path(out_dir or root)
    panos = images[images["is_pano"]].reset_index(drop=True)
    ids = panos["image_id"].astype(str).tolist()
    tag = encoder_tag(encoder)
    n_chunks = math.ceil(len(ids) / chunk_panos)
    embs: list[np.ndarray] = []
    t0, fresh_views, reused = time.monotonic(), 0, 0
    for k in range(n_chunks):
        sl = slice(k * chunk_panos, (k + 1) * chunk_panos)
        chunk_ids = ids[sl]
        path = out / PARTS_DIR / f"part-{k:05d}.npz"
        emb = _load_part(path, chunk_ids, tag)
        if emb is None:
            crops = [
                crop
                for img in panos.iloc[sl].itertuples()
                for crop in crop_all_headings(load_pano(root / img.path), img.compass_angle)
            ]
            emb = encoder.embed(crops).astype(np.float16)
            _save_part(path, chunk_ids, tag, emb)
            fresh_views += len(crops)
            if log:
                rate = fresh_views / max(time.monotonic() - t0, 1e-9)
                left = (len(ids) - min(len(ids), (k + 1) * chunk_panos)) * len(HEADINGS_DEG)
                log(
                    f"  chunk {k + 1}/{n_chunks}: {rate:.1f} views/s, "
                    f"about {left / rate / 60:.0f} min left"
                )
        else:
            reused += 1
        embs.append(emb)
    if log and reused:
        log(f"  reused {reused} chunk(s) saved by an earlier run")

    rows = [
        {"view_id": view_id(i, hd), "image_id": i, "heading_deg": hd, "row": r}
        for r, (i, hd) in enumerate((i, hd) for i in ids for hd in HEADINGS_DEG)
    ]
    views = pd.DataFrame(rows, columns=["view_id", "image_id", "heading_deg", "row"])
    emb = np.concatenate(embs) if embs else np.zeros((0, encoder.dim), np.float16)
    write_views(views, emb, out)
    shutil.rmtree(out / PARTS_DIR, ignore_errors=True)
    return views, emb
