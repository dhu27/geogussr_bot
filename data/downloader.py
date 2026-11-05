"""Bulk downloader. PLACEHOLDER (M0): copies the fixture set instead of hitting the API.

M1 replaces ``download`` with the real sampler + downloader (PLAN.md 7.1): coverage tiles ->
regions -> entity API -> filters -> images.parquet. The output contract stays the same.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from core.config import FIXTURES_DIR
from data.dataset import load_images, write_images
from data.splits import assign_splits, check_no_leakage


def download(out_dir: str | Path, split_fractions: dict[str, float], seed: int = 0) -> pd.DataFrame:
    """Write a dataset (images.parquet + images/) to ``out_dir``; returns the images table."""
    out_dir = Path(out_dir)
    src = load_images(FIXTURES_DIR)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for rel in src["path"]:
        shutil.copy2(FIXTURES_DIR / rel, out_dir / rel)
    src["split"] = assign_splits(src, split_fractions, seed)
    check_no_leakage(src)
    return write_images(src, out_dir)
