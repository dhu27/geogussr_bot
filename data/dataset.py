"""Read and write a dataset directory (layout in core/schemas.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.schemas import validate_images, validate_views


def load_images(root: str | Path, check_files: bool = True) -> pd.DataFrame:
    root = Path(root)
    return validate_images(pd.read_parquet(root / "images.parquet"), root if check_files else None)


def write_images(df: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    root = Path(root)
    df = validate_images(df, root)
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "images.parquet", index=False)
    return df


def load_views(root: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    root = Path(root)
    emb = np.load(root / "embeddings.npy")
    return validate_views(pd.read_parquet(root / "views.parquet"), emb), emb


def write_views(views: pd.DataFrame, embeddings: np.ndarray, root: str | Path) -> None:
    root = Path(root)
    views = validate_views(views, embeddings)
    root.mkdir(parents=True, exist_ok=True)
    views.to_parquet(root / "views.parquet", index=False)
    np.save(root / "embeddings.npy", embeddings)
