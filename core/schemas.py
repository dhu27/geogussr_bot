"""On-disk data formats (PLAN.md 6.1). Every writer calls ``validate_*`` before saving.

A dataset directory looks like::

    <dataset>/
      images.parquet      # one row per Mapillary image
      images/<id>.jpg     # ``images.path`` is relative to <dataset>
      views.parquet       # one row per (panorama, heading)
      embeddings.npy      # (n_views, EMBED_DIM) float16, row i <-> views.row == i
      edges.parquet       # Phase 2
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from core.view_spec import EMBED_DIM

SPLITS = ("train", "val", "test")
CONTINENTS = ("AF", "AS", "EU", "NA", "OC", "SA")

IMAGES_COLUMNS: Mapping[str, str] = {
    "image_id": "string",
    "sequence_id": "string",
    "creator_id": "string",
    "lat": "float64",  # computed_geometry, falling back to geometry
    "lon": "float64",
    "captured_at": "int64",  # unix ms
    "is_pano": "bool",
    "compass_angle": "float64",  # computed_compass_angle, falling back to compass_angle
    "quality_score": "float64",  # NaN when Mapillary has none
    "region_id": "string",
    "country": "string",  # ISO 3166-1 alpha-2 of the region (for balancing)
    "continent": "string",  # one of CONTINENTS (for per-continent metrics)
    "split": "string",  # one of SPLITS
    "path": "string",  # relative to the dataset directory
}

VIEWS_COLUMNS: Mapping[str, str] = {
    "view_id": "string",  # f"{image_id}_{heading:03d}"
    "image_id": "string",
    "heading_deg": "float64",  # compass heading, see core.view_spec
    "row": "int64",  # index into embeddings.npy
}

EDGES_COLUMNS: Mapping[str, str] = {
    "src": "string",
    "dst": "string",
    "kind": "string",  # "seq" | "spatial"
    "bearing_deg": "float64",  # compass bearing src -> dst
    "dist_m": "float64",
}


class SchemaError(ValueError):
    pass


def view_id(image_id: str, heading_deg: float) -> str:
    return f"{image_id}_{round(heading_deg) % 360:03d}"


def coerce(df: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    """Select and order the contract columns, casting to the contract dtypes."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    return df[list(columns)].astype(dict(columns)).reset_index(drop=True)


def validate_images(df: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    df = coerce(df, IMAGES_COLUMNS)
    _check_unique(df, "image_id")
    _check_no_nulls(df, [c for c in IMAGES_COLUMNS if c != "quality_score"])
    if not df["lat"].between(-90, 90).all() or not df["lon"].between(-180, 180).all():
        raise SchemaError("lat/lon out of range")
    if not df["compass_angle"].between(0, 360).all():
        raise SchemaError("compass_angle must be in [0, 360]")
    _check_in(df, "split", SPLITS)
    _check_in(df, "continent", CONTINENTS)
    if root is not None:
        absent = [p for p in df["path"] if not (Path(root) / p).is_file()]
        if absent:
            raise SchemaError(f"{len(absent)} image files missing under {root}, e.g. {absent[0]}")
    return df


def validate_views(df: pd.DataFrame, embeddings: np.ndarray | None = None) -> pd.DataFrame:
    df = coerce(df, VIEWS_COLUMNS)
    _check_unique(df, "view_id")
    _check_no_nulls(df, list(VIEWS_COLUMNS))
    if embeddings is not None:
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
            raise SchemaError(f"embeddings must be (n, {EMBED_DIM}), got {embeddings.shape}")
        if embeddings.dtype != np.float16:
            raise SchemaError(f"embeddings must be float16, got {embeddings.dtype}")
        if df["row"].min() < 0 or df["row"].max() >= len(embeddings):
            raise SchemaError("views.row out of range for embeddings.npy")
    return df


def validate_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce(df, EDGES_COLUMNS)
    _check_no_nulls(df, list(EDGES_COLUMNS))
    _check_in(df, "kind", ("seq", "spatial"))
    return df


def _check_unique(df: pd.DataFrame, col: str) -> None:
    if df[col].duplicated().any():
        raise SchemaError(f"duplicate values in {col}")


def _check_no_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        raise SchemaError(f"nulls in columns: {bad}")


def _check_in(df: pd.DataFrame, col: str, allowed: tuple[str, ...]) -> None:
    bad = set(df[col].unique()) - set(allowed)
    if bad:
        raise SchemaError(f"{col} has values outside {allowed}: {sorted(bad)}")
