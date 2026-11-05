from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import FIXTURES_DIR
from data.dataset import load_images, load_views


@pytest.fixture(scope="session")
def fixture_images() -> pd.DataFrame:
    if not (FIXTURES_DIR / "images.parquet").is_file():
        pytest.fail("tests/fixtures/ is not built: run `uv run python scripts/make_fixtures.py`")
    return load_images(FIXTURES_DIR)


@pytest.fixture(scope="session")
def fixture_views(fixture_images) -> tuple[pd.DataFrame, np.ndarray]:
    return load_views(FIXTURES_DIR)


@pytest.fixture
def column_coded_pano() -> np.ndarray:
    """512x256 equirectangular test image: R = column (u), G = row (v), B = 0. Lets tests read
    back exactly which panorama pixel a crop sampled."""
    h, w = 256, 512
    u = np.linspace(0, 255, w)[None, :].repeat(h, 0)
    v = np.linspace(0, 255, h)[:, None].repeat(w, 1)
    return np.stack([u, v, np.zeros_like(u)], -1).round().astype(np.uint8)
