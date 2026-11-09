"""Mapillary coverage vector tiles: where images are, without a server-side search.

Why tiles: the box search (``MapillaryClient.search_images``) takes 1-20 s per call and returns
500s in dense areas (docs/DATA.md). A tile is a pre-built file, so it is fast and reliable.

The world is cut into a grid of squares ("slippy map" tiles): at zoom ``z`` there are 2**z columns
and 2**z rows, and a tile is addressed as (z, x, y). A zoom-14 tile is about 1.5-2.5 km across.
Its ``image`` layer lists every image inside with id, sequence, creator and ``is_pano``. The
quality score, exact position and thumbnail URLs are NOT in a tile: they come from
``MapillaryClient.image(id)`` for the images we decide to keep.

The token goes in the query string here (the tile server wants it there), so errors are
re-raised without the URL to keep the token out of logs.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import requests

from core.config import load_dotenv
from data.mapillary import RETRY_STATUS
from data.mvt import decode_tile

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
IMAGE_LAYER = "image"
ZOOM = 14  # Mapillary serves its image layer at this zoom


class TileError(RuntimeError):
    """A tile could not be fetched (the message never contains the token)."""


def tile_for(lat: float, lon: float, z: int = ZOOM) -> tuple[int, int]:
    """The (x, y) of the zoom-``z`` tile containing a point (Web Mercator tile scheme)."""
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tiles_around(x: int, y: int, z: int, radius: int) -> list[tuple[int, int]]:
    """The square block of tiles within ``radius`` of (x, y) (radius 1 -> 3x3), clipped to the
    grid. Row by row, west to east."""
    n = 2**z
    return [
        (xx, yy)
        for yy in range(max(y - radius, 0), min(y + radius, n - 1) + 1)
        for xx in range(max(x - radius, 0), min(x + radius, n - 1) + 1)
    ]


def _plain_int(value: Any) -> Any:
    """Tile numbers can arrive as floats; a float id would stringify as '1.0e+15'."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def parse_image_layer(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn decoded ``image`` layer features into candidate dicts. Features without an id or a
    sequence are dropped."""
    out = []
    for feat in features:
        props = feat.get("properties") or {}
        image_id = _plain_int(props.get("id", feat.get("id")))
        sequence_id = _plain_int(props.get("sequence_id"))
        if image_id is None or sequence_id is None:
            continue
        out.append(
            {
                "id": str(image_id),
                "sequence_id": str(sequence_id),
                "creator_id": str(_plain_int(props.get("creator_id", ""))),
                "is_pano": bool(props.get("is_pano")),
                "captured_at": int(props.get("captured_at") or 0),
            }
        )
    return out


class MapillaryTiles:
    """Fetches coverage tiles. ``fetch`` returns the panoramas in a tile."""

    def __init__(self, token: str | None = None, timeout_s: float = 60.0, max_retries: int = 3):
        load_dotenv()
        self.token = token or os.environ.get("MAPILLARY_TOKEN")
        if not self.token:
            raise RuntimeError("MAPILLARY_TOKEN is not set. Add it to .env (see README).")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = requests.Session()

    def fetch_raw(self, z: int, x: int, y: int) -> bytes:
        """The tile's bytes; empty if Mapillary has nothing there (204 / 404)."""
        url = TILE_URL.format(z=z, x=x, y=y)
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            try:
                r = self.session.get(
                    url, params={"access_token": self.token}, timeout=self.timeout_s
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                if last:
                    raise TileError(f"tile {z}/{x}/{y}: {type(e).__name__}") from None
                time.sleep(2**attempt)
                continue
            if r.status_code in RETRY_STATUS and not last:
                time.sleep(2**attempt)
                continue
            if r.status_code in (204, 404):
                return b""
            if r.status_code >= 400:
                raise TileError(f"tile {z}/{x}/{y}: HTTP {r.status_code}")
            return r.content
        raise TileError(f"tile {z}/{x}/{y}: gave up")  # pragma: no cover

    def fetch(self, z: int, x: int, y: int) -> list[dict[str, Any]]:
        """Panoramas in tile (z, x, y): dicts with id, sequence_id, creator_id, is_pano=True,
        captured_at."""
        raw = self.fetch_raw(z, x, y)
        if not raw:
            return []
        features = decode_tile(raw, [IMAGE_LAYER]).get(IMAGE_LAYER, [])
        return [c for c in parse_image_layer(features) if c["is_pano"]]
