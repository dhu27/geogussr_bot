"""Minimal Mapillary Graph API v4 client.

The token comes from ``MAPILLARY_TOKEN`` (put it in the gitignored .env). It is sent in the
Authorization header, never in URLs, so it doesn't end up in logs.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from core.config import load_dotenv

GRAPH_URL = "https://graph.mapillary.com"
# Full record: fetch per image. Asking for these (thumb URLs especially) in a bbox search makes
# the API return 500s, so searches use SEARCH_FIELDS and details come from ``image()``.
IMAGE_FIELDS = (
    "id,sequence,creator,computed_geometry,geometry,captured_at,is_pano,"
    "compass_angle,computed_compass_angle,quality_score,thumb_1024_url,thumb_2048_url"
)
SEARCH_FIELDS = "id,sequence,creator,is_pano,quality_score"
RETRY_STATUS = (429, 500, 502, 503, 504)


class MapillaryClient:
    """Notes from probing the API (Nov 2026): bbox searches are slow (1-20 s) and return 500s
    in dense city centres or for boxes over ~1 km; per-image lookups take ~0.2 s."""

    def __init__(self, token: str | None = None, timeout_s: float = 60.0, max_retries: int = 3):
        load_dotenv()
        self.token = token or os.environ.get("MAPILLARY_TOKEN")
        if not self.token:
            raise RuntimeError("MAPILLARY_TOKEN is not set. Add it to .env (see README).")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"OAuth {self.token}"

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            try:
                r = self.session.get(f"{GRAPH_URL}/{path}", params=params, timeout=self.timeout_s)
            except (requests.Timeout, requests.ConnectionError):
                if last:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code in RETRY_STATUS and not last:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def search_images(
        self,
        bbox: tuple[float, float, float, float],
        *,
        is_pano: bool = True,
        limit: int = 50,
        fields: str = SEARCH_FIELDS,
    ) -> list[dict[str, Any]]:
        """Images inside ``bbox`` = (min_lon, min_lat, max_lon, max_lat). Keep boxes small
        (~1 km) and fields light; then call ``image()`` for the ones you keep."""
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "fields": fields,
            "limit": limit,
        }
        if is_pano:
            params["is_pano"] = "true"
        return self._get("images", params).get("data", [])

    def image(self, image_id: str) -> dict[str, Any]:
        return self._get(image_id, {"fields": IMAGE_FIELDS})

    def download(self, url: str) -> bytes:
        # Thumb URLs are pre-signed CDN links; they don't need (or want) the auth header.
        r = requests.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.content


def to_row(img: dict[str, Any]) -> dict[str, Any]:
    """Flatten an API image record into the images.parquet columns we can fill from it."""
    geom = img.get("computed_geometry") or img.get("geometry")
    lon, lat = geom["coordinates"]
    compass = img.get("computed_compass_angle")
    if compass is None:
        compass = img.get("compass_angle")
    creator = img.get("creator") or {}
    seq = img.get("sequence")
    return {
        "image_id": str(img["id"]),
        "sequence_id": str(seq["id"] if isinstance(seq, dict) else seq),
        "creator_id": str(creator.get("id", "")),
        "creator_username": creator.get("username", ""),
        "lat": float(lat),
        "lon": float(lon),
        "captured_at": int(img.get("captured_at") or 0),
        "is_pano": bool(img.get("is_pano")),
        "compass_angle": float(compass or 0.0) % 360.0,
        "quality_score": float("nan") if img.get("quality_score") is None else img["quality_score"],
    }
