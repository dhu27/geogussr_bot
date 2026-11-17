"""The one canonical function that cuts a perspective view from an equirectangular panorama.

Both the training pipeline (env/) and any consistency check against the game call this. The
game's Pannellum viewer uses the same rectilinear projection, so a crop here and a screenshot
of ``#pano`` at the same heading should embed almost identically (PLAN.md M2 check).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from core.view_spec import HEADINGS_DEG, HFOV_DEG, PITCH_DEG, VIEW_SIZE, heading_to_pano_yaw


def load_pano(path: str | Path) -> np.ndarray:
    """Load an equirectangular panorama as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def crop_view(
    pano: np.ndarray,
    heading_deg: float,
    compass_angle: float,
    *,
    hfov_deg: float = HFOV_DEG,
    pitch_deg: float = PITCH_DEG,
    size: int = VIEW_SIZE,
) -> np.ndarray:
    """Render a square rectilinear view of ``pano`` looking at compass ``heading_deg``.

    Args:
        pano: (H, W, 3) uint8 equirectangular image; its centre column faces ``compass_angle``.
        heading_deg: compass heading to look at (degrees clockwise from north).
        compass_angle: compass heading of the panorama's centre column (Mapillary metadata).
        hfov_deg: horizontal field of view. The output is square, so vfov == hfov.
        pitch_deg: positive looks up.
        size: output width and height in pixels.

    Returns:
        (size, size, 3) uint8 array.
    """
    if pano.ndim != 3 or pano.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) panorama, got shape {pano.shape}")
    h, w = pano.shape[:2]
    yaw = np.deg2rad(heading_to_pano_yaw(heading_deg, compass_angle))
    ray_lon, lat = _rays(hfov_deg, pitch_deg, size)
    lon = ray_lon + yaw  # 0 = panorama centre, positive = right
    u = (lon / (2.0 * np.pi) + 0.5) * w - 0.5
    v = (0.5 - lat / np.pi) * h - 0.5
    return _bilinear(pano, u, v)


@lru_cache(maxsize=8)
def _rays(hfov_deg: float, pitch_deg: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """(longitude before yaw, latitude) of each output pixel's ray. Depends only on the view
    spec, so it is computed once rather than per crop."""
    pitch = np.deg2rad(pitch_deg)
    f = 1.0 / np.tan(np.deg2rad(hfov_deg) / 2.0)
    # Camera rays through pixel centres: x right, y up, z forward.
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(t, -t)
    z = np.full_like(x, f)
    # Pitch: rotate about the x axis.
    y, z = y * np.cos(pitch) + z * np.sin(pitch), -y * np.sin(pitch) + z * np.cos(pitch)
    lon, lat = np.arctan2(x, z), np.arctan2(y, np.hypot(x, z))
    lon.flags.writeable = lat.flags.writeable = False  # shared by every call
    return lon, lat


def crop_all_headings(
    pano: np.ndarray, compass_angle: float, headings: tuple[float, ...] = HEADINGS_DEG
) -> list[np.ndarray]:
    """Crop the standard set of views (8 headings at 45 degrees) from one panorama."""
    return [crop_view(pano, hd, compass_angle) for hd in headings]


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample ``img`` at float pixel coords; wraps horizontally, clamps vertically."""
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    u1 = (u0 + 1) % w
    u0 = u0 % w
    v1 = np.clip(v0 + 1, 0, h - 1)
    v0 = np.clip(v0, 0, h - 1)
    # np.take on the flattened image is ~4x faster than img[v, u] fancy indexing, and only the
    # sampled pixels are converted to float. Same values, same arithmetic as before.
    flat = img.reshape(-1, img.shape[2])

    def px(vv, uu):
        return np.take(flat, vv * w + uu, axis=0).astype(np.float32)

    top = px(v0, u0) * (1 - du) + px(v0, u1) * du
    bot = px(v1, u0) * (1 - du) + px(v1, u1) * du
    out = top * (1 - dv) + bot * dv
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
