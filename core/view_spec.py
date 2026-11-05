"""The view spec shared by training crops and the game's rendered view (PLAN.md 6.2).

Changing anything here invalidates every cached embedding. Change only via reviewed PR.

Heading convention
------------------
All headings in this project are *compass headings*: degrees clockwise from true north, in
[0, 360). A panorama's ``compass_angle`` is the compass heading of the centre column of its
equirectangular image. Pannellum's ``yaw`` (and the crop function's internal yaw) is relative to
that centre column, positive to the right. Use :func:`heading_to_pano_yaw` to convert.
"""

from __future__ import annotations

ENCODER_NAME = "geolocal/StreetCLIP"  # CLIP ViT-L/14 @ 336 px, fine-tuned for geolocation
EMBED_DIM = 768

VIEW_SIZE = 336  # square, pixels; equals StreetCLIP's native input so no resize is needed
HFOV_DEG = 90.0
PITCH_DEG = 0.0
N_HEADINGS = 8
HEADING_STEP_DEG = 360.0 / N_HEADINGS
HEADINGS_DEG: tuple[float, ...] = tuple(i * HEADING_STEP_DEG for i in range(N_HEADINGS))


def wrap_deg(angle: float) -> float:
    """Wrap an angle to [0, 360)."""
    return angle % 360.0


def signed_angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b, in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def heading_to_pano_yaw(heading_deg: float, compass_angle: float) -> float:
    """Compass heading -> yaw relative to the panorama's centre column, in (-180, 180]."""
    return signed_angle_diff(heading_deg, compass_angle)


def pano_yaw_to_heading(yaw_deg: float, compass_angle: float) -> float:
    """Yaw relative to the panorama's centre column -> compass heading in [0, 360)."""
    return wrap_deg(yaw_deg + compass_angle)
