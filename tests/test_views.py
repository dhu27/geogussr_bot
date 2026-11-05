import numpy as np
import pytest

from core.view_spec import (
    HEADINGS_DEG,
    VIEW_SIZE,
    heading_to_pano_yaw,
    pano_yaw_to_heading,
    signed_angle_diff,
)
from core.views import crop_all_headings, crop_view


def test_headings_spec():
    assert HEADINGS_DEG == (0, 45, 90, 135, 180, 225, 270, 315)


@pytest.mark.parametrize("compass", [0.0, 37.5, 180.0, 359.0])
@pytest.mark.parametrize("heading", [0.0, 45.0, 135.0, 270.0])
def test_heading_yaw_roundtrip(compass, heading):
    yaw = heading_to_pano_yaw(heading, compass)
    assert -180 < yaw <= 180
    assert abs(signed_angle_diff(pano_yaw_to_heading(yaw, compass), heading)) < 1e-9


def _centre(view):
    c = VIEW_SIZE // 2
    return view[c - 1 : c + 1, c - 1 : c + 1].astype(float).mean((0, 1))


def test_crop_shape_and_dtype(column_coded_pano):
    v = crop_view(column_coded_pano, 0.0, 0.0)
    assert v.shape == (VIEW_SIZE, VIEW_SIZE, 3) and v.dtype == np.uint8


def test_looking_at_compass_angle_sees_pano_centre(column_coded_pano):
    # Panorama centre faces compass 120; looking at 120 must sample the centre column (R ~ 127.5).
    r, g, _ = _centre(crop_view(column_coded_pano, 120.0, 120.0))
    assert abs(r - 127.5) < 1.5
    assert abs(g - 127.5) < 1.5  # pitch 0 -> horizon row


def test_turning_right_moves_right_in_pano(column_coded_pano):
    # +90 degrees of heading = a quarter of the panorama width to the right (R ~ 191).
    r, _, _ = _centre(crop_view(column_coded_pano, 90.0, 0.0))
    assert abs(r - 255 * 0.75) < 1.5


def test_field_of_view_edges(column_coded_pano):
    # hfov 90: left and right edges of the view sample yaw -45 and +45 degrees.
    v = crop_view(column_coded_pano, 0.0, 0.0).astype(float)
    mid = VIEW_SIZE // 2
    left, right = v[mid, 0, 0], v[mid, -1, 0]
    assert abs(left - 255 * (0.5 - 45 / 360)) < 2
    assert abs(right - 255 * (0.5 + 45 / 360)) < 2


def test_seam_wraps(column_coded_pano):
    # Looking backwards straddles the seam: left half samples the right end of the pano.
    v = crop_view(column_coded_pano, 180.0, 0.0).astype(float)
    mid = VIEW_SIZE // 2
    assert v[mid, 10, 0] > 200 and v[mid, -10, 0] < 55


def test_pitch_up_looks_higher(column_coded_pano):
    _, g, _ = _centre(crop_view(column_coded_pano, 0.0, 0.0, pitch_deg=30.0))
    assert abs(g - 255 * (0.5 - 30 / 180)) < 2


def test_crop_all_headings_are_distinct(column_coded_pano):
    views = crop_all_headings(column_coded_pano, 10.0)
    assert len(views) == 8
    centres = [_centre(v)[0] for v in views]
    assert len({round(c) for c in centres}) == 8
