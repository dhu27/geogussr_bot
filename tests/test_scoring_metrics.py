import numpy as np
import pytest

from core.metrics import summarize, summarize_by_group
from core.scoring import EARTH_RADIUS_KM, haversine_km, score


def test_zero_distance():
    assert haversine_km(10, 20, 10, 20) == 0
    assert score(0) == 5000


def test_known_distance_paris_london():
    assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, abs=1.0)


def test_antipodes_and_dateline():
    assert haversine_km(0, 0, 0, 180) == pytest.approx(np.pi * EARTH_RADIUS_KM)
    assert haversine_km(0, 179.5, 0, -179.5) == pytest.approx(111.2, abs=0.2)


def test_vectorised():
    d = haversine_km(np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.0, 1.0, 2.0]))
    assert d.shape == (3,) and d[0] == 0 and d[2] == pytest.approx(2 * d[1])


def test_score_curve():
    assert score(1492.7) == pytest.approx(5000 / np.e)
    assert np.all(np.diff(score(np.array([0, 10, 100, 1000]))) < 0)


def test_summarize():
    m = summarize([0.5, 20, 150, 700, 2000, 10000])
    assert m["n"] == 6
    assert m["median_km"] == pytest.approx(425)
    assert m["within_1km"] == pytest.approx(1 / 6)
    assert m["within_2500km"] == pytest.approx(5 / 6)
    assert 0 < m["mean_score"] < 5000


def test_summarize_by_group():
    m = summarize_by_group([1, 2, 3], ["EU", "EU", "AS"])
    assert m["overall"]["n"] == 3 and m["EU"]["n"] == 2 and m["AS"]["median_km"] == 3
