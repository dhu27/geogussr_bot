import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).parents[1] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def _pred(rows):
    return pd.DataFrame(rows, columns=["region_id", "image_id", "n_views", "distance_km", "score"])


def test_gain_is_relative_drop_in_median():
    p = _pred(
        [("a", "1", 1, 1000, 1000), ("a", "1", 8, 500, 2000), ("b", "2", 1, 1000, 1000),
         ("b", "2", 8, 500, 2000)]
    )  # fmt: skip
    g = evaluate.gain(p)
    assert g["median_reduction"] == 0.5
    assert g["mean_score_8"] - g["mean_score_1"] == 1000


def test_bootstrap_resamples_regions():
    # One region gains a lot, the other not at all: the interval must span both outcomes.
    p = _pred(
        [("a", "1", 1, 1000, 0), ("a", "1", 8, 100, 0), ("b", "2", 1, 1000, 0),
         ("b", "2", 8, 1000, 0)]
    )  # fmt: skip
    ci = evaluate.bootstrap(p, 200, seed=0)
    lo, hi = ci["median_reduction"]
    assert lo == 0.0 and hi == 0.9


def test_paired_compares_the_same_guesses():
    # Model is 100 km better than the baseline on every guess, in two regions.
    rows, base = [], []
    for region, img in (("a", "1"), ("b", "2")):
        for n_views, view in [(1, h) for h in range(8)] + [(8, -1)]:
            rows.append((region, img, n_views, view, 400.0, 3000.0))
            base.append((region, img, n_views, view, 500.0, 2900.0))
    cols = ["region_id", "image_id", "n_views", "view", "distance_km", "score"]
    p = evaluate.paired(pd.DataFrame(rows, columns=cols), pd.DataFrame(base, columns=cols), 50, 0)
    for key in ("1_view", "8_views"):
        assert p[key]["median_km_change"] == pytest.approx(-0.2)
        assert p[key]["mean_score_diff"] == pytest.approx(100)
        assert p[key]["mean_score_diff_ci95"] == pytest.approx([100, 100])
        assert p[key]["share_closer"] == 1.0
