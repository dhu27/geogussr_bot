import importlib.util
from pathlib import Path

import pandas as pd

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
