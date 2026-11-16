"""The trained geocell guesser (M1 task 3): training, persistence, offline evaluation."""

import numpy as np
import pytest

from core.scoring import haversine_km
from models.guesser.classifier import CellGuesser, NearestMeanGuesser, train_softmax
from models.guesser.evaluation import LargestCellGuesser, predict, summarize, view_matrix

CITIES = {"paris": (48.86, 2.35), "sydney": (-33.87, 151.21), "lima": (-12.05, -77.04)}


def three_city_data(rng, per_city=60, d=32, noise=0.6):
    """Views whose embedding direction depends on the city, plus noise."""
    centres = rng.normal(size=(len(CITIES), d))
    x, lat, lon, city = [], [], [], []
    for c, (name, (la, lo)) in enumerate(CITIES.items()):
        x.append(centres[c] + noise * rng.normal(size=(per_city, d)))
        lat += [la + rng.normal(0, 0.05) for _ in range(per_city)]
        lon += [lo + rng.normal(0, 0.05) for _ in range(per_city)]
        city += [name] * per_city
    x = np.concatenate(x).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.array(lat), np.array(lon), np.array(city)


def test_softmax_training_fits_separable_classes():
    rng = np.random.default_rng(0)
    x, _, _, city = three_city_data(rng)
    y = np.unique(city, return_inverse=True)[1]
    w, b = train_softmax((x - x.mean(0)) / x.std(0), y, 3, l2=1e-3, epochs=200, lr=0.05)
    acc = (((x - x.mean(0)) / x.std(0) @ w + b).argmax(1) == y).mean()
    assert acc > 0.98


def test_cell_guesser_finds_the_right_city_on_held_out_views():
    rng = np.random.default_rng(1)
    x, lat, lon, city = three_city_data(rng)
    train = np.arange(len(x)) % 4 != 0
    g = CellGuesser.fit(x[train], lat[train], lon[train], n_cells=3, epochs=200, lr=0.05)
    hits = 0
    for i in np.flatnonzero(~train):
        guess = g.guess(g.cell_probs(x[i : i + 1]))
        hits += haversine_km(*guess, lat[i], lon[i]) < 100
    assert hits / (~train).sum() > 0.9


def test_cell_probs_are_a_distribution_and_combine_views():
    rng = np.random.default_rng(2)
    x, lat, lon, _ = three_city_data(rng)
    g = CellGuesser.fit(x, lat, lon, n_cells=3, epochs=100)
    for views in (x[:1], x[:8]):
        p = g.cell_probs(views)
        assert p.shape == (g.n_cells,) and np.all(p >= 0) and p.sum() == pytest.approx(1)


@pytest.mark.parametrize("kind", ["softmax", "nearest_mean"])
def test_save_load_roundtrip_keeps_predictions(tmp_path, kind):
    rng = np.random.default_rng(3)
    x, lat, lon, _ = three_city_data(rng)
    cls = CellGuesser if kind == "softmax" else NearestMeanGuesser
    g = cls.fit(x, lat, lon, n_cells=3)
    g.save(tmp_path / "g.npz")
    loaded = CellGuesser.load(tmp_path / "g.npz")  # play.py always calls CellGuesser.load
    assert type(loaded) is cls
    assert np.allclose(loaded.cell_probs(x[:5]), g.cell_probs(x[:5]), atol=1e-6)


def test_loads_a_guesser_saved_before_m1(tmp_path):
    """Pre-M1 files have no `kind` key and hold the nearest-mean placeholder."""
    centroids, means = np.array([[0.0, 0.0], [10.0, 10.0]]), np.eye(2, 768)
    np.savez(
        tmp_path / "old.npz",
        centroids=centroids,
        cell_means=means,
        prior=np.array([0.5, 0.5]),
        temperature=20.0,
    )
    assert isinstance(CellGuesser.load(tmp_path / "old.npz"), NearestMeanGuesser)


def test_predict_and_summarize_on_fixtures(fixture_images, fixture_views):
    views, emb = fixture_views
    train = fixture_images[fixture_images["split"] == "train"]
    x = view_matrix(views, emb, train["image_id"].tolist())
    n, h, d = x.shape
    g = CellGuesser.fit(
        x.reshape(n * h, d),
        np.repeat(train["lat"], h),
        np.repeat(train["lon"], h),
        n_cells=4,
        epochs=50,
    )
    pred = predict(g, fixture_images, views, emb, "val")
    n_val = (fixture_images["split"] == "val").sum()
    assert len(pred) == n_val * 9  # 8 single-view guesses + 1 look-around guess per panorama
    assert set(pred["n_views"]) == {1, 8}
    m = summarize(pred)
    assert m["1_view"]["overall"]["n"] == n_val * 8 and m["8_views"]["overall"]["n"] == n_val
    assert "test" not in set(
        fixture_images.loc[fixture_images["image_id"].isin(pred["image_id"]), "split"]
    )


def test_largest_cell_baseline_is_constant():
    g = LargestCellGuesser(48.0, 2.0)
    assert g.guess(g.cell_probs(np.zeros((8, 768)))) == (48.0, 2.0)
