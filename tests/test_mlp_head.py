import numpy as np
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.ml

from models.guesser import registry  # noqa: E402
from models.guesser.mlp_head import MLPGuesser  # noqa: E402


def _toy(n_per=200, d=16, seed=0):
    """Four far-apart places, each with its own embedding direction: learnable by any head."""
    rng = np.random.default_rng(seed)
    places = np.array([[48.0, 2.0], [-33.0, 151.0], [40.0, -74.0], [-23.0, -46.0]])
    dirs = rng.normal(size=(4, d))
    x = np.concatenate([dirs[i] + 0.3 * rng.normal(size=(n_per, d)) for i in range(4)])
    latlon = np.repeat(places, n_per, axis=0) + rng.normal(scale=0.5, size=(4 * n_per, 2))
    return x.astype(np.float32), latlon[:, 0], latlon[:, 1], dirs, places


def test_learns_and_numpy_inference_matches_training():
    x, lat, lon, dirs, places = _toy()
    g = MLPGuesser.fit(x, lat, lon, n_cells=4, hidden=32, epochs=30, lr=1e-2, log=None)
    for i in range(4):
        p = g.cell_probs(dirs[i : i + 1].astype(np.float32))
        assert p.sum() == pytest.approx(1.0, abs=1e-5)
        guess = g.guess(p)
        assert abs(guess[0] - places[i, 0]) < 5 and abs(guess[1] - places[i, 1]) < 5


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "model_dir", lambda name: tmp_path / name)
    x, lat, lon, dirs, _ = _toy()
    g = MLPGuesser.fit(x, lat, lon, n_cells=4, hidden=8, epochs=2, log=None)
    registry.save_model(g, name="mlp_test", dataset="toy", author="")
    back = registry.load_model("mlp_test")
    assert isinstance(back, MLPGuesser)
    np.testing.assert_allclose(back.cell_probs(dirs), g.cell_probs(dirs), rtol=1e-6)
