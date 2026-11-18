import json

import numpy as np
import pytest

from models.guesser import registry
from models.guesser.classifier import CellGuesser


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "model_dir", lambda name: tmp_path / name)
    return tmp_path


def _cell_guesser():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 8)).astype(np.float32)
    lat, lon = rng.uniform(-60, 60, 60), rng.uniform(-170, 170, 60)
    return CellGuesser.fit(emb, lat, lon, n_cells=4, epochs=5), emb


def test_save_and_load_round_trip(models_dir):
    g, emb = _cell_guesser()
    out = registry.save_model(g, name="tiny", dataset="fixtures", description="d", author="me")
    info = json.loads((out / "model.json").read_text())
    assert info["kind"] == "softmax" and info["dataset"] == "fixtures" and info["author"] == "me"
    back = registry.load_model("tiny")
    assert isinstance(back, CellGuesser)
    np.testing.assert_allclose(back.cell_probs(emb[:3]), g.cell_probs(emb[:3]))


def test_new_kind_is_found_by_its_decorator(models_dir):
    @registry.register("constant_test")
    class Constant:
        def cell_probs(self, view_embs):
            return np.ones(1)

        def guess(self, probs):
            return (1.0, 2.0)

        def save_to(self, folder):
            (folder / "nothing").write_text("")

        @classmethod
        def load_from(cls, folder):
            return cls()

    registry.save_model(Constant(), name="const", dataset="x", author="")
    assert registry.load_model("const").guess(None) == (1.0, 2.0)
    registry.REGISTRY.pop("constant_test")


def test_missing_model_says_how_to_pull(models_dir):
    with pytest.raises(FileNotFoundError, match="pull-model nope"):
        registry.load_model("nope")


def test_teammate_subclass_of_the_geocell_base_loads(models_dir):
    from models.guesser.classifier import _GeocellGuesser, _softmax

    @registry.register("subclass_test")
    class Tiny(_GeocellGuesser):
        def __init__(self, centroids, prior, w):
            super().__init__(centroids, prior)
            self.w = w

        def _per_view_probs(self, view_embs):
            return _softmax(np.atleast_2d(view_embs) @ self.w)

        def _arrays(self):
            return {"w": self.w}

        @classmethod
        def _from_arrays(cls, a):
            return cls(a["centroids"], a["prior"], a["w"])

    g = Tiny(np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([0.5, 0.5]), np.eye(2))
    registry.save_model(g, name="sub", dataset="x", author="")
    back = registry.load_model("sub")
    assert type(back) is Tiny
    np.testing.assert_allclose(back.cell_probs(np.array([[1.0, 0.0]])), g.cell_probs([[1.0, 0.0]]))
    registry.REGISTRY.pop("subclass_test")
