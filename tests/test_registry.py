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
