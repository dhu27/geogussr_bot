import numpy as np
import pytest

from core.interfaces import Action, Encoder, Guesser, Policy
from core.navigation import select_forward_neighbor
from core.scoring import haversine_km
from env.encoders import FakeEncoder
from env.lookaround_env import LookAroundEnv
from models.guesser.classifier import CellGuesser
from models.guesser.combine import combine_views, expected_score_guess, expected_score_matrix
from models.guesser.geocells import fit_geocells
from models.policy.baselines import LookAroundPolicy, RandomPolicy


def test_protocols():
    assert isinstance(FakeEncoder(), Encoder)
    assert isinstance(LookAroundPolicy(), Policy) and isinstance(RandomPolicy(), Policy)
    g = CellGuesser(np.array([[0.0, 0.0], [10.0, 10.0]]), np.eye(2, 768), np.array([0.5, 0.5]))
    assert isinstance(g, Guesser)


def test_fake_encoder_is_deterministic_and_normalised():
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (336, 336, 3), dtype=np.uint8) for _ in range(3)]
    a, b = FakeEncoder().embed(imgs), FakeEncoder().embed(imgs)
    assert a.shape == (3, 768) and np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)


def test_combine_single_view_is_identity():
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(combine_views(p, np.full(3, 1 / 3)), p[0])


def test_combine_removes_repeated_prior():
    # Two views that each just repeat the prior must give back the prior, not prior^2.
    prior = np.array([0.6, 0.3, 0.1])
    assert np.allclose(combine_views(np.stack([prior, prior]), prior), prior)


def test_combine_sharpens_agreeing_views():
    p = np.array([0.5, 0.3, 0.2])
    out = combine_views(np.stack([p, p, p]), np.full(3, 1 / 3))
    assert out[0] > p[0] and out.sum() == pytest.approx(1)


def test_expected_score_picks_centre_of_spread_belief():
    # Four nearby cells at the corners of a square, equal belief: the centre beats any corner.
    cells = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    cands = np.concatenate([cells, [[1.0, 1.0]]])
    (lat, lon), _ = expected_score_guess(
        np.full(4, 0.25), cands, expected_score_matrix(cands, cells)
    )
    assert (lat, lon) == (1.0, 1.0)


def test_geocells_and_guesser_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    lat = np.concatenate([rng.normal(48, 1, 50), rng.normal(-33, 1, 50)])
    lon = np.concatenate([rng.normal(2, 1, 50), rng.normal(151, 1, 50)])
    centroids, labels = fit_geocells(lat, lon, k=2)
    assert centroids.shape == (2, 2) and labels[0] != labels[99]
    assert labels[:50].tolist() == [labels[0]] * 50 and labels[50:].tolist() == [labels[50]] * 50

    emb = np.zeros((100, 768), np.float32)
    emb[:50, 0], emb[50:, 1] = 1, 1
    g = CellGuesser.fit(emb, lat, lon, n_cells=2)
    probs = g.cell_probs(emb[:3])
    assert probs.sum() == pytest.approx(1)
    assert haversine_km(*g.guess(probs), 48, 2) < 300
    g.save(tmp_path / "g.npz")
    g2 = CellGuesser.load(tmp_path / "g.npz")
    assert np.allclose(g2.cell_probs(emb[60:62]), g.cell_probs(emb[60:62]))


def test_select_forward_neighbor():
    nbrs = [("a", 10.0), ("b", 350.0), ("c", 180.0)]
    assert select_forward_neighbor(355.0, nbrs) == "b"
    assert select_forward_neighbor(0.0, nbrs) == "a"  # both 10 degrees off: tie -> smallest id
    assert select_forward_neighbor(90.0, nbrs) is None
    assert select_forward_neighbor(185.0, nbrs) == "c"


def test_lookaround_env_on_fixtures(fixture_images, fixture_views):
    views, emb = fixture_views
    tr = views.merge(fixture_images, on="image_id")
    g = CellGuesser.fit(
        emb[tr["row"]].astype(np.float32), tr["lat"].to_numpy(), tr["lon"].to_numpy(), n_cells=4
    )
    env = LookAroundEnv(fixture_images, views, emb, g)
    policy = LookAroundPolicy(8)
    obs, info = env.reset(seed=0)
    steps = 0
    while True:
        obs, reward, term, trunc, info = env.step(policy.act(obs))
        steps += 1
        if term or trunc:
            break
    assert term and steps == 8 and info["n_views"] == 8
    assert 0 <= info["score"] <= 5000 and reward == info["score"]
    assert obs.view_embs.shape == (8, 768)
    assert sorted(obs.rel_headings_deg.tolist()) == [45.0 * i for i in range(8)]


def test_random_policy_terminates():
    p = RandomPolicy(p_guess=0.0, max_steps=5)
    from core.interfaces import Observation

    obs = Observation(np.zeros((1, 768)), np.zeros(1), step=5)
    assert p.act(obs) == Action.GUESS
