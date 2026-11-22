import pytest

from core import game_api
from core.config import FIXTURES_DIR
from core.scoring import haversine_km, score
from core.view_spec import HEADINGS_DEG
from game.app import create_app


@pytest.fixture
def client(fixture_images):
    return create_app(FIXTURES_DIR, seed=0).test_client()


def test_new_round_contract(client, fixture_images):
    r = client.get(game_api.NEW_ROUND).get_json()
    assert set(r) == set(game_api.NewRoundResponse.__annotations__)
    assert r["heading"] in HEADINGS_DEG
    image_id = r["pano_url"].rsplit("/", 1)[1].removesuffix(".jpg")
    row = fixture_images.set_index("image_id").loc[image_id]
    assert r["north_offset"] == pytest.approx(row["compass_angle"])
    img = client.get(r["pano_url"])
    assert img.status_code == 200 and img.data[:2] == b"\xff\xd8"  # JPEG


def test_guess_scoring(client):
    r = client.get(game_api.NEW_ROUND).get_json()
    url = game_api.GUESS.format(round_id=r["round_id"])
    res = client.post(url, json={"lat": 10.0, "lon": 20.0}).get_json()
    assert set(res) == set(game_api.GuessResponse.__annotations__)
    d = haversine_km(10.0, 20.0, res["true_lat"], res["true_lon"])
    assert res["distance_km"] == pytest.approx(d) and res["score"] == pytest.approx(score(d))
    assert client.post(url, json={"lat": 0, "lon": 0}).status_code == 409


def test_bad_requests(client):
    assert (
        client.post(game_api.GUESS.format(round_id="nope"), json={"lat": 0, "lon": 0}).status_code
        == 404
    )
    r = client.get(game_api.NEW_ROUND).get_json()
    assert (
        client.post(game_api.GUESS.format(round_id=r["round_id"]), json={"lat": "x"}).status_code
        == 400
    )
    assert client.get("/panos/does-not-exist.jpg").status_code == 404


def test_page_has_contract_elements(client):
    html = client.get("/").get_data(as_text=True)
    for sel in (
        game_api.PANO,
        game_api.MAP,
        game_api.SUBMIT,
        game_api.NEXT_ROUND,
        game_api.TURN_LEFT,
        game_api.TURN_RIGHT,
    ):
        assert f'id="{sel[1:]}"' in html
    assert "Mapillary" in html and "CC BY-SA" in html  # ground rule 4
    assert 'id="bot-map"' in html
    assert 'id="bot-pano"' in html
    assert 'id="start-game"' in html
    assert 'id="player-score"' in html
    assert 'id="bot-score"' in html
    assert 'id="win-screen"' in html
    assert 'id="play-again"' in html
    assert 'id="name-screen"' in html
    assert 'id="player-name"' in html


def test_bot_guess_endpoint(fixture_images):
    class FixedGuesser:
        def cell_probs(self, view_embs):
            return view_embs

        def guess(self, probs):
            return (10.0, 20.0)

    app = create_app(FIXTURES_DIR, seed=0, bot_guesser=FixedGuesser())
    client = app.test_client()
    r = client.get(game_api.NEW_ROUND).get_json()
    res = client.get(game_api.BOT_GUESS.format(round_id=r["round_id"]))
    assert res.status_code == 200
    bot = res.get_json()
    assert bot["enabled"] is True
    assert set(bot) == {"enabled", "lat", "lon", "distance_km", "score", "true_lat", "true_lon"}


def test_random_bot_is_the_default_style(fixture_images):
    from models.guesser.random import RandomGuesser

    locations = fixture_images[["lat", "lon"]].to_numpy()
    app = create_app(
        FIXTURES_DIR,
        seed=0,
        bot_guesser=RandomGuesser(locations, seed=0),
    )
    client = app.test_client()
    r = client.get(game_api.NEW_ROUND).get_json()
    bot = client.get(game_api.BOT_GUESS.format(round_id=r["round_id"])).get_json()
    assert bot["enabled"] is True


def test_relative_dataset_path_serves_panoramas(monkeypatch):
    """create_app("tests/fixtures") from the repo root used to 404 every panorama: Flask resolved
    the relative path against the game package."""
    from core.config import REPO_ROOT

    monkeypatch.chdir(REPO_ROOT)
    client = create_app("tests/fixtures", seed=0).test_client()
    r = client.get(game_api.NEW_ROUND).get_json()
    assert client.get(r["pano_url"]).status_code == 200
