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
