"""Browser tests: the real game in headless Chromium, driven by the harness."""

import numpy as np
import pytest

from core.config import FIXTURES_DIR
from core.interfaces import Action
from core.scoring import haversine_km
from core.view_spec import VIEW_SIZE
from core.views import crop_view, load_pano
from env.encoders import FakeEncoder
from game.app import create_app
from game.server import serve_in_thread
from harness.checks import crop_vs_screenshot
from harness.harness import GameHarness, MapClickError

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def harness(fixture_images):
    app = create_app(FIXTURES_DIR, seed=1)
    with serve_in_thread(app) as url, GameHarness(url, headless=True, slow_mo_ms=0) as h:
        yield h


def test_observe_is_view_sized(harness):
    shot = harness.observe()
    assert shot.shape == (VIEW_SIZE, VIEW_SIZE, 3)
    assert shot.std() > 5  # actually rendered, not a blank canvas


def test_turn_right_rotates_45(harness):
    before = harness.game_state()["heading"]
    harness.act(Action.TURN_R)
    assert (harness.game_state()["heading"] - before) % 360 == pytest.approx(45)
    harness.act(Action.TURN_L)
    assert harness.game_state()["heading"] == pytest.approx(before)


@pytest.mark.parametrize("turns", [0, 1, 3])
def test_screenshot_matches_canonical_crop(harness, fixture_images, turns):
    """The M2 consistency check, on fixtures: Pannellum's render == core.views.crop_view."""
    for _ in range(turns):
        harness.act(Action.TURN_R)
    chk = crop_vs_screenshot(harness, FakeEncoder(), fixture_images, FIXTURES_DIR)
    assert chk["cosine"] > 0.95, chk
    assert chk["mean_abs_pixel_diff"] < 20, chk

    # Negative control: the crop one heading step off must match clearly worse, otherwise this
    # check could pass even with a wrong heading convention.
    row = fixture_images.set_index("image_id").loc[chk["image_id"]]
    pano = load_pano(FIXTURES_DIR / row["path"])
    off = crop_view(pano, chk["heading"] + 45, row["compass_angle"])
    mirrored = crop_view(pano, 2 * row["compass_angle"] - chk["heading"], row["compass_angle"])
    for wrong in (off, mirrored[:, ::-1]):
        diff = np.abs(chk["screenshot"].astype(int) - wrong.astype(int)).mean()
        assert diff > 3 * chk["mean_abs_pixel_diff"] + 5, (diff, chk["mean_abs_pixel_diff"])


def test_map_click_lands_near_target_and_submits(harness):
    target = (-33.87, 151.21)
    placed = harness.place_guess(*target)
    assert haversine_km(*target, placed["lat"], placed["lon"]) < 25
    result = harness.submit()
    assert result["distance_km"] == pytest.approx(
        haversine_km(placed["lat"], placed["lon"], result["true_lat"], result["true_lon"])
    )
    harness.next_round()
    assert harness.game_state()["phase"] == "looking"


def test_every_round_starts_on_the_world_map_and_clicks_land(harness):
    """Regression: the result zoom after a guess used to survive into the next round (Leaflet
    re-applies an animation's target when it ends), so later targets were off-screen and the
    click was silently clamped to the map edge, thousands of km off."""
    harness.reload_round()
    # Far-apart guess/truth pairs make the result fitBounds zoom and pan the most.
    targets = [(43.65, -79.39), (12.98, 77.6), (-36.85, 174.76), (64.15, -21.94)]
    for i, target in enumerate(targets):
        view = harness.page.evaluate(
            "() => ({z: window.__map.getZoom(), c: window.__map.getCenter()})"
        )
        assert view["z"] == 1 and abs(view["c"]["lng"]) < 1, f"round {i + 1} view {view}"
        placed = harness.place_guess(*target)
        assert haversine_km(*target, placed["lat"], placed["lon"]) < 25, (i, target, placed)
        harness.submit()
        harness.next_round()  # immediately, while the result animation is still running


def test_off_screen_guess_is_refused_not_clamped(harness):
    harness.reload_round()
    harness.page.evaluate("() => window.__map.setView([48.85, 2.35], 8, {animate: false})")
    with pytest.raises(MapClickError, match="off the visible map"):
        harness.place_guess(-33.87, 151.21)  # Sydney, while the map shows Paris
    harness.reload_round()


def test_five_round_game_shows_attribution_and_summary(harness):
    harness.reload_round()
    for round_number in range(5):
        harness.place_guess(0.0, 0.0)
        harness.submit()
        image_id = harness.game_state()["pano_url"].rsplit("/", 1)[1].removesuffix(".jpg")
        link = harness.page.locator("#attribution a")
        assert link.get_attribute("href") == f"https://www.mapillary.com/app/?pKey={image_id}"
        if round_number < 4:
            harness.next_round()

    harness.page.click("#next-round")
    harness.page.wait_for_function("() => window.__game.phase === 'finished'")
    assert harness.page.locator("#summary").is_visible()
    assert harness.game_state()["total_score"] >= 0
