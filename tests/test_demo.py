"""Browser test: the human-vs-bot demo (?demo=1) plays every round, not just the first."""

import pytest

from core.config import FIXTURES_DIR
from data.dataset import load_images
from game.app import create_app
from game.server import serve_in_thread
from harness.harness import GameHarness
from models.guesser.random import RandomGuesser

pytestmark = pytest.mark.browser


def _wait_phase(page, phase, timeout_ms=15000):
    page.wait_for_function(f"window.__game.phase === '{phase}'", timeout=timeout_ms)


def _play_round(page):
    _wait_phase(page, "searching")  # round open: map clicks and Guess work, bot is searching
    total_before = page.evaluate("window.__game.total_score")
    page.click("#map", position={"x": 200, "y": 150})
    page.click("#submit")
    # The player locked in before the bot finished turning: nothing about the answer is shown yet.
    page.wait_for_function("document.querySelector('#result').textContent === 'Guess locked in'")
    assert not page.evaluate("window.__game.bot_done")
    assert page.evaluate("window.__game.total_score") == total_before
    page.wait_for_selector("#next-round:not([hidden])", timeout=20000)
    # Both locked in: both answers and scores are revealed together.
    assert " km · " in page.inner_text("#result")
    assert " km · " in page.inner_text("#bot-result")


def test_demo_plays_all_rounds_and_again(fixture_images):
    bot = RandomGuesser(load_images(FIXTURES_DIR)[["lat", "lon"]].to_numpy(), seed=0)
    app = create_app(FIXTURES_DIR, seed=0, bot_guesser=bot)
    with serve_in_thread(app) as url, GameHarness(url, headless=True, slow_mo_ms=0) as h:
        page = h.page
        page.goto(url + "/?demo=1")
        page.click("#start-game")
        page.fill("#player-name", "Tester")
        page.press("#player-name", "Enter")
        for _ in range(5):
            _play_round(page)
            page.click("#next-round")
        page.wait_for_selector("#win-screen:not([hidden])", timeout=5000)
        assert page.evaluate("window.__game.bot_rounds.length") == 5

        page.click("#play-again")  # a second game starts fresh and also gets past round 1
        page.fill("#player-name", "Again")
        page.press("#player-name", "Enter")
        _play_round(page)
        page.click("#next-round")
        _play_round(page)
        assert page.evaluate("window.__game.round_number") == 2
