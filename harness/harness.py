"""Playwright harness: screenshots in, clicks out (PLAN.md 7.5).

Ground rule 1: ``observe()`` is the only thing that reaches the model. ``window.__game`` is read
only to wait for the page and to log results.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

from core import game_api
from core.interfaces import Action
from core.view_spec import VIEW_SIZE

# Injected on every page: a dot that follows the agent's mouse so viewers can see it act.
_CURSOR_JS = """
(() => {
  const add = () => {
    const d = document.createElement('div');
    d.id = '__agent_cursor';
    Object.assign(d.style, {position: 'fixed', width: '18px', height: '18px', borderRadius: '50%',
      background: 'rgba(239,68,68,0.55)', border: '2px solid #ef4444', pointerEvents: 'none',
      zIndex: 2147483647, transform: 'translate(-50%,-50%)', left: '-40px', top: '-40px',
      transition: 'left 60ms linear, top 60ms linear'});
    document.body.appendChild(d);
    document.addEventListener('mousemove', e => {
      d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px';
    }, true);
    document.addEventListener('mousedown', () => d.style.background = 'rgba(239,68,68,0.95)', true);
    document.addEventListener('mouseup', () => d.style.background = 'rgba(239,68,68,0.55)', true);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""


class GameHarness:
    def __init__(
        self,
        base_url: str,
        *,
        headless: bool = False,
        slow_mo_ms: int = 300,
        cursor_overlay: bool = True,
        record_video_dir: str | Path | None = None,
        trace_path: str | Path | None = None,
        viewport: dict | None = None,
        map_zoom_steps: int = 4,
        timeout_ms: int = 30_000,
    ):
        self.base_url = base_url.rstrip("/")
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self.cursor_overlay = cursor_overlay
        self.record_video_dir = record_video_dir
        self.trace_path = trace_path
        self.viewport = viewport or {"width": 1280, "height": 800}
        self.map_zoom_steps = map_zoom_steps
        self.timeout_ms = timeout_ms
        self.page: Page | None = None

    # --- lifecycle --------------------------------------------------------------------
    def __enter__(self) -> GameHarness:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo_ms,
            # Pannellum needs WebGL; headless/CI machines fall back to SwiftShader.
            args=[
                "--enable-unsafe-swiftshader",
                "--use-angle=swiftshader",
                "--ignore-gpu-blocklist",
            ],
        )
        ctx_kwargs: dict = {"viewport": self.viewport, "device_scale_factor": 1}
        if self.record_video_dir:
            ctx_kwargs["record_video_dir"] = str(self.record_video_dir)
            ctx_kwargs["record_video_size"] = self.viewport
        self._ctx = self._browser.new_context(**ctx_kwargs)
        self._ctx.set_default_timeout(self.timeout_ms)
        if self.trace_path:
            self._ctx.tracing.start(screenshots=True, snapshots=True)
        if self.cursor_overlay:
            self._ctx.add_init_script(_CURSOR_JS)
        self.page = self._ctx.new_page()
        self.page.goto(self.base_url + "/")
        self._wait_view_ready()
        return self

    def __exit__(self, *exc) -> None:
        if self.trace_path:
            Path(self.trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._ctx.tracing.stop(path=str(self.trace_path))
        self._ctx.close()  # flushes the video
        self._browser.close()
        self._pw.stop()

    @property
    def video_path(self) -> str | None:
        return self.page.video.path() if self.page and self.page.video else None

    # --- page state (logging / sync only) ---------------------------------------------
    def game_state(self) -> dict:
        return self.page.evaluate(f"() => JSON.parse(JSON.stringify({game_api.GAME_STATE_JS}))")

    def _wait_view_ready(self) -> None:
        self.page.wait_for_function(
            f"() => {game_api.GAME_STATE_JS} && {game_api.GAME_STATE_JS}.phase === 'looking' "
            f"&& {game_api.GAME_STATE_JS}.view_ready === true"
        )

    # --- observe ----------------------------------------------------------------------
    def observe(self) -> np.ndarray:
        """Screenshot of ``#pano`` only, as (VIEW_SIZE, VIEW_SIZE, 3) uint8 RGB."""
        self._wait_view_ready()
        self._set_cursor_visible(False)
        try:
            png = self.page.locator(game_api.PANO).screenshot(animations="disabled")
        finally:
            self._set_cursor_visible(True)
        im = Image.open(io.BytesIO(png)).convert("RGB")
        if im.size != (VIEW_SIZE, VIEW_SIZE):
            im = im.resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
        return np.asarray(im)

    def _set_cursor_visible(self, visible: bool) -> None:
        if self.cursor_overlay:
            self.page.evaluate(
                "v => { const d = document.getElementById('__agent_cursor'); "
                "if (d) d.style.visibility = v ? 'visible' : 'hidden'; }",
                visible,
            )

    # --- act --------------------------------------------------------------------------
    def act(self, action: Action) -> None:
        if action == Action.TURN_L:
            self.page.click(game_api.TURN_LEFT)
        elif action == Action.TURN_R:
            self.page.click(game_api.TURN_RIGHT)
        else:
            raise NotImplementedError(f"{action!r} (FORWARD/BACK arrive in Phase 2)")
        self._wait_view_ready()

    def _map_point(self, lat: float, lon: float) -> tuple[float, float]:
        """Page pixel of (lat, lon) using Leaflet's own projection."""
        box = self.page.locator(game_api.MAP).bounding_box()
        pt = self.page.evaluate(
            "([lat, lon]) => { const p = window.__map.latLngToContainerPoint([lat, lon]); "
            "return [p.x, p.y]; }",
            [lat, lon],
        )
        x = min(max(pt[0], 2), box["width"] - 2)
        y = min(max(pt[1], 2), box["height"] - 2)
        return box["x"] + x, box["y"] + y

    def place_guess(self, lat: float, lon: float) -> dict:
        """Zoom in around the target (double-click keeps the point under the cursor), then click.
        Returns the guess the page actually recorded, for logging pixel-quantisation error."""
        for _ in range(self.map_zoom_steps):
            x, y = self._map_point(lat, lon)
            self.page.mouse.dblclick(x, y)
            self.page.wait_for_timeout(350)  # Leaflet zoom animation
        x, y = self._map_point(lat, lon)
        self.page.mouse.click(x, y)
        self.page.wait_for_function(f"() => {game_api.GAME_STATE_JS}.guess !== null")
        return self.game_state()["guess"]

    def submit(self) -> dict:
        self.page.click(game_api.SUBMIT)
        self.page.wait_for_function(f"() => {game_api.GAME_STATE_JS}.phase === 'guessed'")
        return self.game_state()["result"]

    def next_round(self) -> None:
        self.page.click(game_api.NEXT_ROUND)
        self._wait_view_ready()

    def reload_round(self) -> None:
        """Recover from a failed round by asking the page to start a fresh round."""
        self.page.reload(wait_until="domcontentloaded")
        self._wait_view_ready()


def wait_for_server(url: str, timeout_s: float = 15.0) -> None:
    import requests

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if requests.get(url, timeout=1).ok:
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"server at {url} did not come up")
        time.sleep(0.1)
