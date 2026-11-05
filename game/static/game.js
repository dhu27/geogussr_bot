// Game frontend. Element ids and API paths mirror core/game_api.py.
// window.__game is for logging/synchronisation only; the harness never feeds it to a model.
(function () {
  "use strict";
  const STEP = 45; // core/view_spec.py HEADING_STEP_DEG
  const HFOV = 90; // core/view_spec.py HFOV_DEG
  const WORLD = { center: [20, 0], zoom: 1 };

  const $ = (sel) => document.querySelector(sel);
  const wrap = (a) => ((a % 360) + 360) % 360;
  const state = (window.__game = {
    round_id: null, heading: null, phase: "loading", view_ready: false, guess: null, result: null,
  });
  let round = null;
  let viewer = null;

  // ---- map ---------------------------------------------------------------------------
  const map = L.map("map", { worldCopyJump: true, zoomControl: true }).setView(WORLD.center, WORLD.zoom);
  window.__map = map; // harness uses map.latLngToContainerPoint to place clicks
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18, attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  const layer = L.layerGroup().addTo(map);
  let guessMarker = null;

  map.on("click", (e) => {
    if (state.phase !== "looking") return;
    const ll = e.latlng.wrap();
    state.guess = { lat: ll.lat, lon: ll.lng };
    if (guessMarker) guessMarker.setLatLng(e.latlng);
    else guessMarker = L.circleMarker(e.latlng, { radius: 7, color: "#2563eb", weight: 3 }).addTo(layer);
    $("#submit").disabled = false;
  });

  // ---- panorama ----------------------------------------------------------------------
  function markReady() {
    // Two frames so the WebGL canvas has painted the new yaw before the harness screenshots.
    requestAnimationFrame(() => requestAnimationFrame(() => { state.view_ready = true; }));
  }

  function showHeading() {
    $("#heading-label").textContent = `${Math.round(state.heading)}°`;
  }

  function setHeading(heading, animate) {
    state.view_ready = false;
    state.heading = wrap(heading);
    showHeading();
    const yaw = ((state.heading - round.north_offset + 540) % 360) - 180; // heading -> pano yaw
    if (animate) viewer.setYaw(yaw, 250, markReady);
    else { viewer.setYaw(yaw, false); markReady(); }
  }

  function turn(dir) {
    if (!viewer || state.phase !== "looking") return;
    const snapped = Math.round(state.heading / STEP) * STEP;
    setHeading(snapped + dir * STEP, true);
  }

  function loadPano() {
    if (viewer) viewer.destroy();
    const yaw = ((round.heading - round.north_offset + 540) % 360) - 180;
    viewer = pannellum.viewer("pano", {
      type: "equirectangular",
      panorama: round.pano_url,
      autoLoad: true,
      yaw: yaw, pitch: 0, hfov: HFOV, minHfov: HFOV, maxHfov: HFOV,
      northOffset: round.north_offset,
      showControls: false, compass: false, showZoomCtrl: false, showFullscreenCtrl: false,
      mouseZoom: false, keyboardZoom: false, disableKeyboardCtrl: true, doubleClickZoom: false,
    });
    viewer.on("load", () => {
      state.heading = wrap(round.heading);
      showHeading();
      state.phase = "looking";
      markReady();
    });
    // Human drags: keep the logged heading in sync.
    viewer.on("mouseup", () => {
      state.heading = wrap(viewer.getYaw() + round.north_offset);
      showHeading();
    });
    viewer.on("error", (msg) => { state.phase = "error"; state.error = String(msg); });
  }

  // ---- rounds ------------------------------------------------------------------------
  async function newRound() {
    Object.assign(state, { phase: "loading", view_ready: false, guess: null, result: null });
    layer.clearLayers();
    guessMarker = null;
    map.setView(WORLD.center, WORLD.zoom, { animate: false });
    $("#submit").disabled = true;
    $("#next-round").hidden = true;
    $("#result").textContent = "";
    round = await (await fetch("/api/round/new")).json();
    state.round_id = round.round_id;
    state.pano_url = round.pano_url; // logging + consistency checks only
    loadPano();
  }

  async function submit() {
    if (state.phase !== "looking" || !state.guess) return;
    state.phase = "submitting";
    $("#submit").disabled = true;
    const res = await fetch(`/api/round/${round.round_id}/guess`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.guess),
    });
    const r = await res.json();
    const truth = [r.true_lat, r.true_lon];
    L.circleMarker(truth, { radius: 7, color: "#16a34a", weight: 3 }).addTo(layer);
    L.polyline([[state.guess.lat, state.guess.lon], truth], { color: "#6b6b66", dashArray: "4 6" }).addTo(layer);
    map.fitBounds(L.latLngBounds([[state.guess.lat, state.guess.lon], truth]).pad(0.3), { maxZoom: 8 });
    $("#result").textContent = `${r.distance_km.toFixed(1)} km · ${Math.round(r.score)} pts`;
    $("#next-round").hidden = false;
    state.result = r;
    state.phase = "guessed";
  }

  $("#turn-left").addEventListener("click", () => turn(-1));
  $("#turn-right").addEventListener("click", () => turn(1));
  $("#submit").addEventListener("click", submit);
  $("#next-round").addEventListener("click", newRound);
  document.addEventListener("keydown", (e) => {
    if (e.key === "q") turn(-1);
    else if (e.key === "e") turn(1);
  });

  newRound();
})();
