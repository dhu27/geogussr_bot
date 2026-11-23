// Game frontend. Element ids and API paths mirror core/game_api.py.
(function () {
  "use strict";
  const STEP = 45;
  const HFOV = 90;
  const WORLD = { center: [20, 0], zoom: 1 };
  const TOTAL_ROUNDS = 5;
  const GRACE_SECONDS = 30;
  const BOT_TURN_MS = 650;
  const LEADERBOARD_KEY = "geogussr_bot_leaderboard";
  const DEMO = new URLSearchParams(window.location.search).get("demo") === "1";
  document.documentElement.classList.toggle("demo", DEMO);
  const $ = (sel) => document.querySelector(sel);
  const wrap = (a) => ((a % 360) + 360) % 360;
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const state = (window.__game = {
    round_id: null, heading: null, phase: "loading", view_ready: false, guess: null, result: null,
    round_number: 0, total_rounds: TOTAL_ROUNDS, total_score: 0, bot_result: null,
    bot_total_score: 0, bot_done: false, human_done: false,
    player_rounds: [], bot_rounds: [],
    player_name: "Player",
  });
  let round = null;
  let viewer = null;
  let botViewer = null;
  let botRequest = null;
  let botHeading = 0;
  let graceTimer = null;
  let graceDeadline = 0;
  let graceOwner = null;

  const map = L.map("map", { worldCopyJump: true, zoomControl: true }).setView(WORLD.center, WORLD.zoom);
  window.__map = map;
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18, attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  const layer = L.layerGroup().addTo(map);
  const botMap = L.map("bot-map", { worldCopyJump: true, zoomControl: true }).setView(WORLD.center, WORLD.zoom);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18, attribution: "© OpenStreetMap contributors",
  }).addTo(botMap);
  const botLayer = L.layerGroup().addTo(botMap);
  let guessMarker = null;
  let botGuessMarker = null;

  map.on("click", (e) => {
    if (!["looking", "searching"].includes(state.phase)) return;
    const ll = e.latlng.wrap();
    state.guess = { lat: ll.lat, lon: ll.lng };
    if (guessMarker) guessMarker.setLatLng(e.latlng);
    else guessMarker = L.circleMarker(e.latlng, { radius: 7, color: "#2563eb", weight: 3 }).addTo(layer);
    $("#submit").disabled = false;
  });

  function markReady() {
    requestAnimationFrame(() => requestAnimationFrame(() => { state.view_ready = true; }));
  }

  function showHeading() {
    $("#heading-label").textContent = `${Math.round(state.heading)}°`;
  }

  function showBotHeading() {
    $("#bot-heading-label").textContent = `Looking at ${Math.round(botHeading)}°`;
  }

  function showRoundProgress() {
    $("#round-label").textContent = `Round ${state.round_number} of ${TOTAL_ROUNDS}`;
    $("#player-score").textContent = Math.round(state.total_score).toLocaleString();
    $("#bot-score").textContent = Math.round(state.bot_total_score).toLocaleString();
  }

  function setHeading(heading, animate) {
    state.view_ready = false;
    state.heading = wrap(heading);
    showHeading();
    const yaw = ((state.heading - round.north_offset + 540) % 360) - 180;
    if (animate) viewer.setYaw(yaw, 250, markReady);
    else { viewer.setYaw(yaw, false); markReady(); }
  }

  function turn(dir) {
    if (!viewer || !["looking", "searching"].includes(state.phase)) return;
    const snapped = Math.round(state.heading / STEP) * STEP;
    setHeading(snapped + dir * STEP, true);
  }

  function loadPano() {
    if (viewer) viewer.destroy();
    const yaw = ((round.heading - round.north_offset + 540) % 360) - 180;
    viewer = pannellum.viewer("pano", {
      type: "equirectangular", panorama: round.pano_url, autoLoad: true,
      yaw, pitch: 0, hfov: HFOV, minHfov: HFOV, maxHfov: HFOV,
      northOffset: round.north_offset,
      showControls: false, compass: false, showZoomCtrl: false, showFullscreenCtrl: false,
      mouseZoom: false, keyboardZoom: false, disableKeyboardCtrl: true, doubleClickZoom: false,
    });
    viewer.on("load", () => {
      state.heading = wrap(round.heading);
      showHeading();
      if (!DEMO) state.phase = "looking";
      markReady();
    });
    viewer.on("mouseup", () => {
      state.heading = wrap(viewer.getYaw() + round.north_offset);
      showHeading();
    });
  }

  function loadBotPano() {
    if (botViewer) botViewer.destroy();
    botHeading = wrap(round.heading);
    const yaw = ((botHeading - round.north_offset + 540) % 360) - 180;
    botViewer = pannellum.viewer("bot-pano", {
      type: "equirectangular", panorama: round.pano_url, autoLoad: true,
      yaw, pitch: 0, hfov: HFOV, minHfov: HFOV, maxHfov: HFOV,
      northOffset: round.north_offset,
      showControls: false, compass: false, showZoomCtrl: false, showFullscreenCtrl: false,
      mouseZoom: false, keyboardZoom: false, disableKeyboardCtrl: true, doubleClickZoom: false,
    });
    botViewer.on("load", () => showBotHeading());
  }

  async function mapIdle() {
    map.stop();
    botMap.stop();
    if (map._animatingZoom) await new Promise((resolve) => map.once("zoomend", resolve));
  }

  function clearGraceTimer() {
    if (graceTimer) clearInterval(graceTimer);
    graceTimer = null;
    graceOwner = null;
    graceDeadline = 0;
    $("#grace-timer").textContent = "";
  }

  function startGrace(owner) {
    if (graceTimer) clearGraceTimer();
    graceOwner = owner;
    graceDeadline = Date.now() + GRACE_SECONDS * 1000;
    $("#grace-timer").textContent =
      owner === "human" ? "You have 30 seconds" : "Bot has 30 seconds";
    graceTimer = setInterval(() => {
      const remaining = Math.max(0, Math.ceil((graceDeadline - Date.now()) / 1000));
      $("#grace-timer").textContent =
        graceOwner === "human"
          ? `You have ${remaining} seconds`
          : `Bot has ${remaining} seconds`;
      if (remaining === 0) {
        const owner = graceOwner;
        clearGraceTimer();
        if (owner === "human") submitHuman({ lat: 0, lon: 0 }, true);
        else finishBot(true);
      }
    }, 250);
  }

  async function finishBot(forced = false) {
    if (state.bot_done) return;
    state.bot_done = true;
    if (graceTimer) clearGraceTimer();
    $("#bot-heading-label").textContent = forced ? "Bot guess locked" : "Bot finished";
    try {
      if (!botRequest) {
        botRequest = fetch(`/api/round/${round.round_id}/bot`).then((res) => res.json());
      }
      state.bot_result = await botRequest;
      if (!state.bot_result.enabled) {
        $("#bot-result").textContent = "No bot model loaded";
      } else {
        botGuessMarker = L.circleMarker([state.bot_result.lat, state.bot_result.lon], {
          radius: 7, color: "#dc2626", weight: 3,
        }).addTo(botLayer);
        $("#bot-result").textContent =
          `${state.bot_result.distance_km.toFixed(1)} km · ${Math.round(state.bot_result.score)} pts`;
        state.bot_total_score += state.bot_result.score;
        const botTruth = [state.bot_result.true_lat, state.bot_result.true_lon];
        L.circleMarker(botTruth, { radius: 7, color: "#16a34a", weight: 3 }).addTo(botLayer);
        L.polyline([[state.bot_result.lat, state.bot_result.lon], botTruth], {
          color: "#6b6b66", dashArray: "4 6",
        }).addTo(botLayer);
        botMap.fitBounds(
          L.latLngBounds([[state.bot_result.lat, state.bot_result.lon], botTruth]).pad(0.3),
          { maxZoom: 8 },
        );
        state.bot_rounds.push({
          round: state.round_number,
          distance_km: state.bot_result.distance_km,
          score: state.bot_result.score,
        });
      }
    } catch (_error) {
      $("#bot-result").textContent = "Bot unavailable";
    }
    showRoundProgress();
    if (!state.human_done) startGrace("human");
    else finishRound();
  }

  async function runBotSearch() {
    if (botViewer) {
      for (let i = 1; i <= 8; i++) {
        if (state.bot_done) return;
        botHeading = wrap(round.heading + i * STEP);
        const yaw = ((botHeading - round.north_offset + 540) % 360) - 180;
        botViewer.setYaw(yaw, 250);
        showBotHeading();
        await sleep(BOT_TURN_MS);
      }
    }
    await finishBot(false);
  }

  function finishRound() {
    clearGraceTimer();
    state.phase = "guessed";
    $("#next-round").hidden = false;
    $("#next-round").textContent =
      state.round_number >= TOTAL_ROUNDS ? "See final results" : "Next round";
  }

  async function submitHuman(guess, forced = false) {
    if (
      state.human_done ||
      !["looking", "searching"].includes(state.phase) ||
      (!guess && !state.guess)
    ) return;
    state.human_done = true;
    state.phase = "submitting";
    $("#submit").disabled = true;
    const chosen = guess || state.guess;
    const res = await fetch(`/api/round/${round.round_id}/guess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chosen),
    });
    const r = await res.json();
    const truth = [r.true_lat, r.true_lon];
    L.circleMarker([chosen.lat, chosen.lon], {
      radius: 7, color: "#2563eb", weight: 3,
    }).addTo(layer);
    L.circleMarker(truth, { radius: 7, color: "#16a34a", weight: 3 }).addTo(layer);
    L.polyline([[chosen.lat, chosen.lon], truth], { color: "#6b6b66", dashArray: "4 6" }).addTo(layer);
    map.fitBounds(L.latLngBounds([[chosen.lat, chosen.lon], truth]).pad(0.3), { maxZoom: 8 });
    $("#result").textContent =
      `${forced ? "Timeout: " : ""}${r.distance_km.toFixed(1)} km · ${Math.round(r.score)} pts`;
    state.total_score += r.score;
    state.result = r;
    state.player_rounds.push({
      round: state.round_number,
      distance_km: r.distance_km,
      score: r.score,
    });
    showRoundProgress();
    $("#attribution").hidden = false;
    const imageId = round.pano_url.split("/").pop().replace(/\.jpg$/, "");
    $("#attribution").innerHTML =
      `Imagery: <a href="https://www.mapillary.com/app/?pKey=${encodeURIComponent(imageId)}" ` +
      'target="_blank" rel="noopener">View on Mapillary</a>';
    if (DEMO) {
      if (graceTimer) clearGraceTimer();
      if (!state.bot_done) startGrace("bot");
      else finishRound();
    } else {
      finishRound();
    }
  }

  function submit() {
    submitHuman(null, false);
  }

  function loadLeaderboard() {
    try {
      const entries = JSON.parse(localStorage.getItem(LEADERBOARD_KEY) || "[]");
      return Array.isArray(entries) ? entries : [];
    } catch (_error) {
      return [];
    }
  }

  function saveLeaderboardScore(name, score) {
    const entries = loadLeaderboard();
    entries.push({
      name,
      score: Math.round(score),
      playedAt: new Date().toISOString(),
    });
    entries.sort((a, b) => b.score - a.score || a.playedAt.localeCompare(b.playedAt));
    localStorage.setItem(LEADERBOARD_KEY, JSON.stringify(entries.slice(0, 10)));
  }

  function renderLeaderboard() {
    const entries = loadLeaderboard();
    if (entries.length === 0) {
      $("#leaderboard-body").innerHTML =
        '<tr><td colspan="3">Complete a game to take the first spot.</td></tr>';
      return;
    }
    $("#leaderboard-body").innerHTML = entries.map((entry, index) =>
      `<tr><td>${index + 1}</td><td>${escapeHtml(entry.name)}</td>` +
      `<td>${Math.round(entry.score).toLocaleString()}</td></tr>`
    ).join("");
  }

  function escapeHtml(value) {
    const text = document.createElement("span");
    text.textContent = String(value);
    return text.innerHTML;
  }

  function showWinScreen() {
    clearGraceTimer();
    state.phase = "finished";
    const player = Math.round(state.total_score);
    const bot = Math.round(state.bot_total_score);
    $("#final-player-score").textContent = player.toLocaleString();
    $("#final-bot-score").textContent = bot.toLocaleString();
    $("#final-player-name").textContent = state.player_name;
    if (player > bot) {
      $("#winner-label").textContent = `${state.player_name} wins!`;
    } else if (bot > player) {
      $("#winner-label").textContent = "Bot wins!";
    } else {
      $("#winner-label").textContent = "It's a tie!";
    }
    saveLeaderboardScore(state.player_name, player);
    renderLeaderboard();
    $("#win-screen").hidden = false;
  }

  async function newRound() {
    if (state.round_number >= TOTAL_ROUNDS) {
      if (DEMO) {
        showWinScreen();
      } else {
        state.phase = "finished";
        $("#next-round").hidden = true;
        $("#summary").hidden = false;
        $("#summary").textContent =
          `Game complete — ${Math.round(state.total_score)} points`;
      }
      return;
    }
    clearGraceTimer();
    state.round_number += 1;
    Object.assign(state, {
      phase: "loading", view_ready: false, guess: null, result: null,
      bot_result: null, bot_done: false, human_done: false,
    });
    layer.clearLayers();
    botLayer.clearLayers();
    guessMarker = null;
    botGuessMarker = null;
    await mapIdle();
    map.setView(WORLD.center, WORLD.zoom, { animate: false });
    botMap.setView(WORLD.center, WORLD.zoom, { animate: false });
    $("#submit").disabled = true;
    $("#next-round").hidden = true;
    $("#result").textContent = "";
    $("#bot-result").textContent = "Bot is searching…";
    $("#bot-heading-label").textContent = "Looking";
    $("#summary").hidden = true;
    $("#attribution").hidden = true;
    showRoundProgress();
    round = await (await fetch("/api/round/new")).json();
    state.round_id = round.round_id;
    state.pano_url = round.pano_url;
    botRequest = null;
    loadPano();
    loadBotPano();
    if (DEMO) {
      state.phase = "loading";
    } else {
      state.phase = "looking";
      botViewer = null;
      if (state.round_number === 1) {
        botRequest = Promise.resolve({ enabled: false });
      }
    }
  }

  function startGame() {
    if (state.round_number !== 0) return;
    $("#name-screen").hidden = false;
    $("#player-name").value = "";
    $("#player-name").focus();
  }

  function beginNamedGame(name) {
    state.player_name = name.trim().replace(/\s+/g, " ").slice(0, 24) || "Player";
    $("#name-screen").hidden = true;
    document.querySelector(".game-header").classList.remove("is-ready");
    document.querySelector(".score-player .score-name").textContent = state.player_name;
    document.querySelector(".competitor.player-panel h2").textContent = state.player_name;
    $("#start-game").hidden = true;
    newRound();
    setTimeout(() => {
      if (DEMO) {
        state.phase = "searching";
        runBotSearch();
      }
    }, 800);
  }

  function playAgain() {
    clearGraceTimer();
    state.round_number = 0;
    state.total_score = 0;
    state.bot_total_score = 0;
    state.player_rounds = [];
    state.bot_rounds = [];
    state.bot_done = false;
    state.human_done = false;
    state.phase = "ready";
    $("#win-screen").hidden = true;
    $("#start-game").hidden = false;
    document.querySelector(".game-header").classList.add("is-ready");
    $("#next-round").hidden = true;
    $("#round-label").textContent = "Ready";
    $("#grace-timer").textContent = "";
    $("#heading-label").textContent = "Waiting";
    $("#bot-heading-label").textContent = "Waiting";
    $("#player-score").textContent = "0";
    $("#bot-score").textContent = "0";
    document.querySelector(".score-player .score-name").textContent = "Player";
    document.querySelector(".competitor.player-panel h2").textContent = "Player";
    $("#name-screen").hidden = false;
    $("#player-name").value = "";
    $("#player-name").focus();
  }

  $("#turn-left").addEventListener("click", () => turn(-1));
  $("#turn-right").addEventListener("click", () => turn(1));
  $("#submit").addEventListener("click", submit);
  $("#start-game").addEventListener("click", startGame);
  $("#next-round").addEventListener("click", newRound);
  $("#play-again").addEventListener("click", playAgain);
  $("#name-form").addEventListener("submit", (event) => {
    event.preventDefault();
    beginNamedGame($("#player-name").value);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "q") turn(-1);
    else if (e.key === "e") turn(1);
  });

  if (!DEMO) newRound();
})();
