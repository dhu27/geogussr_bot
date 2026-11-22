# GeoGuessr RL Agent

An agent that plays a local GeoGuessr-style game in a real browser. See [PLAN.md](PLAN.md) for
goals, contracts, and milestones.

## Setup

```bash
uv sync                                   # Python 3.11 + deps (add --extra ml for StreetCLIP/torch)
uv run playwright install chromium        # browser for the harness
uv run pre-commit install                 # ruff on every commit
```

## Run the placeholder pipeline (M0 "done when")

```bash
uv run pytest                                   # unit + browser tests on tests/fixtures/
uv run python scripts/run_e2e.py                # full chain, headless: data -> embed -> guesser -> game -> harness
uv run python scripts/run_e2e.py --headed       # same, in a visible window with slow-mo and a cursor dot
uv run python -m game.app                       # play the game yourself at http://127.0.0.1:5055
```

## Presentation: human vs. bot

The local game can run five rounds in a side-by-side layout. By default it uses the repo's random
baseline, so no model download or extra setup is required. The human uses the left map and the bot
uses the right map. Hit **Start game** to begin; both players search independently, and the bot
visibly turns through all eight views. Whichever player finishes first gives the other a clear
30-second grace timer. Once both guesses are locked, the result and map reveal together.

Before the first round, enter a player name. The name is asked again when choosing **New player /
play again**, and the final screen shows a browser-persistent top-10 leaderboard. The leaderboard
uses local browser storage, so it remains available while the presentation stays open or refreshes.

```bash
uv run python -m game.app --open
```

Each round awards up to 5,000 points using `5000 × exp(-distance_km / 1492.7)`; a
perfect guess earns 5,000 and score decreases exponentially with distance.

Once a trained model is available, `--model` accepts a model name from `runs/models/` or a saved
model folder:

```bash
uv run python -m game.app --model softmax_k256 --dataset datasets/world_v1_sc --open
```

Saved models use the
registry in `models/guesser/registry.py` (`model.json` plus a registered guesser implementation),
so a teammate's trained variant can be plugged in without changing the game. The dataset must
include `views.parquet` and `embeddings.npy` for the bot to compute all eight StreetCLIP views.

`run_e2e.py` writes `runs/e2e/<timestamp>/`: `report.json` (metrics), a video, a Playwright
trace (`uv run playwright show-trace runs/e2e/<ts>/trace.zip`), and a side-by-side image of the
harness screenshot vs. the canonical training crop.

## Layout

| Path | What |
|---|---|
| `core/` | Contracts: view spec + crop function, schemas, interfaces, game API, scoring, metrics. Change only by reviewed PR. |
| `data/` | Mapillary client + coverage tiles, downloader (`--source mapillary`; `fixtures` for CI), splits (region + creator, balanced per continent), Hugging Face hub push/pull |
| `env/` | Encoders (`FakeEncoder`, `StreetCLIPEncoder`), crop + embed pass, offline look-around env |
| `models/guesser/` | Geocells, trained softmax classifier (+ M0 nearest-mean baseline), multi-view combining, expected-score guess, offline evaluation, model registry (`registry.py`). Results: [docs/RESULTS.md](docs/RESULTS.md); building a variant: [docs/GUESSER_VARIANTS.md](docs/GUESSER_VARIANTS.md) |
| `models/policy/` | Fixed strategies (look-around, random); PPO in Phase 2 |
| `game/` | Flask backend + Pannellum/Leaflet frontend (vendored, works offline) |
| `harness/` | Playwright harness, round loop, crop-vs-screenshot check |
| `scripts/` | `run_e2e.py`, `download.py`, `embed.py`, `train_guesser.py`, `evaluate.py`, `play.py`, `data_hub.py`, `make_fixtures.py`, `setup_github.py` |
| `configs/` | One YAML per module |
| `tests/fixtures/` | ~50 real Mapillary panoramas (CC BY-SA, see its ATTRIBUTION.md) + fake embeddings |

## Data

Datasets and weights never go in git (ground rule 3). The team shares them through a private
Hugging Face dataset repo; locally they live under `datasets/` (gitignored) or wherever
`GEO_DATA_ROOT` points. Tokens go in `.env` (copy `.env.example`; never put real tokens in it).

```bash
uv run python scripts/data_hub.py list              # datasets on the hub
uv run python scripts/data_hub.py pull pilot_v1     # -> datasets/pilot_v1/
```

Setup, the download path, and the hub layout are in **[docs/DATA.md](docs/DATA.md)**.

Imagery © Mapillary contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
