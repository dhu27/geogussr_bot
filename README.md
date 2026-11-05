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

`run_e2e.py` writes `runs/e2e/<timestamp>/`: `report.json` (metrics), a video, a Playwright
trace (`uv run playwright show-trace runs/e2e/<ts>/trace.zip`), and a side-by-side image of the
harness screenshot vs. the canonical training crop.

## Layout

| Path | What |
|---|---|
| `core/` | Contracts: view spec + crop function, schemas, interfaces, game API, scoring, metrics. Change only by reviewed PR. |
| `data/` | Mapillary client, splits (region + creator), downloader (placeholder: copies fixtures) |
| `env/` | Encoders (`FakeEncoder`, `StreetCLIPEncoder`), crop + embed pass, offline look-around env |
| `models/guesser/` | Geocells, placeholder classifier, multi-view combining, expected-score guess |
| `models/policy/` | Fixed strategies (look-around, random); PPO in Phase 2 |
| `game/` | Flask backend + Pannellum/Leaflet frontend (vendored, works offline) |
| `harness/` | Playwright harness, round loop, crop-vs-screenshot check |
| `scripts/` | `run_e2e.py`, `make_fixtures.py`, `setup_github.py` |
| `configs/` | One YAML per module |
| `tests/fixtures/` | ~50 real Mapillary panoramas (CC BY-SA, see its ATTRIBUTION.md) + fake embeddings |

## Data

Datasets and weights never go in git (ground rule 3). They live under `datasets/` and `runs/`
(gitignored), or wherever `GEO_DATA_ROOT` points. The Mapillary token goes in `.env`
(copy `.env.example`).

Imagery © Mapillary contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
