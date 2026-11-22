"""Game HTTP API and page elements (PLAN.md 6.4), shared by game/ and harness/."""

from __future__ import annotations

from typing import TypedDict

# --- HTTP ---------------------------------------------------------------------------------
NEW_ROUND = "/api/round/new"  # GET
GUESS = "/api/round/{round_id}/guess"  # POST {lat, lon}
BOT_GUESS = "/api/round/{round_id}/bot"  # GET
NEIGHBORS = "/api/node/{node_id}/neighbors"  # GET, Phase 2


class NewRoundResponse(TypedDict):
    round_id: str
    pano_url: str
    heading: float  # compass heading the viewer starts at; one of view_spec.HEADINGS_DEG
    north_offset: float  # compass heading of the panorama's centre column (Pannellum northOffset)


class GuessRequest(TypedDict):
    lat: float
    lon: float


class GuessResponse(TypedDict):
    distance_km: float
    score: float
    true_lat: float
    true_lon: float


class Neighbor(TypedDict):
    node_id: str
    bearing_deg: float


# --- Page elements ------------------------------------------------------------------------
PANO = "#pano"  # the only element the harness screenshots for the model
MAP = "#map"
SUBMIT = "#submit"
NEXT_ROUND = "#next-round"
TURN_LEFT = "#turn-left"  # rotate exactly -45 degrees (outside #pano)
TURN_RIGHT = "#turn-right"  # rotate exactly +45 degrees (outside #pano)

# window.__game: {round_id, pano_url, heading, phase, view_ready, guess, result}. FOR LOGGING AND
# SYNCHRONISATION ONLY: the harness must never pass any of it to a model (ground rule 1).
GAME_STATE_JS = "window.__game"
