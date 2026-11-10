"""Local GeoGuessr-style game: Flask backend serving rounds from a cached dataset (PLAN.md 7.4).

Run: ``uv run python -m game.app`` (uses configs/game.yaml), then open http://127.0.0.1:5055.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from core import game_api
from core.config import load_config, resolve
from core.scoring import haversine_km, score
from core.view_spec import HEADINGS_DEG
from data.dataset import load_images

STATIC = Path(__file__).parent / "static"


def create_app(
    dataset: str | Path, splits: list[str] | None = None, seed: int | None = None
) -> Flask:
    root = Path(dataset)
    images = load_images(root)
    images = images[images["is_pano"]]
    if splits:
        images = images[images["split"].isin(splits)]
    if images.empty:
        raise ValueError(f"no panoramas in {root} for splits {splits}")
    images = images.set_index("image_id")
    rng = np.random.default_rng(seed)
    rounds: dict[str, dict] = {}

    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/panos/<image_id>.jpg")
    def pano(image_id: str):
        if image_id not in images.index:
            abort(404)
        path = root / images.loc[image_id, "path"]
        return send_from_directory(path.parent, path.name)

    @app.get(game_api.NEW_ROUND)
    def new_round():
        image_id = str(images.index[rng.integers(len(images))])
        row = images.loc[image_id]
        round_id = uuid.uuid4().hex[:12]
        rounds[round_id] = {"image_id": image_id, "guessed": False}
        body: game_api.NewRoundResponse = {
            "round_id": round_id,
            "pano_url": f"/panos/{image_id}.jpg",
            "heading": float(HEADINGS_DEG[rng.integers(len(HEADINGS_DEG))]),
            "north_offset": float(row["compass_angle"]),
        }
        return jsonify(body)

    @app.post(game_api.GUESS.replace("{round_id}", "<round_id>"))
    def guess(round_id: str):
        rnd = rounds.get(round_id)
        if rnd is None:
            abort(404)
        if rnd["guessed"]:
            abort(409, "round already guessed")
        req = request.get_json(force=True)
        try:
            lat, lon = float(req["lat"]), float(req["lon"])
        except (KeyError, TypeError, ValueError):
            abort(400, "expected JSON {lat, lon}")
        row = images.loc[rnd["image_id"]]
        d = haversine_km(lat, lon, row["lat"], row["lon"])
        rnd["guessed"] = True
        body: game_api.GuessResponse = {
            "distance_km": d,
            "score": score(d),
            "true_lat": float(row["lat"]),
            "true_lon": float(row["lon"]),
        }
        return jsonify(body)

    return app


def main() -> None:
    cfg = load_config("game")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=cfg["dataset"])
    ap.add_argument("--port", type=int, default=cfg["port"])
    ap.add_argument(
        "--splits",
        default=None,
        help="comma-separated dataset splits to serve (default: configs/game.yaml)",
    )
    args = ap.parse_args()
    splits = (
        [split.strip() for split in args.splits.split(",") if split.strip()]
        if args.splits is not None
        else cfg.get("splits")
    )
    app = create_app(resolve(args.dataset), splits, cfg.get("seed"))
    app.run(host=cfg["host"], port=args.port)


if __name__ == "__main__":
    main()
