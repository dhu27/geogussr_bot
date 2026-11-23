"""Local GeoGuessr-style game: Flask backend serving rounds from a cached dataset (PLAN.md 7.4).

Run: ``uv run python -m game.app`` (uses configs/game.yaml), then open http://127.0.0.1:5055.
"""

from __future__ import annotations

import argparse
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from core import game_api
from core.config import load_config, resolve
from core.scoring import haversine_km, score
from core.view_spec import HEADINGS_DEG
from data.dataset import load_images, load_views
from models.guesser.evaluation import view_matrix
from models.guesser.random import RandomGuesser
from models.guesser.registry import load_model

STATIC = Path(__file__).parent / "static"


def create_app(
    dataset: str | Path,
    splits: list[str] | None = None,
    seed: int | None = None,
    bot_guesser=None,
    bot_requires_embeddings: bool = False,
) -> Flask:
    # Absolute: send_from_directory resolves a relative path against this package, not the CWD,
    # so create_app("datasets/x") used to 404 on every panorama.
    root = Path(dataset).resolve()
    images = load_images(root)
    images = images[images["is_pano"]]
    if splits:
        images = images[images["split"].isin(splits)]
    if images.empty:
        raise ValueError(f"no panoramas in {root} for splits {splits}")
    images = images.set_index("image_id")
    bot_views = None
    bot_embeddings = None
    if bot_guesser is not None and bot_requires_embeddings:
        try:
            bot_views, bot_embeddings = load_views(root)
        except FileNotFoundError as error:
            raise ValueError(
                f"--model requires views.parquet and embeddings.npy in {root}"
            ) from error
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

    @app.get(game_api.BOT_GUESS.replace("{round_id}", "<round_id>"))
    def bot_guess(round_id: str):
        rnd = rounds.get(round_id)
        if rnd is None:
            abort(404)
        if bot_guesser is None:
            return jsonify({"enabled": False})
        if "bot_result" not in rnd:
            image_id = rnd["image_id"]
            if bot_views is not None and bot_embeddings is not None:
                x = view_matrix(bot_views, bot_embeddings, [image_id])
                probs = bot_guesser.cell_probs(x[0])
            else:
                probs = bot_guesser.cell_probs(np.empty((1, 768), dtype=np.float32))
            lat, lon = bot_guesser.guess(probs)
            row = images.loc[image_id]
            d = haversine_km(lat, lon, row["lat"], row["lon"])
            rnd["bot_result"] = {
                "enabled": True,
                "lat": float(lat),
                "lon": float(lon),
                "distance_km": d,
                "score": score(d),
                "true_lat": float(row["lat"]),
                "true_lon": float(row["lon"]),
            }
        return jsonify(rnd["bot_result"])

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
    ap.add_argument(
        "--model",
        default=None,
        help="saved model name or folder for the presentation bot (e.g. softmax_k256)",
    )
    ap.add_argument("--open", action="store_true", help="open the game in the default browser")
    args = ap.parse_args()
    splits = (
        [split.strip() for split in args.splits.split(",") if split.strip()]
        if args.splits is not None
        else cfg.get("splits")
    )
    root = resolve(args.dataset)
    if args.model:
        bot_guesser = load_model(args.model)
        bot_requires_embeddings = True
    else:
        images = load_images(root)
        training = images[images["split"] == "train"]
        if training.empty:
            training = images
        locations = training[["lat", "lon"]].to_numpy()
        bot_guesser = RandomGuesser(locations, seed=cfg.get("seed") or 0)
        bot_requires_embeddings = False
    app = create_app(
        root,
        splits,
        cfg.get("seed"),
        bot_guesser,
        bot_requires_embeddings=bot_requires_embeddings,
    )
    if args.open:
        webbrowser.open(f"http://{cfg['host']}:{args.port}/?demo=1")
    app.run(host=cfg["host"], port=args.port)


if __name__ == "__main__":
    main()
