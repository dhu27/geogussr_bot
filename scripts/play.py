"""Play dataset-backed game rounds through the browser harness.

Examples:
    uv run python scripts/play.py --dataset tests/fixtures --rounds 20 --headed
    uv run python scripts/play.py --dataset datasets/pilot_v1 --rounds 100
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import Error as PlaywrightError

from core.config import load_config, resolve
from core.interfaces import Encoder, Guesser, Policy
from data.dataset import load_images
from env.encoders import make_encoder
from game.app import create_app
from game.server import serve_in_thread
from harness.agent import play_round
from harness.harness import GameHarness, MapClickError
from models.guesser.classifier import CellGuesser
from models.guesser.random import RandomGuesser
from models.policy.baselines import LookAroundPolicy


def _parser() -> argparse.ArgumentParser:
    cfg = load_config("harness")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset directory")
    parser.add_argument("--rounds", type=int, required=True, help="number of rounds to play")
    parser.add_argument("--guesser", choices=("random", "cell"), default="random")
    parser.add_argument("--weights", type=Path, help="CellGuesser .npz weights")
    parser.add_argument("--views", type=int, choices=(1, 8), default=8)
    parser.add_argument(
        "--splits", default=None, help="comma-separated game splits (default: configs/game.yaml)"
    )
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0, help="per-round timeout in seconds")
    parser.add_argument(
        "--port", type=int, default=0, help="local server port (0 selects a free port)"
    )
    parser.add_argument(
        "--out", type=Path, help="output directory (default: runs/play/<timestamp>)"
    )
    parser.set_defaults(
        slow_mo_ms=cfg["slow_mo_ms"],
        cursor_overlay=cfg["cursor_overlay"],
        viewport=cfg["viewport"],
    )
    return parser


def _make_guesser(args: argparse.Namespace, images: pd.DataFrame) -> Guesser:
    if args.guesser == "cell":
        if args.weights is None:
            raise SystemExit("--weights is required with --guesser cell")
        return CellGuesser.load(args.weights)
    locations = images.loc[images["split"] == "train", ["lat", "lon"]].to_numpy()
    if len(locations) == 0:
        locations = images[["lat", "lon"]].to_numpy()
    return RandomGuesser(locations, seed=args.seed)


def _round_record(result: dict) -> dict:
    image_id = Path(result["pano_url"]).stem
    return {
        "image_id": image_id,
        "n_views": result["n_views"],
        "pred_lat": result["model_guess"][0],
        "pred_lon": result["model_guess"][1],
        "source": "harness",
        "round_id": result["round_id"],
        "status": "ok",
        "seconds": result["seconds"],
        "distance_km": result["distance_km"],  # scored by the server at the PLACED guess
        "score": result["score"],
        # How far the map click landed from pred_lat/lon. Near 0-10 km normally; anything
        # large means distance_km/score don't measure the model's guess.
        "click_error_km": result["click_error_km"],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    root = resolve(args.dataset)
    images = load_images(root)
    game_cfg = load_config("game")
    harness_cfg = load_config("harness")
    out = args.out or resolve(f"runs/play/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    guesser = _make_guesser(args, images)
    encoder_cfg = load_config("env")
    encoder: Encoder = make_encoder(encoder_cfg)
    policy: Policy = LookAroundPolicy(args.views)
    splits = (
        [split.strip() for split in args.splits.split(",") if split.strip()]
        if args.splits is not None
        else game_cfg.get("splits")
    )
    app = create_app(root, splits=splits, seed=args.seed)
    trace_path = out / "trace.zip"
    records: list[dict] = []
    predictions: list[dict] = []
    round_loaded = True

    with (
        serve_in_thread(app, port=args.port) as url,
        GameHarness(
            url,
            headless=not args.headed,
            slow_mo_ms=harness_cfg["slow_mo_ms"] if args.headed else 0,
            cursor_overlay=harness_cfg["cursor_overlay"] if args.headed else False,
            record_video_dir=out / "videos" if args.headed else None,
            trace_path=trace_path,
            viewport=harness_cfg["viewport"],
            timeout_ms=int(args.timeout * 1000),
        ) as harness,
    ):
        for index in range(args.rounds):
            if not round_loaded:
                try:
                    harness.reload_round()
                    round_loaded = True
                except PlaywrightError as error:
                    records.append({"round": index + 1, "status": "failed", "error": str(error)})
                    predictions.append(
                        {
                            "image_id": None,
                            "n_views": None,
                            "pred_lat": None,
                            "pred_lon": None,
                            "source": "harness",
                        }
                    )
                    continue
            elif index:
                try:
                    harness.next_round()
                except PlaywrightError as error:
                    records.append({"round": index + 1, "status": "failed", "error": str(error)})
                    predictions.append(
                        {
                            "image_id": None,
                            "n_views": None,
                            "pred_lat": None,
                            "pred_lon": None,
                            "source": "harness",
                        }
                    )
                    round_loaded = False
                    continue

            started = time.monotonic()
            try:
                alarm_signal = getattr(signal, "SIGALRM", None)
                if alarm_signal is None:
                    result = play_round(harness, encoder, guesser, policy)
                else:

                    def stop_round(_signum, _frame):
                        raise TimeoutError(f"round exceeded {args.timeout:.1f}s timeout")

                    previous_handler = signal.signal(alarm_signal, stop_round)
                    signal.setitimer(signal.ITIMER_REAL, args.timeout)
                    try:
                        result = play_round(harness, encoder, guesser, policy)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(alarm_signal, previous_handler)
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(f"round exceeded {args.timeout:.1f}s timeout")
            except (PlaywrightError, TimeoutError, ValueError, MapClickError) as error:
                records.append(
                    {
                        "round": index + 1,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                predictions.append(
                    {
                        "image_id": None,
                        "n_views": None,
                        "pred_lat": None,
                        "pred_lon": None,
                        "source": "harness",
                    }
                )
                round_loaded = False
                continue

            state = harness.game_state()
            result["pano_url"] = state["pano_url"]
            record = _round_record(result)
            records.append(record)
            predictions.append(
                {
                    key: record[key]
                    for key in ("image_id", "n_views", "pred_lat", "pred_lon", "source")
                }
            )
            round_loaded = state["round_number"] < state["total_rounds"]

    pd.DataFrame(
        predictions, columns=["image_id", "n_views", "pred_lat", "pred_lon", "source"]
    ).to_parquet(
        out / "predictions.parquet",
        index=False,
    )
    (out / "rounds.json").write_text(json.dumps(records, indent=2))
    print(f"played {len(records)} rounds; outputs: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
