"""End-to-end pipeline on the fixtures: data -> crops + embeddings -> guesser -> offline eval ->
game -> Playwright harness plays rounds -> metrics. Runs in CI on every PR.

    uv run python scripts/run_e2e.py              # headless, fast
    uv run python scripts/run_e2e.py --headed     # watch it play (slow-mo + cursor dot)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from core.config import load_config, resolve
from core.interfaces import Encoder, Guesser, Policy
from core.metrics import summarize, summarize_by_group
from core.scoring import haversine_km
from data.dataset import load_views
from data.downloader import download
from env.embed import embed_dataset
from env.encoders import make_encoder
from env.lookaround_env import LookAroundEnv
from game.app import create_app
from game.server import serve_in_thread
from harness.agent import play_round
from harness.checks import crop_vs_screenshot
from harness.harness import GameHarness
from models.guesser.classifier import CellGuesser
from models.policy.baselines import LookAroundPolicy

CONSISTENCY_MIN_COSINE = 0.95


def log(msg: str) -> None:
    print(f"[e2e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def offline_eval(env: LookAroundEnv, images, n_views: int) -> dict:
    test_ids = images.loc[images["split"] == "test", "image_id"].tolist()
    policy = LookAroundPolicy(n_views)
    errs, conts = [], []
    for i, iid in enumerate(test_ids):
        obs, _ = env.reset(seed=i, image_id=iid)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy.act(obs))
            done = term or trunc
        errs.append(info["distance_km"])
        conts.append(images.set_index("image_id").loc[iid, "continent"])
    return summarize_by_group(errs, conts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=f"runs/e2e/{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--headed", action="store_true", help="visible browser with slow-mo")
    ap.add_argument("--video", action="store_true", help="record video (always on with --headed)")
    ap.add_argument("--encoder", default="fake", choices=["fake", "streetclip"])
    args = ap.parse_args()

    out = resolve(args.out)
    ds = out / "dataset"
    data_cfg, env_cfg, g_cfg = load_config("data"), load_config("env"), load_config("guesser")
    h_cfg, p_cfg = load_config("harness"), load_config("policy")
    report: dict = {"out": str(out)}

    log(f"data: placeholder downloader -> {ds}")
    split_fracs = {k: v for k, v in data_cfg["splits"].items() if k != "seed"}
    images = download(ds, split_fracs, data_cfg["splits"]["seed"])
    report["splits"] = images["split"].value_counts().to_dict()

    log(f"env: crop {len(images)} panoramas x 8 headings, embed with {args.encoder}")
    encoder: Encoder = make_encoder({**env_cfg, "encoder": args.encoder})
    embed_dataset(images, ds, encoder)
    views, emb = load_views(ds)

    log("guesser: fit geocells + softmax classifier on train split")
    v = views.merge(images[["image_id", "lat", "lon", "split"]], on="image_id")
    tr = v[v["split"] == "train"]
    guesser: Guesser = CellGuesser.fit(
        emb[tr["row"].to_numpy()].astype(np.float32),
        tr["lat"].to_numpy(),
        tr["lon"].to_numpy(),
        n_cells=g_cfg["n_cells"],
        l2=g_cfg["l2"],
        epochs=g_cfg["epochs"],
        lr=g_cfg["lr"],
        iters=g_cfg["kmeans_iters"],
        seed=g_cfg["seed"],
    )
    guesser.save(out / "guesser.npz")
    guesser = CellGuesser.load(out / "guesser.npz")  # exercise the round trip

    log("eval: offline env, single view vs look-around (test split)")
    env = LookAroundEnv(images, views, emb, guesser)
    report["offline"] = {f"{n}_view": offline_eval(env, images, n) for n in (1, p_cfg["n_views"])}

    log(f"game + harness: {args.rounds} rounds, {'headed' if args.headed else 'headless'}")
    app = create_app(ds, splits=["test"], seed=0)
    policy: Policy = LookAroundPolicy(p_cfg["n_views"])
    record = args.video or args.headed
    rounds = []
    with (
        serve_in_thread(app) as url,
        GameHarness(
            url,
            headless=not args.headed,
            slow_mo_ms=h_cfg["slow_mo_ms"] if args.headed else 0,
            cursor_overlay=h_cfg["cursor_overlay"],
            record_video_dir=out / "videos" if record else None,
            trace_path=out / "trace.zip",
            viewport=h_cfg["viewport"],
        ) as h,
    ):
        chk = crop_vs_screenshot(h, encoder, images, ds)
        Image.fromarray(np.concatenate([chk.pop("screenshot"), chk.pop("crop")], axis=1)).save(
            out / "consistency_screenshot_vs_crop.png"
        )
        report["consistency"] = chk
        log(
            f"consistency: cosine {chk['cosine']:.4f}, "
            f"mean |pixel diff| {chk['mean_abs_pixel_diff']:.1f}"
        )
        for i in range(args.rounds):
            if i:
                h.advance()  # next round, or a new game after the 5th
            r = play_round(h, encoder, guesser, policy)
            rounds.append(r)
            log(
                f"round {i + 1}: {r['distance_km']:.0f} km, {r['score']:.0f} pts, "
                f"{r['n_views']} views, click error {r['click_error_km']:.1f} km, "
                f"{r['seconds']:.1f}s"
            )
        video = h.video_path
    report["harness"] = {"rounds": rounds, "metrics": summarize([r["distance_km"] for r in rounds])}
    if video:
        report["video"] = video

    def to_json(o):
        return str(o) if isinstance(o, Path) else float(o)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=to_json))
    log(f"report: {out / 'report.json'}")

    # --- checks that fail CI ------------------------------------------------------------
    failures = []
    if chk["cosine"] < CONSISTENCY_MIN_COSINE:
        failures.append(f"crop/screenshot cosine {chk['cosine']:.3f} < {CONSISTENCY_MIN_COSINE}")
    for r in rounds:
        recomputed = haversine_km(*r["placed_guess"], r["true_lat"], r["true_lon"])
        if abs(recomputed - r["distance_km"]) > 1e-6 * max(1.0, recomputed):
            failures.append(f"round {r['round_id']}: server distance disagrees with core.scoring")
        if r["n_views"] != p_cfg["n_views"]:
            failures.append(f"round {r['round_id']}: saw {r['n_views']} views")
        if r["click_error_km"] > 50:
            failures.append(f"round {r['round_id']}: map click off by {r['click_error_km']:.0f} km")
    for f in failures:
        log(f"FAIL: {f}")
    log("OK" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
