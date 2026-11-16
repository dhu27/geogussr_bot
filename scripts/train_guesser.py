"""Train the geocell guesser on an embedded dataset and report validation metrics (M1 task 3).

    uv run python scripts/data_hub.py pull pilot_v2_sc
    uv run python scripts/train_guesser.py --name pilot_v2_sc

Sweeps n_cells x l2 (configs/guesser.yaml), picks the setting with the best single-view mean
score on **val**, and compares it with three baselines on val: the M0 placeholder
(NearestMeanGuesser), always guessing the largest training cell, and RandomGuesser.
The test split is never read. Writes to runs/guesser/<name>/:

    guesser.npz         the chosen model (load with CellGuesser.load; use with play.py --weights)
    metrics.json        every model x {1 view, 8 views} x {overall, per continent}
    report.md           the tables below
    predictions.parquet the chosen model's val guesses (image_id, n_views, pred_lat, pred_lon)
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from core.config import load_config, resolve
from data.dataset import load_images, load_views
from data.hub import dataset_dir
from models.guesser.classifier import CellGuesser, NearestMeanGuesser
from models.guesser.evaluation import LargestCellGuesser, predict, summarize, view_matrix
from models.guesser.random import RandomGuesser

COLS = ("median_km", "mean_score", "within_25km", "within_200km", "within_750km", "within_2500km")


def table(rows: list[tuple[str, dict]], key: str) -> str:
    head = "| model | " + " | ".join(COLS) + " |\n|---|" + "---|" * len(COLS) + "\n"
    body = ""
    for name, m in rows:
        o = m[key]["overall"]
        cells = [f"{o['median_km']:.0f}", f"{o['mean_score']:.0f}"]
        cells += [f"{o[c]:.1%}" for c in COLS[2:]]
        body += f"| {name} | " + " | ".join(cells) + " |\n"
    return head + body


def continent_table(m: dict) -> str:
    out = (
        "| continent | panoramas | 1 view: median km | 1 view: mean score "
        "| 8 views: median km | 8 views: mean score |\n|---|---|---|---|---|---|\n"
    )
    for c in [k for k in m["1_view"] if k != "overall"] + ["overall"]:
        a, b = m["1_view"][c], m["8_views"][c]
        out += (
            f"| {c} | {b['n']} | {a['median_km']:.0f} | {a['mean_score']:.0f} | "
            f"{b['median_km']:.0f} | {b['mean_score']:.0f} |\n"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="embedded dataset, e.g. pilot_v2_sc")
    ap.add_argument("--out", help="default: runs/guesser/<name>")
    ap.add_argument("--quick", action="store_true", help="one setting, no sweep (for smoke tests)")
    args = ap.parse_args()
    cfg = load_config("guesser")
    root = dataset_dir(args.name)
    out = resolve(args.out or f"runs/guesser/{args.name}")
    out.mkdir(parents=True, exist_ok=True)

    images = load_images(root, check_files=False)
    views, emb = load_views(root)
    train = images[(images["split"] == "train") & images["is_pano"]].reset_index(drop=True)
    x = view_matrix(views, emb, train["image_id"].tolist())  # (n, 8, D)
    n, h, d = x.shape
    x_tr = x.reshape(n * h, d)
    lat_tr, lon_tr = np.repeat(train["lat"].to_numpy(), h), np.repeat(train["lon"].to_numpy(), h)
    n_val = int(((images["split"] == "val") & images["is_pano"]).sum())
    print(f"{args.name}: {n} train panoramas ({n * h} views), {n_val} val panoramas, D={d}")

    sweep = cfg["sweep"]
    grid = (
        [(cfg["n_cells"], cfg["l2"])]
        if args.quick
        else list(itertools.product(sweep["n_cells"], sweep["l2"]))
    )
    common = {
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "iters": cfg["kmeans_iters"],
        "seed": cfg["seed"],
    }
    results = []
    for k, l2 in grid:
        t0 = time.monotonic()
        g = CellGuesser.fit(x_tr, lat_tr, lon_tr, n_cells=k, l2=l2, **common)
        pred = predict(g, images, views, emb, "val")
        m = summarize(pred)
        results.append((k, l2, g, pred, m))
        one, eight = m["1_view"]["overall"], m["8_views"]["overall"]
        print(
            f"  K={k:>3} l2={l2:<7g} val mean score: 1 view {one['mean_score']:5.0f}, "
            f"8 views {eight['mean_score']:5.0f} | 1-view median {one['median_km']:6.0f} km "
            f"({time.monotonic() - t0:.0f}s)"
        )

    best_k, best_l2, best, best_pred, best_m = max(
        results, key=lambda r: r[4]["1_view"]["overall"]["mean_score"]
    )
    print(f"chosen on val: K={best_k}, l2={best_l2:g}")

    # Baselines on the same split, the placeholder with the chosen number of cells.
    nm = NearestMeanGuesser.fit(
        x_tr,
        lat_tr,
        lon_tr,
        n_cells=best_k,
        temperature=cfg["temperature"],
        iters=cfg["kmeans_iters"],
        seed=cfg["seed"],
    )
    largest = int(np.argmax(best.prior))
    baselines = {
        "M0 placeholder (nearest cell mean)": nm,
        "always the largest cell": LargestCellGuesser(*best.centroids[largest]),
        "random training location": RandomGuesser(
            train[["lat", "lon"]].to_numpy(), seed=cfg["seed"]
        ),
    }
    rows = [(f"**CellGuesser** (K={best_k}, l2={best_l2:g})", best_m)]
    for name, g in baselines.items():
        rows.append((name, summarize(predict(g, images, views, emb, "val"))))

    best.save(out / "guesser.npz")
    best_pred.assign(source="offline")[
        ["image_id", "n_views", "pred_lat", "pred_lon", "source"]
    ].to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": args.name,
                "split": "val",
                "chosen": {"n_cells": best_k, "l2": best_l2},
                "sweep": [{"n_cells": k, "l2": l2, **m} for k, l2, _, _, m in results],
                "models": dict(rows),
            },
            indent=2,
            default=float,
        )
    )
    report = (
        f"# Guesser on {args.name} (val split, {n_val} panoramas)\n\n"
        f"## Single view (each of 8 views on its own)\n\n{table(rows, '1_view')}\n"
        f"## Look-around (8 views combined)\n\n{table(rows, '8_views')}\n"
        f"## CellGuesser per continent\n\n{continent_table(best_m)}"
    )
    (out / "report.md").write_text(report)
    print("\n" + report + f"\nwrote {out}")


if __name__ == "__main__":
    main()
