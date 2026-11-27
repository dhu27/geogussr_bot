"""Build the nearest-neighbour guesser (models/guesser/knn.py) and tune it on val.

    uv run python scripts/train_knn.py              # sweep, save the best
    uv run python scripts/train_knn.py --quick      # one setting (smoke test)

Then: uv run python scripts/evaluate.py --model knn_daniel_v1
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from data.dataset import load_images, load_views
from data.hub import dataset_dir
from models.guesser.evaluation import predict, summarize, view_matrix
from models.guesser.knn import KNNGuesser
from models.guesser.registry import save_model

DATASET = "world_v1_sc"
# First pass (k 10/30/100, temperature 0.02/0.05) peaked at the largest k, so the grid starts there.
SWEEP = {"k": [100, 300, 1000], "temperature": [0.05, 0.1], "smoothing": [0.1, 0.3]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="knn_daniel_v1")
    ap.add_argument("--quick", action="store_true", help="one setting")
    args = ap.parse_args()

    root = dataset_dir(DATASET)
    images = load_images(root, check_files=False)
    views, emb = load_views(root)
    train = images[(images["split"] == "train") & images["is_pano"]]
    x = view_matrix(views, emb, train["image_id"].tolist())  # (n_panoramas, 8, 768)
    n, h, d = x.shape
    lat = np.repeat(train["lat"].to_numpy(), h)
    lon = np.repeat(train["lon"].to_numpy(), h)
    model = KNNGuesser.fit(x.reshape(n * h, d), lat, lon, n_cells=256)
    print(f"index: {n * h} train views")

    t0 = time.monotonic()
    val = images[(images["split"] == "val") & images["is_pano"]]
    model.warm(view_matrix(views, emb, val["image_id"].tolist()).reshape(-1, d))
    print(f"searched all val views in {time.monotonic() - t0:.0f}s")

    grid = (
        [{"k": 30, "temperature": 0.05, "smoothing": 0.3}]
        if args.quick
        else [dict(zip(SWEEP, v, strict=True)) for v in itertools.product(*SWEEP.values())]
    )
    best = None
    for setting in grid:
        t0 = time.monotonic()
        model.k, model.temperature, model.smoothing = (
            setting["k"], setting["temperature"], setting["smoothing"]
        )  # fmt: skip
        m = summarize(predict(model, images, views, emb, "val"))
        one, eight = m["1_view"]["overall"], m["8_views"]["overall"]
        print(
            f"  {setting}: val mean score 1 view {one['mean_score']:.0f}, "
            f"8 views {eight['mean_score']:.0f} | 1-view median {one['median_km']:.0f} km "
            f"({time.monotonic() - t0:.0f}s)"
        )
        if best is None or one["mean_score"] > best[0]:
            best = (one["mean_score"], setting)

    score, setting = best
    print(f"chosen on val (1-view mean score {score:.0f}): {setting}")
    model.k, model.temperature, model.smoothing = (
        setting["k"], setting["temperature"], setting["smoothing"]
    )  # fmt: skip
    out = save_model(
        model,
        name=args.name,
        dataset=DATASET,
        description=(
            f"Nearest neighbours: top {setting['k']} train views by cosine, weight "
            f"exp(sim/{setting['temperature']}), {setting['smoothing']} prior mix, 256 geocells"
        ),
        config={"n_cells": 256} | setting,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
