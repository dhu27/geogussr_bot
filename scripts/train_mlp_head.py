"""Train the MLP-head guesser (models/guesser/mlp_head.py), choosing settings on val.

    uv run --extra ml python scripts/train_mlp_head.py              # experiment, save the best
    uv run --extra ml python scripts/train_mlp_head.py --quick      # one setting (smoke test)

Then: uv run python scripts/evaluate.py --model mlp_head_daniel_v2

History (val, 1-view mean score; baseline softmax_k256: 2286):
* v1 sweep over epochs {5, 15} x dropout {0.3, 0.5} x label smoothing {0, 0.1}: best 2241 with
  5 epochs, dropout 0.5, label smoothing 0.1. Every form of regularisation helped.
* v2 (this script) keeps those and tests two additions:
  - a country head as a second training target (country_weight in COUNTRY_WEIGHTS);
  - calibration: the temperature T is chosen on val after training (TEMPERATURES).
  The winner maximises the mean of the 1-view and 8-view val mean scores, since calibration
  matters most when views are combined.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from data.dataset import load_images, load_views
from data.hub import dataset_dir
from models.guesser.evaluation import predict, summarize, view_matrix
from models.guesser.mlp_head import MLPGuesser
from models.guesser.registry import save_model

DATASET = "world_v1_sc"
FIXED = {
    "n_cells": 256, "hidden": 1024, "epochs": 5, "dropout": 0.5, "label_smoothing": 0.1,
    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024,
}  # fmt: skip
COUNTRY_WEIGHTS = [0.0, 0.5, 1.0]
TEMPERATURES = [0.5, 0.7, 1.0, 1.4, 2.0]


def val_scores(model, images, views, emb) -> tuple[float, float]:
    m = summarize(predict(model, images, views, emb, "val"))
    return m["1_view"]["overall"]["mean_score"], m["8_views"]["overall"]["mean_score"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="mlp_head_daniel_v2")
    ap.add_argument("--quick", action="store_true", help="one setting, 3 epochs, T = 1")
    args = ap.parse_args()

    root = dataset_dir(DATASET)
    images = load_images(root, check_files=False)
    views, emb = load_views(root)
    train = images[(images["split"] == "train") & images["is_pano"]]
    x = view_matrix(views, emb, train["image_id"].tolist())  # (n_panoramas, 8, 768)
    n, h, d = x.shape
    x = x.reshape(n * h, d)
    lat = np.repeat(train["lat"].to_numpy(), h)
    lon = np.repeat(train["lon"].to_numpy(), h)
    country = np.repeat(train["country"].to_numpy(), h)
    print(f"{n} train panoramas ({n * h} views), {train['country'].nunique()} countries")

    weights, temps, fixed = COUNTRY_WEIGHTS, TEMPERATURES, FIXED
    if args.quick:
        weights, temps, fixed = [0.5], [1.0], FIXED | {"epochs": 3, "hidden": 512}

    best = None
    for cw in weights:
        t0 = time.monotonic()
        print(f"  country_weight {cw}:")
        model = MLPGuesser.fit(x, lat, lon, country=country, country_weight=cw, **fixed)
        for t in temps:
            model.temperature = t
            one, eight = val_scores(model, images, views, emb)
            print(f"    T={t}: val mean score 1 view {one:.0f}, 8 views {eight:.0f}")
            if best is None or (one + eight) / 2 > best[0]:
                best = ((one + eight) / 2, cw, t, model)
        print(f"    ({time.monotonic() - t0:.0f}s)")

    _, cw, t, model = best
    model.temperature = t
    print(f"chosen on val: country_weight {cw}, temperature {t}")
    out = save_model(
        model,
        name=args.name,
        dataset=DATASET,
        description=(
            f"MLP head (768 -> {fixed['hidden']} -> 256 geocells), country aux weight {cw}, "
            f"calibrated T={t}"
        ),
        config=fixed | {"country_weight": cw, "temperature": t},
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
