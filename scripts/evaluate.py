"""Evaluate a trained guesser on a held-out split: single view vs 8-view look-around (M3).

    uv run python scripts/evaluate.py --name world_v1_sc            # test split, default weights

This is the value-of-information test behind decision point G1 (PLAN.md section 10): how much
lower is the median error when the agent combines all 8 views instead of using one?

* 1 view: each of a panorama's 8 views guessed on its own (the average single-view result).
* 8 views: all 8 combined through the guesser (the look-around agent).

The gain gets a 95% confidence interval from a bootstrap over **regions**, not panoramas:
panoramas from one city are strongly correlated, so resampling them individually would make the
interval far too narrow. Writes runs/eval/<name>_<split>/{report.md, metrics.json,
predictions.parquet}. Train with scripts/train_guesser.py first (it never reads the test split).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from core.config import resolve
from core.scoring import score
from data.dataset import load_images, load_views
from data.hub import dataset_dir
from models.guesser.classifier import CellGuesser
from models.guesser.evaluation import predict, summarize

COLS = ("median_km", "mean_score", "within_200km", "within_750km", "within_2500km")
G1_SUGGESTED = 0.15  # PLAN.md section 10: suggested starting point for the navigation threshold


def gain(pred: pd.DataFrame) -> dict[str, float]:
    """Look-around gain: relative drop in median error and rise in mean score, 1 -> 8 views."""
    one, eight = pred[pred["n_views"] == 1], pred[pred["n_views"] == 8]
    med1, med8 = one["distance_km"].median(), eight["distance_km"].median()
    return {
        "median_km_1": float(med1),
        "median_km_8": float(med8),
        "median_reduction": float(1 - med8 / med1),
        "mean_score_1": float(one["score"].mean()),
        "mean_score_8": float(eight["score"].mean()),
    }


def bootstrap(pred: pd.DataFrame, n: int, seed: int) -> dict[str, list[float]]:
    """95% intervals for the gain, resampling whole regions with replacement."""
    rng = np.random.default_rng(seed)
    groups = {r: g for r, g in pred.groupby("region_id")}
    regions = list(groups)
    draws = []
    for _ in range(n):
        sample = pd.concat([groups[r] for r in rng.choice(regions, len(regions))])
        g = gain(sample)
        draws.append((g["median_reduction"], g["mean_score_8"] - g["mean_score_1"]))
    d = np.array(draws)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {"median_reduction": [lo[0], hi[0]], "mean_score_gain": [lo[1], hi[1]]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="embedded dataset, e.g. world_v1_sc")
    ap.add_argument("--weights", help="default: runs/guesser/<name>/guesser.npz")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: runs/eval/<name>_<split>")
    args = ap.parse_args()

    root = dataset_dir(args.name)
    weights = resolve(args.weights or f"runs/guesser/{args.name}/guesser.npz")
    out = resolve(args.out or f"runs/eval/{args.name}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    images = load_images(root, check_files=False)
    views, emb = load_views(root)
    guesser = CellGuesser.load(weights)
    pred = predict(guesser, images, views, emb, args.split)
    pred = pred.merge(images[["image_id", "region_id"]], on="image_id")
    pred["score"] = score(pred["distance_km"].to_numpy())
    m = summarize(pred)

    g = gain(pred)
    ci = bootstrap(pred, args.bootstrap, args.seed)
    n_panos, n_regions = pred["image_id"].nunique(), pred["region_id"].nunique()
    per_cont = {
        c: gain(p) | {"regions": int(p["region_id"].nunique())}
        for c, p in pred.groupby("continent")
    }
    # Per panorama: does looking around beat that panorama's average single view?
    paired = pred.pivot_table(index="image_id", columns="n_views", values="distance_km")
    better = float((paired[8] < paired[1]).mean())

    lo, hi = ci["median_reduction"]
    verdict = "above" if lo >= G1_SUGGESTED else "below" if hi < G1_SUGGESTED else "inconclusive vs"
    report = [
        f"# Look-around value of information: {args.name}, {args.split} split",
        "",
        f"Guesser: `{weights.relative_to(resolve('.'))}`. {n_panos} panoramas in {n_regions} "
        f"regions (never seen in training). 95% intervals: bootstrap over regions, "
        f"{args.bootstrap} resamples.",
        "",
        "| setting | " + " | ".join(COLS) + " |",
        "|---|" + "---|" * len(COLS),
    ]
    for key, label in (("1_view", "1 view"), ("8_views", "8 views (look-around)")):
        o = m[key]["overall"]
        cells = [f"{o['median_km']:.0f}", f"{o['mean_score']:.0f}"]
        cells += [f"{o[c]:.1%}" for c in COLS[2:]]
        report.append(f"| {label} | " + " | ".join(cells) + " |")
    report += [
        "",
        f"**Median error reduction from looking around: {g['median_reduction']:.1%}** "
        f"(95% CI {lo:.1%} to {hi:.1%}). Mean score +{g['mean_score_8'] - g['mean_score_1']:.0f} "
        f"(95% CI +{ci['mean_score_gain'][0]:.0f} to +{ci['mean_score_gain'][1]:.0f}).",
        f"Looking around beats the average single view on {better:.0%} of panoramas.",
        "",
        f"G1 (PLAN.md section 10): the suggested threshold is {G1_SUGGESTED:.0%}; this result is "
        f"**{verdict}** it.",
        "",
        "| continent | regions | 1 view median km | 8 views median km | reduction "
        "| 1 view mean score | 8 views mean score |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, x in per_cont.items():
        report.append(
            f"| {c} | {x['regions']} | {x['median_km_1']:.0f} | {x['median_km_8']:.0f} | "
            f"{x['median_reduction']:.1%} | {x['mean_score_1']:.0f} | {x['mean_score_8']:.0f} |"
        )
    text = "\n".join(report) + "\n"
    (out / "report.md").write_text(text)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "split": args.split,
                "weights": str(weights),
                "panoramas": n_panos,
                "regions": n_regions,
                "metrics": m,
                "gain": g,
                "gain_ci95": ci,
                "per_continent": per_cont,
                "share_panos_8_better": better,
            },
            indent=2,
        )
    )
    pred.assign(source="offline")[
        ["image_id", "n_views", "pred_lat", "pred_lon", "source"]
    ].to_parquet(out / "predictions.parquet", index=False)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
