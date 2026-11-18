"""Evaluate any trained guesser the same way: single view vs look-around, and vs the baseline.

    uv run python scripts/evaluate.py --model softmax_k256                 # val split
    uv run python scripts/evaluate.py --model my_model --split test        # final numbers

``--model`` is a model name (runs/models/<name>/, see models/guesser/registry.py; a teammate's
model: ``scripts/data_hub.py pull-model <name>``) or a path to its folder. The dataset comes from
the model's model.json.

Reports, for 1 view (each of the 8 views on its own) and 8 views (all combined):

* the metrics in core/metrics.py, overall and per continent;
* the look-around gain (8 views vs 1), the value-of-information test behind decision point G1;
* a **paired** comparison with the baseline model (``baseline_model`` in configs/guesser.yaml):
  both models guess the same panoramas, and the per-guess differences are what get compared, so
  how hard each city is cancels out.

95% intervals come from a bootstrap over **regions**: panoramas from one city are strongly
correlated, so resampling them one by one would make the intervals far too narrow.

Choose settings on val. Use the test split once per model, for its final numbers.

Writes runs/eval/<model>/<split>/{report.md, metrics.json, predictions.parquet}. The report ends
with a table row: paste it into the leaderboard in docs/RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from core.config import load_config, resolve
from core.scoring import score
from data.dataset import load_images, load_views
from data.hub import dataset_dir
from models.guesser.evaluation import predict, summarize
from models.guesser.registry import load_model, model_info

COLS = ("median_km", "mean_score", "within_200km", "within_750km", "within_2500km")
G1_SUGGESTED = 0.15  # PLAN.md section 10: suggested starting point for the navigation threshold
KEY = ["image_id", "n_views", "view"]  # identifies one guess across models


# --- bootstrap over regions -------------------------------------------------------------
def region_bootstrap(
    pred: pd.DataFrame,
    cols: list[str],
    stat: Callable[[dict[str, np.ndarray]], tuple[float, ...]],
    n: int,
    seed: int,
) -> np.ndarray:
    """(n, len(stat)) draws of ``stat`` over resamples of whole regions (with replacement)."""
    rng = np.random.default_rng(seed)
    groups = [{c: g[c].to_numpy() for c in cols} for _, g in pred.groupby("region_id")]
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        sample = {c: np.concatenate([groups[i][c] for i in pick]) for c in cols}
        draws.append(stat(sample))
    return np.array(draws)


def _ci(draws: np.ndarray) -> list[list[float]]:
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return [[float(a), float(b)] for a, b in zip(lo, hi, strict=True)]


# --- look-around gain (8 views vs 1) --------------------------------------------------------
def gain(pred: pd.DataFrame) -> dict[str, float]:
    """Relative drop in median error and rise in mean score, 1 -> 8 views."""
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
    """95% intervals for the look-around gain."""
    p = pred.assign(
        d1=np.where(pred["n_views"] == 1, pred["distance_km"], np.nan),
        d8=np.where(pred["n_views"] == 8, pred["distance_km"], np.nan),
        s1=np.where(pred["n_views"] == 1, pred["score"], np.nan),
        s8=np.where(pred["n_views"] == 8, pred["score"], np.nan),
    )

    def stat(s):
        return (
            1 - np.nanmedian(s["d8"]) / np.nanmedian(s["d1"]),
            np.nanmean(s["s8"]) - np.nanmean(s["s1"]),
        )

    (red, sc) = _ci(region_bootstrap(p, ["d1", "d8", "s1", "s8"], stat, n, seed))
    return {"median_reduction": red, "mean_score_gain": sc}


# --- paired comparison with a baseline model --------------------------------------------
def paired(pred: pd.DataFrame, base: pd.DataFrame, n: int, seed: int) -> dict[str, dict]:
    """Model minus baseline on the same guesses, per setting (1 view, 8 views)."""
    both = pred.merge(
        base[KEY + ["distance_km", "score"]], on=KEY, suffixes=("", "_base"), validate="1:1"
    )
    if len(both) != len(pred):
        raise ValueError("model and baseline guessed different panoramas; same dataset/split?")

    def stat(s):
        return (
            np.median(s["d"]) / np.median(s["db"]) - 1,
            np.mean(s["s"] - s["sb"]),
            np.mean(s["d"] < s["db"]),
        )

    out = {}
    for n_views, key in ((1, "1_view"), (8, "8_views")):
        b = both[both["n_views"] == n_views].rename(
            columns={"distance_km": "d", "distance_km_base": "db", "score": "s", "score_base": "sb"}
        )
        point = stat({c: b[c].to_numpy() for c in ("d", "db", "s", "sb")})
        ci = _ci(region_bootstrap(b, ["d", "db", "s", "sb"], stat, n, seed))
        out[key] = {
            "median_km_change": float(point[0]),
            "median_km_change_ci95": ci[0],
            "mean_score_diff": float(point[1]),
            "mean_score_diff_ci95": ci[1],
            "share_closer": float(point[2]),
        }
    return out


# --- run ------------------------------------------------------------------------------------
def evaluate_model(model: str, split: str, images, views, emb) -> pd.DataFrame:
    pred = predict(load_model(model), images, views, emb, split)
    pred = pred.merge(images[["image_id", "region_id"]], on="image_id")
    pred["score"] = score(pred["distance_km"].to_numpy())
    return pred


def _git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def leaderboard_row(r: dict) -> str:
    """One row for the leaderboard table in docs/RESULTS.md (same columns)."""
    one, eight = r["metrics"]["1_view"]["overall"], r["metrics"]["8_views"]["overall"]
    vs = r.get("vs_baseline")
    if vs:
        e = vs["8_views"]
        lo, hi = e["mean_score_diff_ci95"]
        versus = f"{e['mean_score_diff']:+.0f} ({lo:+.0f} to {hi:+.0f})"
    else:
        versus = "(baseline)"
    m = r["model"]
    return (
        f"| `{m['name']}` | {m.get('author') or '?'} | {r['split']} | {one['median_km']:.0f} | "
        f"{one['mean_score']:.0f} | {eight['median_km']:.0f} | {eight['mean_score']:.0f} | "
        f"{versus} | {m.get('description') or ''} |"
    )


def report_md(r: dict) -> str:
    m, lo = r["metrics"], r["look_around"]
    lines = [
        f"# {r['model']['name']} on {r['dataset']}, {r['split']} split",
        "",
        f"{r['model'].get('description') or ''} Kind `{r['model']['kind']}`, by "
        f"{r['model'].get('author') or '?'}. {r['panoramas']} panoramas in {r['regions']} regions "
        f"(never seen in training). 95% intervals: bootstrap over regions.",
        "",
        "| setting | " + " | ".join(COLS) + " |",
        "|---|" + "---|" * len(COLS),
    ]
    for key, label in (("1_view", "1 view"), ("8_views", "8 views")):
        o = m[key]["overall"]
        cells = [f"{o['median_km']:.0f}", f"{o['mean_score']:.0f}"]
        cells += [f"{o[c]:.1%}" for c in COLS[2:]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    red, red_ci = lo["median_reduction"], lo["median_reduction_ci95"]
    verdict = (
        "above"
        if red_ci[0] >= G1_SUGGESTED
        else "below"
        if red_ci[1] < G1_SUGGESTED
        else "inconclusive vs"
    )
    sg, sg_ci = lo["mean_score_8"] - lo["mean_score_1"], lo["mean_score_gain_ci95"]
    lines += [
        "",
        f"**Look-around:** median error {red:.1%} lower with 8 views "
        f"(95% CI {red_ci[0]:.1%} to {red_ci[1]:.1%}); mean score {sg:+.0f} "
        f"({sg_ci[0]:+.0f} to {sg_ci[1]:+.0f}). G1's suggested {G1_SUGGESTED:.0%} threshold: "
        f"**{verdict}**.",
    ]
    vs = r.get("vs_baseline")
    if vs:
        lines += [
            "",
            f"## Paired with the baseline `{vs['baseline']}` (same panoramas)",
            "",
            "| setting | median km change | mean score difference | closer than baseline |",
            "|---|---|---|---|",
        ]
        for key, label in (("1_view", "1 view"), ("8_views", "8 views")):
            x = vs[key]
            c, d = x["median_km_change_ci95"], x["mean_score_diff_ci95"]
            lines.append(
                f"| {label} | {_pct(x['median_km_change'])} ({_pct(c[0])} to {_pct(c[1])}) | "
                f"{x['mean_score_diff']:+.0f} ({d[0]:+.0f} to {d[1]:+.0f}) | "
                f"{x['share_closer']:.0%} |"
            )
        lines += ["", "An interval that excludes 0 is a real difference, not luck."]
    lines += [
        "",
        "## Per continent",
        "",
        "| continent | panoramas | 1 view median km | 1 view mean score | 8 views median km "
        "| 8 views mean score |",
        "|---|---|---|---|---|---|",
    ]
    for c in [k for k in m["1_view"] if k != "overall"]:
        a, b = m["1_view"][c], m["8_views"][c]
        lines.append(
            f"| {c} | {b['n']} | {a['median_km']:.0f} | {a['mean_score']:.0f} | "
            f"{b['median_km']:.0f} | {b['mean_score']:.0f} |"
        )
    lines += ["", "## Row for the leaderboard in docs/RESULTS.md", "", leaderboard_row(r)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="model name (runs/models/<name>) or folder")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument(
        "--baseline",
        help="model to compare with, paired (default: baseline_model in configs/guesser.yaml; "
        "'none' to skip)",
    )
    ap.add_argument("--dataset", help="override the dataset named in the model's model.json")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: runs/eval/<model>/<split>")
    args = ap.parse_args()

    info = model_info(args.model)
    name = info["name"]
    dataset = args.dataset or info["dataset"]
    baseline = args.baseline or load_config("guesser").get("baseline_model")
    if baseline in (None, "", "none") or baseline == name:
        baseline = None
    if args.split == "test":
        print("test split: use it once per model, for final numbers (choose settings on val)")

    t0 = time.monotonic()
    root = dataset_dir(dataset)
    images = load_images(root, check_files=False)
    views, emb = load_views(root)
    pred = evaluate_model(args.model, args.split, images, views, emb)
    m = summarize(pred)
    g = gain(pred)
    ci = bootstrap(pred, args.bootstrap, args.seed)
    result = {
        "model": {k: info.get(k) for k in ("name", "kind", "author", "description", "config")}
        | {"code_commit": info.get("code_commit"), "dataset": info["dataset"]},
        "dataset": dataset,
        "split": args.split,
        "panoramas": int(pred["image_id"].nunique()),
        "regions": int(pred["region_id"].nunique()),
        "metrics": m,
        "look_around": g
        | {
            "median_reduction_ci95": ci["median_reduction"],
            "mean_score_gain_ci95": ci["mean_score_gain"],
        },
        "vs_baseline": None,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "eval_commit": _git_commit(),
    }
    if baseline:
        base_pred = evaluate_model(baseline, args.split, images, views, emb)
        result["vs_baseline"] = {"baseline": model_info(baseline)["name"]} | paired(
            pred, base_pred, args.bootstrap, args.seed
        )

    out = resolve(args.out or f"runs/eval/{name}/{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    text = report_md(result)
    (out / "report.md").write_text(text)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    pred.assign(source="offline")[
        KEY + ["pred_lat", "pred_lon", "distance_km", "source"]
    ].to_parquet(out / "predictions.parquet", index=False)
    print(text)
    print(f"wrote {out} ({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
