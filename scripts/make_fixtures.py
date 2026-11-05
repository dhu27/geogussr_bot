"""Build tests/fixtures/: ~50 real Mapillary panoramas (2 per region), images.parquet, fake
embeddings, and ATTRIBUTION.md (CC-BY-SA, ground rule 4).

    uv run python scripts/make_fixtures.py               # download + derive (needs MAPILLARY_TOKEN)
    uv run python scripts/make_fixtures.py --derive-only # recompute splits + fake embeddings

The fixtures are committed so CI and everyone's tests use the same set. Re-run only when the
contract in core/ changes; it rewrites the files in place.
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
from PIL import Image

from core.config import FIXTURES_DIR, load_config
from core.schemas import IMAGES_COLUMNS
from data.dataset import load_images, write_images
from data.splits import assign_splits, check_no_leakage
from env.embed import embed_dataset
from env.encoders import FakeEncoder

JPEG_QUALITY = 85
MIN_TOTAL = 40


def _seq(c: dict) -> str:
    s = c.get("sequence")
    return str(s["id"] if isinstance(s, dict) else s)


def rank(candidates: list[dict]) -> list[dict]:
    """Order candidates: best quality per distinct sequence first, then the rest (from the far
    end of the API's ordering, to spread same-sequence picks apart)."""
    by_quality = sorted(candidates, key=lambda c: -(c.get("quality_score") or 0.0))
    first, seen = [], set()
    for c in by_quality:
        if _seq(c) not in seen:
            first.append(c)
            seen.add(_seq(c))
    rest = [c for c in reversed(candidates) if c not in first]
    return first + rest


def search_region(client, reg: dict, half0: float, need: int) -> tuple[list[dict], float]:
    """Search around the region centre. Dense city centres make the API return 500s, so on an
    error the box shrinks; when too few panoramas come back, it grows."""
    import requests

    tried: set[float] = set()
    half = half0
    found: list[dict] = []
    while half not in tried and 0.001 <= half <= 0.08:
        tried.add(half)
        bbox = (reg["lon"] - half, reg["lat"] - half, reg["lon"] + half, reg["lat"] + half)
        try:
            found = [c for c in client.search_images(bbox) if c.get("is_pano")]
        except requests.RequestException as e:
            print(
                f"  {reg['region_id']}: search ±{half:.4f}° failed ({type(e).__name__}), shrinking"
            )
            half /= 2
            continue
        if len(found) >= need:
            break
        half *= 2
    return found, half


def download(cfg: dict) -> pd.DataFrame:
    from data.mapillary import MapillaryClient, to_row

    client = MapillaryClient()
    fx = cfg["fixtures"]
    (FIXTURES_DIR / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for reg in fx["regions"]:
        try:
            found, half = search_region(client, reg, fx["bbox_half_deg"], fx["per_region"])
            got = 0
            for c in rank(found):
                if got == fx["per_region"]:
                    break
                full = client.image(c["id"])
                if not (full.get("is_pano") and full.get("thumb_1024_url")):
                    continue
                rel = f"images/{full['id']}.jpg"
                if not (FIXTURES_DIR / rel).is_file():  # resume: keep earlier downloads
                    raw = client.download(full["thumb_1024_url"])
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    if abs(im.width - 2 * im.height) > 2:  # partial panorama
                        continue
                    im.save(FIXTURES_DIR / rel, quality=JPEG_QUALITY, optimize=True)
                row = to_row(full)
                row.update(
                    region_id=reg["region_id"],
                    country=reg["country"],
                    continent=reg["continent"],
                    path=rel,
                    split="train",
                )
                rows.append(row)
                got += 1
            print(f"{reg['region_id']:<18} {got}/{fx['per_region']} (box ±{half:.4f}°)")
        except Exception as e:  # one bad region must not lose the others
            print(f"{reg['region_id']:<18} FAILED: {type(e).__name__}: {str(e)[:120]}")
    if len(rows) < MIN_TOTAL:
        print(f"WARNING: only {len(rows)} panoramas (want >= {MIN_TOTAL}); re-run to retry")
    return pd.DataFrame(rows)


def write_attribution(df: pd.DataFrame) -> None:
    lines = [
        "# Fixture attribution",
        "",
        "Panoramas in `images/` are from [Mapillary](https://www.mapillary.com) and licensed",
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Downscaled to 1024 px.",
        "",
        "| image | creator | region |",
        "|---|---|---|",
    ]
    for r in df.sort_values(["region_id", "image_id"]).itertuples():
        link = f"https://www.mapillary.com/app/?pKey={r.image_id}"
        lines.append(
            f"| [{r.image_id}]({link}) | {r.creator_username or r.creator_id} | {r.region_id} |"
        )
    (FIXTURES_DIR / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--derive-only", action="store_true", help="skip the download")
    args = ap.parse_args()
    cfg = load_config("data")

    if args.derive_only:
        df = load_images(FIXTURES_DIR)
    else:
        df = download(cfg)
        write_attribution(df)
    df["split"] = assign_splits(df, cfg["splits"], cfg["splits"]["seed"])
    check_no_leakage(df)
    df = write_images(df[list(IMAGES_COLUMNS)], FIXTURES_DIR)
    views, _ = embed_dataset(df, FIXTURES_DIR, FakeEncoder())
    print(
        f"{len(df)} panoramas, {len(views)} views, splits: {df['split'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
