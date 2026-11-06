"""Bulk downloader (PLAN.md 7.1): coverage tiles -> regions -> image lookups -> filters ->
``images.parquet`` + ``images/``. One entry point, two sources:

* ``source="fixtures"`` (the default): copies tests/fixtures/. ``scripts/run_e2e.py``, CI and the
  hub smoke test use this, so nothing that runs in CI ever calls Mapillary.
* ``source="mapillary"``: the real downloader behind ``pilot_v1`` and later datasets
  (``uv run python scripts/download.py --name pilot_v1 --source mapillary``).

How the real path works, per region listed in ``configs/data.yaml`` (``pilot.regions``):

1. A region is a block of zoom-14 coverage tiles around its centre (``data/tiles.py``). The
   tiles list every image in the block, with ``sequence_id``, ``creator_id`` and ``is_pano``.
2. Candidates are ordered to spread over many capture sequences, with at most
   ``max_images_per_sequence`` per sequence (``select_candidates``).
3. For each candidate, in order, until ``images_per_region`` are accepted: look up its details
   (``MapillaryClient.image``), apply the filters (panorama, has a position, quality score),
   download the thumbnail and keep it only if it is a true 2:1 panorama. Thumbnail URLs expire,
   so the image itself is stored, never the URL.
4. All rows get ``split`` from ``data.splits.assign_splits`` (by region AND creator) and are
   written with ``data.dataset.write_images``, which validates the schema and the files.

Re-running into the same directory resumes: image lookups are cached under ``<dataset>/.cache``
and images already on disk are not downloaded again. A region that fails is logged and skipped;
it never loses the other regions.
"""

from __future__ import annotations

import io
import json
import shutil
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from PIL import Image

from core.config import FIXTURES_DIR, load_config
from core.schemas import CONTINENTS, IMAGES_COLUMNS
from data.dataset import load_images, write_images
from data.mapillary import MapillaryClient, to_row
from data.splits import assign_splits, check_no_leakage
from data.tiles import MapillaryTiles, TileError, tile_for, tiles_around

CACHE_DIR = ".cache"  # inside the dataset dir; never uploaded (hub.push sends only listed files)
JPEG_QUALITY = 85  # only used if a thumbnail comes back in another format
MAX_ASPECT_SLACK_PX = 2  # |width - 2*height| allowed for a "2:1" panorama
ABORT_AFTER_FAILED_REGIONS = 3  # this many failures before any success -> stop (token/network)


class RegionError(RuntimeError):
    """A region produced no usable images."""


# --- entry point --------------------------------------------------------------------------
def download(
    out_dir: str | Path,
    split_fractions: dict[str, float],
    seed: int = 0,
    *,
    source: str = "fixtures",
    cfg: dict[str, Any] | None = None,
    log=print,
) -> pd.DataFrame:
    """Write a dataset (images.parquet + images/) to ``out_dir``; returns the images table."""
    if source == "fixtures":
        return _download_fixtures(out_dir, split_fractions, seed)
    if source == "mapillary":
        return download_mapillary(out_dir, split_fractions, seed, cfg=cfg, log=log)
    raise ValueError(f"unknown source {source!r}; use 'fixtures' or 'mapillary'")


def _download_fixtures(
    out_dir: str | Path, split_fractions: dict[str, float], seed: int
) -> pd.DataFrame:
    """The M0 placeholder: copy the fixture set instead of hitting the API."""
    out_dir = Path(out_dir)
    src = load_images(FIXTURES_DIR)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for rel in src["path"]:
        shutil.copy2(FIXTURES_DIR / rel, out_dir / rel)
    src["split"] = assign_splits(src, split_fractions, seed)
    check_no_leakage(src)
    return write_images(src, out_dir)


# --- regions and candidate selection (pure, unit-tested) ------------------------------------
def load_regions(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    """The validated ``pilot.regions`` list from configs/data.yaml."""
    regions = pilot.get("regions") or []
    if not regions:
        raise ValueError("configs/data.yaml has no pilot.regions")
    seen: set[str] = set()
    for r in regions:
        missing = [k for k in ("region_id", "country", "continent", "lat", "lon") if k not in r]
        if missing:
            raise ValueError(f"region {r!r} is missing {missing}")
        if r["region_id"] in seen:
            raise ValueError(f"duplicate region_id {r['region_id']!r}")
        seen.add(r["region_id"])
        if r["continent"] not in CONTINENTS:
            raise ValueError(f"{r['region_id']}: continent must be one of {CONTINENTS}")
        if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 180):
            raise ValueError(f"{r['region_id']}: lat/lon out of range")
    return regions


def select_candidates(
    candidates: list[dict[str, Any]], per_sequence: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Order candidates so that any prefix is spread over as many capture sequences as possible.

    At most ``per_sequence`` images are kept from each sequence (a random subset). Sequences are
    visited in random order, one image per sequence per pass, so the first N picks come from N
    different sequences when that many exist. Deterministic for a given ``rng`` seed.
    """
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: c["id"]):  # fixed order before any shuffling
        by_seq[c["sequence_id"]].append(c)
    queues = []
    for seq in sorted(by_seq):
        items = by_seq[seq]
        queues.append([items[i] for i in rng.permutation(len(items))[:per_sequence]])
    queues = [queues[i] for i in rng.permutation(len(queues))]
    ordered: list[dict[str, Any]] = []
    for depth in range(max((len(q) for q in queues), default=0)):
        ordered.extend(q[depth] for q in queues if depth < len(q))
    return ordered


# --- one image ----------------------------------------------------------------------------
def _details(client, image_id: str, cache: Path, refresh: bool = False) -> dict[str, Any]:
    """Image details from the local cache, else from the API (and cached)."""
    path = cache / f"{image_id}.json"
    if path.is_file() and not refresh:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass  # a half-written cache file: fetch again
    full = client.image(image_id)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(full))
    tmp.replace(path)
    return full


def _check_details(full: dict[str, Any], width: int, min_quality: float) -> str | None:
    """None if the image passes the filters, else the reason it is skipped."""
    if not full.get("is_pano"):
        return "not a panorama"
    if not (full.get("computed_geometry") or full.get("geometry")):
        return "no position"
    if not full.get(f"thumb_{width}_url"):
        return "no thumbnail url"
    score = full.get("quality_score")
    if score is not None and score < min_quality:  # no score = unknown = keep
        return "low quality"
    return None


def _is_good_image(path: Path) -> bool:
    """A finished download: decodable, and 2:1."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            w, h = im.size
    except (OSError, ValueError):
        return False
    return abs(w - 2 * h) <= MAX_ASPECT_SLACK_PX


def _save_image(client, full: dict[str, Any], path: Path, width: int) -> str | None:
    """Download the thumbnail to ``path``. None on success, else the reason it is skipped."""
    raw = client.download(full[f"thumb_{width}_url"])
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (OSError, ValueError):
        return "undecodable image"
    if abs(im.width - 2 * im.height) > MAX_ASPECT_SLACK_PX:
        return "not 2:1"  # a partial (non-360) panorama
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")  # write then rename: a crash never leaves a half image
    if im.format == "JPEG":
        tmp.write_bytes(raw)
    else:
        im.convert("RGB").save(tmp, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    tmp.replace(path)
    return None


def _process(
    client, cand: dict[str, Any], out_dir: Path, region: dict[str, Any], width: int, min_q: float
) -> tuple[dict[str, Any] | None, str]:
    """Look up, filter and download one candidate. Returns (row, "") or (None, skip reason)."""
    image_id = cand["id"]
    rel = f"images/{image_id}.jpg"
    path = out_dir / rel
    cache = out_dir / CACHE_DIR / "meta"
    # An image that was downloaded and found not to be 2:1 leaves a marker, so a resumed run
    # does not download it again (the image itself is deleted, so nothing else remembers it).
    not_2to1 = out_dir / CACHE_DIR / f"not_2to1_{width}" / image_id
    if not_2to1.exists():
        return None, "not 2:1"
    try:
        full = _details(client, image_id, cache)
        reason = _check_details(full, width, min_q)
        if reason:
            return None, reason
        if not _is_good_image(path):  # resume: an image already on disk is kept
            try:
                reason = _save_image(client, full, path, width)
            except requests.HTTPError:  # a cached, signed URL has expired: ask for a fresh one
                full = _details(client, image_id, cache, refresh=True)
                reason = _save_image(client, full, path, width)
            if reason == "not 2:1":
                not_2to1.parent.mkdir(parents=True, exist_ok=True)
                not_2to1.touch()
            if reason:
                return None, reason
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        return None, f"api error ({type(e).__name__}{f' {status}' if status else ''})"
    row = to_row(full)
    row.update(
        region_id=region["region_id"],
        country=region["country"],
        continent=region["continent"],
        path=rel,
        split="train",  # placeholder; assigned for the whole table at the end
    )
    return row, ""


# --- one region ---------------------------------------------------------------------------
def _fetch_region(
    region: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
    client,
    tiles,
    seed: int,
    log,
) -> tuple[list[dict[str, Any]], Counter]:
    pilot = cfg["pilot"]
    z = int(pilot.get("tile_zoom", 14))
    target = int(pilot["images_per_region"])
    width = int(cfg["image_width"])
    min_q = float(cfg.get("min_quality_score", 0.0))
    rid = region["region_id"]

    cx, cy = tile_for(region["lat"], region["lon"], z)
    found: dict[str, dict[str, Any]] = {}
    n_ok = 0
    block = tiles_around(cx, cy, z, int(pilot.get("tile_radius", 1)))
    for x, y in block:
        try:
            for cand in tiles.fetch(z, x, y):
                found.setdefault(cand["id"], cand)  # neighbouring tiles can repeat an image
            n_ok += 1
        except TileError as e:
            log(f"    {rid}: {e}")
    if n_ok == 0:
        raise RegionError(f"all {len(block)} coverage tiles failed")
    if not found:
        raise RegionError("no panoramas in the region's tiles")

    rng = np.random.default_rng([seed, zlib.crc32(rid.encode())])
    ordered = select_candidates(list(found.values()), int(cfg["max_images_per_sequence"]), rng)
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    pos = 0
    with ThreadPoolExecutor(max_workers=int(pilot.get("workers", 4))) as pool:
        while len(rows) < target and pos < len(ordered):
            batch = ordered[pos : pos + (target - len(rows))]  # never more than still needed
            pos += len(batch)
            results = pool.map(lambda c: _process(client, c, out_dir, region, width, min_q), batch)
            for row, reason in results:
                if row is None:
                    skipped[reason] += 1
                else:
                    rows.append(row)
    if not rows:
        raise RegionError(f"no usable images among {len(ordered)} candidates: {dict(skipped)}")
    return rows, skipped


# --- the run ------------------------------------------------------------------------------
def download_mapillary(
    out_dir: str | Path,
    split_fractions: dict[str, float],
    seed: int = 0,
    *,
    cfg: dict[str, Any] | None = None,
    client=None,
    tiles=None,
    log=print,
) -> pd.DataFrame:
    """The real downloader. ``client`` and ``tiles`` can be passed in (tests use fakes)."""
    cfg = cfg if cfg is not None else load_config("data")
    pilot = cfg.get("pilot") or {}
    regions = load_regions(pilot)
    out_dir = Path(out_dir)
    client = client or MapillaryClient()
    tiles = tiles or MapillaryTiles()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    target = int(pilot["images_per_region"])
    rows: list[dict[str, Any]] = []
    failed: dict[str, str] = {}
    for i, region in enumerate(regions, 1):
        rid = region["region_id"]
        try:
            region_rows, skipped = _fetch_region(region, cfg, out_dir, client, tiles, seed, log)
        except Exception as e:  # one bad region must not lose the others
            failed[rid] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"[{i}/{len(regions)}] {rid:<20} FAILED  {failed[rid]}")
            if not rows and len(failed) >= ABORT_AFTER_FAILED_REGIONS:
                raise RuntimeError(
                    f"the first {len(failed)} regions all failed, so stopping early. Check "
                    f"MAPILLARY_TOKEN and your network. Last error: {failed[rid]}"
                ) from e
            continue
        rows.extend(region_rows)
        skip_note = f"  skipped {dict(skipped)}" if skipped else ""
        log(f"[{i}/{len(regions)}] {rid:<20} {len(region_rows):>3}/{target}{skip_note}")

    if not rows:
        raise RuntimeError("no images were downloaded for any region")
    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df, split_fractions, seed)
    check_no_leakage(df)
    _log_summary(df, regions, failed, pilot, log)
    return write_images(df[list(IMAGES_COLUMNS)], out_dir)


def _log_summary(df: pd.DataFrame, regions: list, failed: dict, pilot: dict, log) -> None:
    per_region = df.groupby("region_id").size()
    log(f"\n{len(df)} images from {len(per_region)}/{len(regions)} regions")
    per_cont = df.groupby("continent")["region_id"].nunique()
    log("regions per continent: " + ", ".join(f"{c}={per_cont.get(c, 0)}" for c in CONTINENTS))
    missing = [c for c in CONTINENTS if per_cont.get(c, 0) == 0]
    if missing:
        log(f"WARNING: no images at all on {missing}")
    small = per_region[per_region < int(pilot.get("min_images_per_region", 0))]
    if len(small):
        log(f"WARNING: fewer than {pilot['min_images_per_region']} images: {small.to_dict()}")
    if failed:
        log(f"WARNING: {len(failed)} region(s) failed: {sorted(failed)}")
    scores = df["quality_score"].dropna()
    if len(scores):
        q = scores.quantile([0, 0.1, 0.25, 0.5, 0.75, 1]).round(3).tolist()
        log(
            f"quality_score: {int(df['quality_score'].isna().sum())} missing; "
            f"min/p10/p25/median/p75/max = {q}  (tune min_quality_score in configs/data.yaml)"
        )
    else:
        log("quality_score: none of the images has one, so min_quality_score filtered nothing")
