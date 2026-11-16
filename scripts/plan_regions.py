"""Plan the M2 region list: ~500 regions balanced by country, each checked for panorama
coverage BEFORE the download (so dead regions are swapped out up front, not found mid-run).

    uv run python scripts/plan_regions.py                       # -> configs/regions/world_v1.yaml
    uv run python scripts/download.py --name world_v1 --source mapillary \\
        --regions-file configs/regions/world_v1.yaml

Steps:
1. Candidates: GeoNames cities (CC BY 4.0) with population >= --min-pop, the --per-country
   largest per country. Downloaded once into <data_root>/_geonames/.
2. Screen: read each candidate's centre zoom-14 coverage tile and estimate its *capacity*, the
   panoramas the downloader could take (at most max_images_per_sequence per capture sequence).
3. Pick round-robin over countries (each round, every country takes its largest city that
   passes), at most --max-per-country per country, centres at least --min-km apart so the tile
   blocks of two regions never overlap.
4. Check each pick's full tile block (the downloader's 3x3) and keep it only if its capacity is
   at least --min-capacity, leaving headroom for images the quality/2:1 filters drop.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import threading
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import yaml

from core.config import load_config, resolve
from core.schemas import CONTINENTS
from core.scoring import haversine_km
from data.hub import data_root
from data.tiles import MapillaryTiles, TileError, tile_for, tiles_around

GEONAMES = "https://download.geonames.org/export/dump/"
CITY_COLS = {1: "name", 4: "lat", 5: "lon", 7: "feature_code", 8: "country", 14: "population"}


# --- candidates -----------------------------------------------------------------------------
def fetch_geonames(folder: Path) -> tuple[Path, Path]:
    """cities15000.txt and countryInfo.txt, downloaded once."""
    folder.mkdir(parents=True, exist_ok=True)
    cities, countries = folder / "cities15000.txt", folder / "countryInfo.txt"
    if not cities.is_file():
        r = requests.get(GEONAMES + "cities15000.zip", timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            cities.write_bytes(z.read("cities15000.txt"))
    if not countries.is_file():
        r = requests.get(GEONAMES + "countryInfo.txt", timeout=120)
        r.raise_for_status()
        countries.write_bytes(r.content)
    return cities, countries


def load_candidates(cities: Path, countries: Path, min_pop: int, per_country: int) -> pd.DataFrame:
    df = pd.read_csv(
        cities,
        sep="\t",
        header=None,
        usecols=list(CITY_COLS),
        quoting=3,
        keep_default_na=False,
        dtype=str,
    ).rename(columns=CITY_COLS)
    df[["lat", "lon"]] = df[["lat", "lon"]].astype(float)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    rows = [ln.split("\t") for ln in countries.read_text().splitlines() if ln and ln[0] != "#"]
    continent = {r[0]: r[8] for r in rows if len(r) > 8}
    df["continent"] = df["country"].map(continent)
    df = df[
        df["feature_code"].str.startswith("PPL")
        & (df["population"] >= min_pop)
        & df["continent"].isin(CONTINENTS)
    ]
    df = df.sort_values(["country", "population"], ascending=[True, False])
    return df.groupby("country").head(per_country).reset_index(drop=True)


def slug(text: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_") or "x"


# --- coverage -------------------------------------------------------------------------------
class TileCounter:
    """Panoramas per tile, one HTTP session per thread. Results are cached in memory and, if
    ``cache_dir`` is given, on disk, so an interrupted or repeated run doesn't refetch tiles.
    A failed fetch is never cached."""

    def __init__(self, z: int, cache_dir: Path | None = None):
        self.z, self._local, self._cache, self._lock = z, threading.local(), {}, threading.Lock()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _disk(self, x: int, y: int) -> Path | None:
        return self.cache_dir / f"{self.z}_{x}_{y}.json" if self.cache_dir else None

    def _tiles(self) -> MapillaryTiles:
        if not hasattr(self._local, "t"):
            self._local.t = MapillaryTiles()
        return self._local.t

    def panos(self, x: int, y: int) -> list[dict] | None:
        with self._lock:
            if (x, y) in self._cache:
                return self._cache[(x, y)]
        disk = self._disk(x, y)
        if disk and disk.is_file():
            found = [{"id": i, "sequence_id": q} for i, q in json.loads(disk.read_text())]
        else:
            try:
                found = self._tiles().fetch(self.z, x, y)
            except TileError:
                found = None  # unknown, not empty
            if disk and found is not None:
                tmp = disk.with_suffix(".part")
                tmp.write_text(json.dumps([[p["id"], p["sequence_id"]] for p in found]))
                tmp.replace(disk)
        with self._lock:
            self._cache[(x, y)] = found
        return found


def capacity(panos: list[dict], per_sequence: int) -> tuple[int, int, int]:
    """(panoramas, sequences, capacity = sum over sequences of min(count, per_sequence))."""
    per_seq = Counter(p["sequence_id"] for p in {p["id"]: p for p in panos}.values())
    return sum(per_seq.values()), len(per_seq), sum(min(n, per_sequence) for n in per_seq.values())


# --- planning -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--min-pop", type=int, default=50_000)
    ap.add_argument("--per-country", type=int, default=30, help="candidate cities per country")
    ap.add_argument("--max-per-country", type=int, default=12)
    ap.add_argument("--min-km", type=float, default=25.0, help="minimum distance between centres")
    ap.add_argument("--min-centre", type=int, default=15, help="screen: centre-tile capacity")
    ap.add_argument("--min-capacity", type=int, default=150, help="full-block capacity to keep")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="configs/regions/world_v1.yaml")
    args = ap.parse_args()

    cfg = load_config("data")
    plan = cfg["pilot"]
    z, radius = int(plan.get("tile_zoom", 14)), int(plan.get("tile_radius", 1))
    per_seq = int(cfg["max_images_per_sequence"])

    cand = load_candidates(
        *fetch_geonames(data_root() / "_geonames"), args.min_pop, args.per_country
    )
    print(f"{len(cand)} candidate cities in {cand['country'].nunique()} countries")

    counter = TileCounter(z, data_root() / "_tile_cache")  # coverage as of this run
    cand["tile"] = [tile_for(la, lo, z) for la, lo in zip(cand["lat"], cand["lon"], strict=True)]
    with ThreadPoolExecutor(args.workers) as pool:
        centre = list(pool.map(lambda t: counter.panos(*t), cand["tile"]))
    cand["centre_capacity"] = [capacity(p, per_seq)[2] if p is not None else -1 for p in centre]
    passed = cand[cand["centre_capacity"] >= args.min_centre]
    print(
        f"screen: {len(passed)} candidates have centre-tile capacity >= {args.min_centre} "
        f"({(cand['centre_capacity'] < 0).sum()} tiles failed to load)"
    )

    # Fetch the full blocks of the cities that could be picked (each country's largest, up to its
    # cap plus a few spares), in parallel; picking below then reads them from the cache.
    likely = passed.groupby("country").head(args.max_per_country + 4)
    block_tiles = {t for c in likely["tile"] for t in tiles_around(*c, z, radius)}
    print(f"fetching {len(block_tiles)} block tiles for {len(likely)} likely picks...")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda t: counter.panos(*t), block_tiles))

    # Round-robin over countries, largest city first within each.
    queues = {c: g.to_dict("records") for c, g in passed.groupby("country")}
    order = sorted(queues, key=lambda c: -passed.loc[passed["country"] == c, "population"].sum())
    chosen, rejected = [], Counter()
    taken = defaultdict(int)
    while len(chosen) < args.target and any(queues.values()):
        progress = False
        for country in order:
            if len(chosen) >= args.target:
                break
            q = queues[country]
            while q and taken[country] < args.max_per_country:
                c = q.pop(0)
                if any(
                    haversine_km(c["lat"], c["lon"], o["lat"], o["lon"]) < args.min_km
                    for o in chosen
                ):
                    rejected["too close to a chosen region"] += 1
                    continue
                block = [counter.panos(x, y) for x, y in tiles_around(*c["tile"], z, radius)]
                if any(b is None for b in block):
                    rejected["tile failed to load"] += 1
                    continue
                n, seqs, cap = capacity([p for b in block for p in b], per_seq)
                if cap < args.min_capacity:
                    rejected[f"block capacity < {args.min_capacity}"] += 1
                    continue
                c.update(block_panos=n, block_sequences=seqs, block_capacity=cap)
                chosen.append(c)
                taken[country] += 1
                progress = True
                break
        if not progress:
            break
        print(f"  {len(chosen)} regions chosen", end="\r", flush=True)

    out = pd.DataFrame(chosen)
    out["region_id"] = [
        f"{c.lower()}_{k.lower()}_{slug(n)}"
        for c, k, n in zip(out["continent"], out["country"], out["name"], strict=True)
    ]
    dup = out["region_id"].duplicated(keep=False)
    out.loc[dup, "region_id"] += "_" + out.loc[dup].groupby("region_id").cumcount().astype(str)
    write_plan(out, resolve(args.out), args, rejected)
    print(f"\n{len(out)} regions in {out['country'].nunique()} countries -> {args.out}")
    print(
        "per continent:",
        out["continent"].value_counts().reindex(CONTINENTS, fill_value=0).to_dict(),
    )
    print("rejected while picking:", dict(rejected))
    cap = out["block_capacity"]
    print(f"block capacity: min {cap.min()}, median {int(cap.median())}")


def write_plan(out: pd.DataFrame, path: Path, args, rejected: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# M2 regions, generated by scripts/plan_regions.py ({len(out)} regions, "
        f"{out['country'].nunique()} countries). Do not hand-edit; re-run the script.\n"
        f"# Settings: target {args.target}, min population {args.min_pop}, "
        f"<= {args.max_per_country} per country, centres >= {args.min_km:g} km apart, "
        f"full-block capacity >= {args.min_capacity}.\n"
        "# Capacity = panoramas the downloader could take from the 3x3 tile block (at most\n"
        "# max_images_per_sequence per capture sequence), counted from Mapillary coverage tiles.\n"
        "# City names and populations: GeoNames (geonames.org), CC BY 4.0.\n"
    )
    records = [
        {
            "region_id": r.region_id,
            "country": r.country,
            "continent": r.continent,
            "lat": round(float(r.lat), 5),
            "lon": round(float(r.lon), 5),
            "city": r.name,
            "population": int(r.population),
            "block_capacity": int(r.block_capacity),
        }
        for r in out.sort_values(
            ["continent", "country", "population"], ascending=[True, True, False]
        ).itertuples()
    ]
    path.write_text(
        header
        + yaml.safe_dump({"regions": records}, sort_keys=False, allow_unicode=True, width=200)
    )


if __name__ == "__main__":
    main()
