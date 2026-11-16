"""Plan the M2 region list: ~500 regions balanced by country, each checked for panorama
coverage BEFORE the download, so dead regions are swapped out up front, not found mid-run.

    uv run python scripts/plan_regions.py                       # -> configs/regions/world_v1.yaml
    uv run python scripts/download.py --name world_v1 --source mapillary \\
        --regions-file configs/regions/world_v1.yaml

Steps:
1. Candidates: GeoNames cities (CC BY 4.0) with population >= --min-pop, the --per-country
   largest per country. Downloaded once into <data_root>/_geonames/.
2. Coverage check on each candidate's CENTRE zoom-14 tile only: its *capacity* is the
   panoramas the downloader could take there (at most max_images_per_sequence per capture
   sequence). Why only the centre: tiles are 1-4 MB each in cities, so checking whole 3x3 blocks
   for every candidate would mean ~25 GB of downloads. The centre is ~1/9 of the block, so a
   centre capacity >= --min-centre (40) leaves wide headroom for the 100-image target.
   Tiles go into the shared cache (<data_root>/_tiles_z14/), so the downloader reuses them.
3. Pick round-robin over countries (each round, every country takes its largest city that
   passes), at most --max-per-country per country, centres at least --min-km apart so two
   regions' tile blocks never overlap, until --target regions. The default target (550) is
   10% over 500 so regions that come up short in the download still leave ~500.
"""

from __future__ import annotations

import argparse
import io
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
from data.downloader import tile_cache_dir
from data.hub import data_root
from data.tiles import MapillaryTiles, TileError, tile_for

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
        cities, sep="\t", header=None, usecols=list(CITY_COLS), quoting=3,
        keep_default_na=False, dtype=str,
    ).rename(columns=CITY_COLS)  # fmt: skip
    df[["lat", "lon"]] = df[["lat", "lon"]].astype(float)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    rows = [ln.split("\t") for ln in countries.read_text().splitlines() if ln and ln[0] != "#"]
    continent = {r[0]: r[8] for r in rows if len(r) > 8}
    df["continent"] = df["country"].map(continent)
    keep = (
        df["feature_code"].str.startswith("PPL")
        & (df["population"] >= min_pop)
        & df["continent"].isin(CONTINENTS)
    )
    df = df[keep].sort_values(["country", "population"], ascending=[True, False])
    return df.groupby("country").head(per_country).reset_index(drop=True)


def slug(text: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_") or "x"


# --- coverage -------------------------------------------------------------------------------
def capacity(panos: list[dict], per_sequence: int) -> tuple[int, int, int]:
    """(panoramas, sequences, capacity = sum over sequences of min(count, per_sequence))."""
    per_seq = Counter(p["sequence_id"] for p in {p["id"]: p for p in panos}.values())
    return sum(per_seq.values()), len(per_seq), sum(min(n, per_sequence) for n in per_seq.values())


def centre_capacities(
    tiles: list[tuple[int, int]], z: int, cache: Path, per_seq: int, workers: int
) -> list[int]:
    """Capacity of each tile; -1 if it could not be fetched. One HTTP session per thread."""
    local = threading.local()
    lock, done = threading.Lock(), [0]

    def one(t: tuple[int, int]) -> int:
        if not hasattr(local, "tiles"):
            local.tiles = MapillaryTiles(cache_dir=cache)
        try:
            cap = capacity(local.tiles.fetch(z, *t), per_seq)[2]
        except TileError:
            cap = -1
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  checked {done[0]}/{len(tiles)} centre tiles", flush=True)
        return cap

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, tiles))


def pick(passed: pd.DataFrame, target: int, max_per_country: int, min_km: float) -> list[dict]:
    """Round-robin over countries (biggest total population first), largest city first."""
    queues = {c: g.to_dict("records") for c, g in passed.groupby("country")}
    order = sorted(queues, key=lambda c: -sum(r["population"] for r in queues[c]))
    chosen: list[dict] = []
    taken: dict[str, int] = defaultdict(int)
    while len(chosen) < target:
        added = False
        for country in order:
            if len(chosen) >= target:
                break
            q = queues[country]
            while q and taken[country] < max_per_country:
                c = q.pop(0)
                far = all(
                    haversine_km(c["lat"], c["lon"], o["lat"], o["lon"]) >= min_km for o in chosen
                )
                if far:
                    chosen.append(c)
                    taken[country] += 1
                    added = True
                    break
        if not added:
            break
    return chosen


# --- main -----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", type=int, default=550)
    ap.add_argument("--min-pop", type=int, default=50_000)
    ap.add_argument("--per-country", type=int, default=10, help="candidate cities per country")
    ap.add_argument("--max-per-country", type=int, default=10)
    ap.add_argument("--min-km", type=float, default=25.0, help="minimum distance between centres")
    ap.add_argument("--min-centre", type=int, default=40, help="centre-tile capacity to keep")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="configs/regions/world_v1.yaml")
    args = ap.parse_args()

    cfg = load_config("data")
    z = int(cfg["pilot"].get("tile_zoom", 14))
    per_seq = int(cfg["max_images_per_sequence"])

    geonames = fetch_geonames(data_root() / "_geonames")
    cand = load_candidates(*geonames, args.min_pop, args.per_country)
    print(f"{len(cand)} candidate cities in {cand['country'].nunique()} countries", flush=True)
    cand["tile"] = [tile_for(la, lo, z) for la, lo in zip(cand["lat"], cand["lon"], strict=True)]
    cand["centre_capacity"] = centre_capacities(
        cand["tile"].tolist(), z, tile_cache_dir(cfg), per_seq, args.workers
    )
    failed = int((cand["centre_capacity"] < 0).sum())
    passed = cand[cand["centre_capacity"] >= args.min_centre]
    print(
        f"{len(passed)} candidates in {passed['country'].nunique()} countries have centre "
        f"capacity >= {args.min_centre} ({failed} tiles failed to load)"
    )

    out = pd.DataFrame(pick(passed, args.target, args.max_per_country, args.min_km))
    out["region_id"] = [
        f"{c.lower()}_{k.lower()}_{slug(n)}"
        for c, k, n in zip(out["continent"], out["country"], out["name"], strict=True)
    ]
    dup = out["region_id"].duplicated(keep=False)
    out.loc[dup, "region_id"] += "_" + out.loc[dup].groupby("region_id").cumcount().astype(str)
    write_plan(out, resolve(args.out), args)

    per_cont = out["continent"].value_counts().reindex(CONTINENTS, fill_value=0)
    cap = out["centre_capacity"]
    print(f"\n{len(out)} regions in {out['country'].nunique()} countries -> {args.out}")
    print("per continent:", per_cont.to_dict())
    print(f"centre capacity: min {cap.min()}, median {int(cap.median())}")
    print("most regions per country:", out["country"].value_counts().head(5).to_dict())


def write_plan(out: pd.DataFrame, path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# M2 regions, generated by scripts/plan_regions.py ({len(out)} regions, "
        f"{out['country'].nunique()} countries). Do not hand-edit; re-run the script.\n"
        f"# Settings: target {args.target}, population >= {args.min_pop}, the "
        f"{args.per_country} largest cities per country as candidates, <= "
        f"{args.max_per_country} per country,\n# centres >= {args.min_km:g} km apart, "
        f"centre-tile capacity >= {args.min_centre}.\n"
        "# centre_capacity = panoramas the downloader could take from the centre zoom-14 tile\n"
        "# alone (at most max_images_per_sequence per sequence); the download uses the 3x3 block.\n"
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
            "centre_capacity": int(r.centre_capacity),
        }  # fmt: skip
        for r in out.sort_values(
            ["continent", "country", "population"], ascending=[True, True, False]
        ).itertuples()
    ]
    body = yaml.safe_dump({"regions": records}, sort_keys=False, allow_unicode=True, width=200)
    path.write_text(header + body)


if __name__ == "__main__":
    main()
