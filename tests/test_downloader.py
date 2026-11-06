"""The real downloader, run against canned Mapillary-shaped responses. Never calls the network."""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import requests
from PIL import Image

from core.config import load_config
from core.schemas import CONTINENTS, validate_images
from data.dataset import load_images
from data.downloader import (
    download,
    download_mapillary,
    load_regions,
    select_candidates,
)
from data.splits import check_no_leakage
from data.tiles import TileError, tile_for

FRACS = {"train": 0.6, "val": 0.2, "test": 0.2}
LONDON = {
    "region_id": "eu_london",
    "country": "GB",
    "continent": "EU",
    "lat": 51.507,
    "lon": -0.128,
}
LIMA = {"region_id": "sa_lima", "country": "PE", "continent": "SA", "lat": -12.046, "lon": -77.043}
QUITO = {"region_id": "sa_quito", "country": "EC", "continent": "SA", "lat": -0.180, "lon": -78.468}
NYC = {"region_id": "na_nyc", "country": "US", "continent": "NA", "lat": 40.713, "lon": -74.006}


def centre_tile(region: dict) -> tuple[int, int]:
    return tile_for(region["lat"], region["lon"], 14)


def make_cfg(regions, per_region=5, cap=2, min_quality=0.3) -> dict:
    return {
        "image_width": 1024,
        "min_quality_score": min_quality,
        "max_images_per_sequence": cap,
        "pilot": {
            "images_per_region": per_region,
            "tile_zoom": 14,
            "tile_radius": 0,  # one tile per region keeps the fakes simple
            "workers": 2,
            "min_images_per_region": 1,
            "regions": regions,
        },
    }


def jpeg(w: int, h: int, colour=(120, 60, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG")
    return buf.getvalue()


def cand(image_id: int, seq: str, creator: str = "c1") -> dict:
    """What a coverage tile gives us (data.tiles.parse_image_layer output)."""
    return {
        "id": str(image_id),
        "sequence_id": seq,
        "creator_id": creator,
        "is_pano": True,
        "captured_at": 0,
    }


def record(image_id: int, seq: str, creator: str = "c1", *, pano=True, q=0.9, lat=51.5, lon=-0.1):
    """What MapillaryClient.image(id) returns (the fields in data.mapillary.IMAGE_FIELDS)."""
    return {
        "id": str(image_id),
        "sequence": {"id": seq},
        "creator": {"id": creator, "username": f"user_{creator}"},
        "computed_geometry": {"type": "Point", "coordinates": [lon, lat]},
        "geometry": {"type": "Point", "coordinates": [lon + 0.0001, lat]},
        "captured_at": 1_700_000_000_000 + image_id,
        "is_pano": pano,
        "compass_angle": 90.0,
        "computed_compass_angle": 91.0,
        "quality_score": q,
        "thumb_1024_url": f"https://cdn.example/{image_id}?sig=v1",
    }


class FakeTiles:
    def __init__(self, by_tile: dict, failing: set | None = None):
        self.by_tile, self.failing, self.calls = by_tile, failing or set(), []

    def fetch(self, z, x, y):
        self.calls.append((x, y))
        if (x, y) in self.failing:
            raise TileError(f"tile {z}/{x}/{y}: HTTP 500")
        return [dict(c) for c in self.by_tile.get((x, y), [])]


class FakeClient:
    def __init__(self, records: list[dict], sizes: dict[int, tuple[int, int]] | None = None):
        self.records = {r["id"]: r for r in records}
        self.sizes = sizes or {}
        self.image_calls: list[str] = []
        self.download_calls: list[str] = []

    def image(self, image_id: str) -> dict:
        self.image_calls.append(image_id)
        if image_id not in self.records:
            raise requests.HTTPError("404 Not Found")
        return copy.deepcopy(self.records[image_id])

    def download(self, url: str) -> bytes:
        self.download_calls.append(url)
        image_id = int(url.rsplit("/", 1)[1].split("?")[0])
        return jpeg(*self.sizes.get(image_id, (64, 32)))  # 64x32 = a 2:1 panorama


class NoNetwork:
    """Any call fails the test: used to prove a re-run needs neither the API nor the CDN."""

    def image(self, *_):
        raise AssertionError("API call during a resumed run")

    def download(self, *_):
        raise AssertionError("download during a resumed run")


@pytest.fixture
def london_world():
    """One region, 8 candidates once the cap (2 per sequence) is applied: 5 usable and 3 rejected,
    each for a different reason. Only S1 is over the cap, so the rejected ones always survive it."""
    cands = [
        cand(101, "S1"), cand(102, "S1"), cand(103, "S1"),  # 3 in one sequence (cap is 2)
        cand(201, "S2"), cand(202, "S2"),
        cand(301, "S3", "c2"), cand(302, "S3", "c2"),
        cand(303, "S4", "c2"),
        cand(401, "S5", "c3"),
    ]  # fmt: skip
    records = [
        record(101, "S1"), record(102, "S1"), record(103, "S1"),
        record(201, "S2"), record(202, "S2", q=0.1),  # low quality
        record(301, "S3", "c2", pano=False),  # details say it is not a panorama
        record(302, "S3", "c2"),  # square thumbnail (see sizes below)
        record(303, "S4", "c2", q=None),  # no quality score: kept
        record(401, "S5", "c3"),
    ]  # fmt: skip
    tiles = FakeTiles({centre_tile(LONDON): cands})
    client = FakeClient(records, sizes={302: (64, 64)})
    return tiles, client


def run(out_dir, regions, tiles, client, **cfg_kw):
    logs: list[str] = []
    df = download_mapillary(
        out_dir,
        FRACS,
        0,
        cfg=make_cfg(regions, **cfg_kw),
        client=client,
        tiles=tiles,
        log=logs.append,
    )
    return df, "\n".join(logs)


# --- filters, caps, output contract -----------------------------------------------------------
def test_filters_cap_and_output_contract(tmp_path, london_world):
    tiles, client = london_world
    df, log = run(tmp_path, [LONDON], tiles, client, per_region=50)

    ids = set(df["image_id"])
    assert len(df) == 5  # 2 of S1's 3 (the cap) + the three other usable images
    assert {"201", "303", "401"} <= ids
    assert not ids & {"202", "301", "302"}  # low quality, not a pano, not 2:1
    assert len(ids & {"101", "102", "103"}) == 2  # at most max_images_per_sequence
    assert "low quality" in log and "not a panorama" in log and "not 2:1" in log

    validate_images(df, tmp_path)  # schema + every file exists
    assert (df["region_id"] == "eu_london").all() and (df["continent"] == "EU").all()
    assert df["sequence_id"].str.startswith("S").all()  # sequence_id saved (Phase 2 needs it)
    assert df["lat"].notna().all() and df["lon"].notna().all()
    assert df.loc[df["image_id"] == "303", "quality_score"].isna().all()  # missing score kept
    for rel in df["path"]:
        with Image.open(tmp_path / rel) as im:
            assert im.width == 2 * im.height
    assert not list((tmp_path / "images").glob("*.part"))  # no half-written files
    assert (tmp_path / "images" / "302.jpg").exists() is False  # rejected images are not kept
    assert load_images(tmp_path).equals(df)  # what was returned is what was written


def test_stops_at_images_per_region(tmp_path, london_world):
    tiles, client = london_world
    df, _ = run(tmp_path, [LONDON], tiles, client, per_region=3)
    assert len(df) == 3
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 3  # nothing downloaded beyond that


def test_low_quality_threshold_is_configurable(tmp_path, london_world):
    tiles, client = london_world
    df, _ = run(tmp_path, [LONDON], tiles, client, per_region=50, min_quality=0.0)
    assert "202" in set(df["image_id"])  # 0.1 passes a threshold of 0


def test_same_seed_same_selection(tmp_path, london_world):
    tiles, client = london_world
    a, _ = run(tmp_path / "a", [LONDON], tiles, client, per_region=4)
    b, _ = run(tmp_path / "b", [LONDON], tiles, client, per_region=4)
    assert sorted(a["image_id"]) == sorted(b["image_id"])


# --- resumability and expiring URLs -------------------------------------------------------------
def test_rerun_resumes_without_network(tmp_path, london_world):
    tiles, client = london_world
    first, _ = run(tmp_path, [LONDON], tiles, client, per_region=50)
    again, _ = run(tmp_path, [LONDON], tiles, NoNetwork(), per_region=50)
    assert sorted(again["image_id"]) == sorted(first["image_id"])


def test_expired_thumbnail_url_gets_a_fresh_one(tmp_path, london_world):
    tiles, client = london_world
    first, _ = run(tmp_path, [LONDON], tiles, client, per_region=50)
    victim = first["image_id"].iloc[0]
    (tmp_path / "images" / f"{victim}.jpg").unlink()  # lose the file, keep the cached details

    class Expiring(FakeClient):
        def download(self, url):
            if url.endswith("sig=v1"):  # the URL saved in the cache has expired
                raise requests.HTTPError("403 Forbidden")
            return super().download(url)

    fresh = copy.deepcopy(client.records)
    for r in fresh.values():
        r["thumb_1024_url"] = r["thumb_1024_url"].replace("sig=v1", "sig=v2")
    new_client = Expiring(list(fresh.values()), sizes={302: (64, 64)})
    again, _ = run(tmp_path, [LONDON], tiles, new_client, per_region=50)
    assert (tmp_path / "images" / f"{victim}.jpg").is_file()
    assert sorted(again["image_id"]) == sorted(first["image_id"])


def test_partial_image_on_disk_is_redownloaded(tmp_path, london_world):
    tiles, client = london_world
    first, _ = run(tmp_path, [LONDON], tiles, client, per_region=50)
    victim = first["image_id"].iloc[0]
    (tmp_path / "images" / f"{victim}.jpg").write_bytes(b"not a jpeg")  # an interrupted download
    again, _ = run(tmp_path, [LONDON], tiles, client, per_region=50)
    with Image.open(tmp_path / "images" / f"{victim}.jpg") as im:
        assert im.width == 2 * im.height
    assert len(again) == len(first)


# --- regions, splits, failures ----------------------------------------------------------------
def three_region_world():
    by_tile, records, n = {}, [], 1000
    for k, reg in enumerate([LONDON, LIMA, NYC]):
        cands = []
        for j in range(6):
            n += 1
            creator = f"creator_{k}_{j % 2}"
            cands.append(cand(n, f"seq_{k}_{j % 3}", creator))
            records.append(record(n, f"seq_{k}_{j % 3}", creator, lat=reg["lat"], lon=reg["lon"]))
        by_tile[centre_tile(reg)] = cands
    return FakeTiles(by_tile), FakeClient(records)


def test_splits_by_region_and_creator(tmp_path):
    tiles, client = three_region_world()
    df, log = run(tmp_path, [LONDON, LIMA, NYC], tiles, client, per_region=6)
    assert df["region_id"].nunique() == 3
    check_no_leakage(df)  # no region or creator in two splits
    assert df.groupby("region_id")["split"].nunique().max() == 1
    assert "regions per continent" in log and "quality_score:" in log


def test_a_failing_region_is_logged_and_skipped(tmp_path):
    tiles, client = three_region_world()
    tiles.failing = {centre_tile(LIMA)}
    df, log = run(tmp_path, [LONDON, LIMA, NYC], tiles, client, per_region=6)
    assert set(df["region_id"]) == {"eu_london", "na_nyc"}
    assert "sa_lima" in log and "FAILED" in log and "1 region(s) failed" in log


def test_a_region_with_no_panoramas_is_logged_and_skipped(tmp_path):
    tiles, client = three_region_world()
    tiles.by_tile[centre_tile(NYC)] = []
    df, log = run(tmp_path, [LONDON, LIMA, NYC], tiles, client, per_region=6)
    assert "na_nyc" not in set(df["region_id"])
    assert "no panoramas" in log
    assert "no images at all on" in log and "'NA'" in log  # the continent warning lists NA


def test_stops_early_when_everything_fails(tmp_path):
    tiles, client = three_region_world()
    tiles.failing = {centre_tile(r) for r in (LONDON, LIMA, NYC)}
    with pytest.raises(RuntimeError, match="stopping early"):
        run(tmp_path, [LONDON, LIMA, NYC, QUITO], tiles, client, per_region=6)
    assert len(tiles.calls) == 3  # it did not keep going through the whole list


def test_bad_token_shows_up_as_api_errors_and_stops(tmp_path):
    tiles, _ = three_region_world()

    class Unauthorized(FakeClient):
        def image(self, image_id):
            err = requests.HTTPError("401 Unauthorized")
            err.response = type("R", (), {"status_code": 401})()
            raise err

    with pytest.raises(RuntimeError, match="stopping early") as exc:
        run(tmp_path, [LONDON, LIMA, NYC], tiles, Unauthorized([]), per_region=6)
    assert "HTTPError 401" in str(exc.value)


# --- pure helpers -----------------------------------------------------------------------------
def test_select_candidates_caps_and_spreads():
    cands = [cand(i, f"s{i // 10}") for i in range(30)]  # 3 sequences x 10
    out = select_candidates(cands, 4, np.random.default_rng(0))
    assert len(out) == 12  # 4 per sequence
    assert len({c["sequence_id"] for c in out[:3]}) == 3  # the first picks span all sequences
    assert len({c["id"] for c in out}) == 12
    again = select_candidates(cands, 4, np.random.default_rng(0))
    assert [c["id"] for c in out] == [c["id"] for c in again]


def test_select_candidates_empty():
    assert select_candidates([], 5, np.random.default_rng(0)) == []


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"region_id": "x", "country": "GB", "continent": "EU", "lat": 1}, "missing"),
        ({**LONDON, "continent": "XX"}, "continent"),
        ({**LONDON, "lat": 99}, "out of range"),
    ],
)
def test_load_regions_rejects_bad_entries(bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_regions({"regions": [bad]})


def test_load_regions_rejects_duplicates_and_empty():
    with pytest.raises(ValueError, match="duplicate"):
        load_regions({"regions": [LONDON, dict(LONDON)]})
    with pytest.raises(ValueError, match="no pilot.regions"):
        load_regions({})


def test_the_real_pilot_region_list_is_sane():
    pilot = load_config("data")["pilot"]
    regions = load_regions(pilot)
    assert 45 <= len(regions) <= 55  # "about 50 pilot regions"
    assert {r["continent"] for r in regions} == set(CONTINENTS)  # all six continents
    per_country: dict[str, int] = {}
    for r in regions:
        per_country[r["country"]] = per_country.get(r["country"], 0) + 1
    assert max(per_country.values()) <= 2  # balanced by country
    centres = {centre_tile(r) for r in regions}
    assert len(centres) == len(regions)  # no two regions share a tile


# --- the entry point keeps its old behaviour ----------------------------------------------------
def test_fixtures_source_still_copies_the_fixtures(tmp_path):
    df = download(tmp_path / "d", FRACS, 0)  # what run_e2e.py and CI call
    assert 40 <= len(df) <= 60 and df["is_pano"].all()
    check_no_leakage(df)


def test_unknown_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown source"):
        download(tmp_path, FRACS, 0, source="flickr")
