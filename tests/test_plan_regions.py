"""scripts/plan_regions.py: candidate loading and the capacity estimate (no network)."""

import importlib.util
import sys
from pathlib import Path

import yaml

from data.downloader import load_regions

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plan_regions", ROOT / "scripts" / "plan_regions.py")
plan_regions = importlib.util.module_from_spec(spec)
sys.modules["plan_regions"] = plan_regions
spec.loader.exec_module(plan_regions)


def _city(gid, name, lat, lon, fcode, cc, pop):
    cols = [""] * 19
    cols[0], cols[1], cols[4], cols[5], cols[7], cols[8], cols[14] = (
        str(gid),
        name,
        str(lat),
        str(lon),
        fcode,
        cc,
        str(pop),
    )
    return "\t".join(cols)


def test_load_candidates_filters_and_caps(tmp_path):
    cities = tmp_path / "cities.txt"
    cities.write_text(
        "\n".join(
            [
                _city(1, "Big", 1.0, 1.0, "PPLC", "FR", 2_000_000),
                _city(2, "Mid", 2.0, 2.0, "PPLA", "FR", 400_000),
                _city(3, "Small", 3.0, 3.0, "PPL", "FR", 60_000),
                _city(4, "Tiny", 4.0, 4.0, "PPL", "FR", 20_000),  # below min population
                _city(5, "Airport", 5.0, 5.0, "AIRP", "FR", 900_000),  # not a populated place
                _city(
                    6, "Base", -77.8, 166.7, "PPL", "AQ", 100_000
                ),  # Antarctica: not a continent we use
            ]
        )
        + "\n"
    )
    countries = tmp_path / "countries.txt"
    countries.write_text(
        "#ISO\tx\n"
        + "\n".join(
            "\t".join([cc, "", "", "", "", "", "", "", cont, ""])
            for cc, cont in [("FR", "EU"), ("AQ", "AN")]
        )
        + "\n"
    )
    c = plan_regions.load_candidates(cities, countries, min_pop=50_000, per_country=2)
    assert c["name"].tolist() == ["Big", "Mid"]  # largest two, filters applied
    assert set(c["continent"]) == {"EU"}


def test_capacity_caps_each_sequence():
    panos = [{"id": f"a{i}", "sequence_id": "A"} for i in range(50)]
    panos += [{"id": f"b{i}", "sequence_id": "B"} for i in range(5)]
    panos += [{"id": "a0", "sequence_id": "A"}]  # duplicate from a neighbouring tile
    assert plan_regions.capacity(panos, 20) == (55, 2, 25)


def test_slug_is_ascii_and_stable():
    assert plan_regions.slug("São Paulo") == "sao_paulo"
    assert plan_regions.slug("Ho Chi Minh City") == "ho_chi_minh_city"
    assert plan_regions.slug("北京") == "x"


def test_generated_plan_is_a_valid_region_list():
    plan = ROOT / "configs" / "regions" / "world_v1.yaml"
    if not plan.is_file():
        return  # not generated in this checkout
    regions = load_regions(yaml.safe_load(plan.read_text()))
    assert 300 <= len(regions) <= 600
    assert len({r["region_id"] for r in regions}) == len(regions)
    per_country: dict[str, int] = {}
    for r in regions:
        per_country[r["country"]] = per_country.get(r["country"], 0) + 1
    assert max(per_country.values()) <= 12


def test_tile_counter_disk_cache_skips_refetch(tmp_path, monkeypatch):
    calls = []

    class FakeTiles:
        def fetch(self, z, x, y):
            calls.append((x, y))
            return [{"id": "1", "sequence_id": "s"}, {"id": "2", "sequence_id": "s"}]

    monkeypatch.setattr(plan_regions, "MapillaryTiles", FakeTiles)
    first = plan_regions.TileCounter(14, tmp_path)
    assert len(first.panos(3, 4)) == 2
    second = plan_regions.TileCounter(14, tmp_path)  # a new run: memory cache is empty
    assert second.panos(3, 4) == [{"id": "1", "sequence_id": "s"}, {"id": "2", "sequence_id": "s"}]
    assert calls == [(3, 4)]
