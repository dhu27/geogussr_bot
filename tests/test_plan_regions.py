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
    assert max(per_country.values()) <= 10


def test_pick_round_robins_countries_caps_and_spaces():
    import pandas as pd

    rows = [  # FR: 3 cities, one 10 km from Paris; DE: 2 cities; LU: 1
        ("Paris", "FR", 48.86, 2.35, 2_100_000), ("Near Paris", "FR", 48.90, 2.45, 900_000),
        ("Lyon", "FR", 45.76, 4.84, 500_000), ("Berlin", "DE", 52.52, 13.40, 3_600_000),
        ("Hamburg", "DE", 53.55, 9.99, 1_800_000), ("Luxembourg", "LU", 49.61, 6.13, 120_000),
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["name", "country", "lat", "lon", "population"])
    chosen = [c["name"] for c in plan_regions.pick(df, target=10, max_per_country=2, min_km=25)]
    # one per country per round, countries by total candidate population (DE 5.4M > FR 3.5M)
    assert chosen[:3] == ["Berlin", "Paris", "Luxembourg"]
    assert "Near Paris" not in chosen  # too close to Paris
    assert sorted(chosen) == ["Berlin", "Hamburg", "Luxembourg", "Lyon", "Paris"]
    assert len(plan_regions.pick(df, target=2, max_per_country=2, min_km=25)) == 2
