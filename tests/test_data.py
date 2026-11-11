import numpy as np
import pandas as pd
import pytest

from core.config import FIXTURES_DIR
from core.schemas import SchemaError, validate_images
from core.view_spec import EMBED_DIM, N_HEADINGS
from data.mapillary import to_row
from data.splits import assign_splits, check_no_leakage

FRACS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_fixture_manifest(fixture_images):
    df = fixture_images
    assert 40 <= len(df) <= 60
    assert df["is_pano"].all()
    assert df["region_id"].nunique() >= 20
    assert set(df["continent"]) == {"AF", "AS", "EU", "NA", "OC", "SA"}
    assert set(df["split"]) == {"train", "val", "test"}
    check_no_leakage(df)


def test_fixture_attribution_lists_every_image(fixture_images):
    text = (FIXTURES_DIR / "ATTRIBUTION.md").read_text()
    assert "CC BY-SA" in text
    missing = [i for i in fixture_images["image_id"] if i not in text]
    assert not missing


def test_fixture_views_and_embeddings(fixture_images, fixture_views):
    views, emb = fixture_views
    assert len(views) == N_HEADINGS * len(fixture_images)
    assert emb.shape == (len(views), EMBED_DIM) and emb.dtype == np.float16
    assert set(views["image_id"]) == set(fixture_images["image_id"])


def test_schema_rejects_bad_rows(fixture_images):
    bad = fixture_images.copy()
    bad.loc[0, "split"] = "holdout"
    with pytest.raises(SchemaError):
        validate_images(bad)
    with pytest.raises(SchemaError):
        validate_images(fixture_images.drop(columns=["creator_id"]))


def test_mapillary_record_to_row():
    rec = {
        "id": "123",
        "sequence": "seqA",
        "creator": {"id": "77", "username": "alice"},
        "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
        "computed_geometry": {"type": "Point", "coordinates": [2.1, 48.1]},
        "captured_at": 1700000000000,
        "is_pano": True,
        "compass_angle": 10.0,
        "computed_compass_angle": 370.5,
        "quality_score": None,
    }
    row = to_row(rec)
    assert (row["lat"], row["lon"]) == (48.1, 2.1)  # computed_geometry preferred
    assert row["compass_angle"] == pytest.approx(10.5)  # computed preferred, wrapped
    assert row["creator_id"] == "77" and row["sequence_id"] == "seqA"
    assert np.isnan(row["quality_score"])
    del rec["computed_geometry"], rec["computed_compass_angle"]
    row = to_row(rec)
    assert (row["lat"], row["lon"], row["compass_angle"]) == (48.0, 2.0, 10.0)


def test_splits_keep_regions_and_creators_apart():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "region_id": rng.integers(0, 120, n).astype(str),
            "creator_id": rng.integers(0, 300, n).astype(str),
        }
    )
    # A few prolific creators spanning many regions (the leakage case that matters).
    df.loc[rng.choice(n, 40, replace=False), "creator_id"] = "prolific"
    df["split"] = assign_splits(df, FRACS, seed=1)
    check_no_leakage(df)
    assert set(df["split"]) <= {"train", "val", "test"}


def test_splits_roughly_hit_targets():
    df = pd.DataFrame(
        {
            "region_id": [f"r{i // 10}" for i in range(1000)],
            "creator_id": [f"c{i // 5}" for i in range(1000)],
        }
    )
    share = assign_splits(df, FRACS).value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.05)
    assert share["test"] == pytest.approx(0.15, abs=0.05)


def test_splits_deterministic():
    df = pd.DataFrame({"region_id": list("aabbccddeeff"), "creator_id": list("xyzxyzuvwuvw")})
    assert assign_splits(df, FRACS, 3).tolist() == assign_splits(df, FRACS, 3).tolist()


def _pilot_like(seed: int) -> pd.DataFrame:
    """The pilot_v1 shape: 45 regions (EU 10, NA 9, SA 8, AS 8, AF 5, OC 5), 20-100 images each,
    a few creators per region."""
    rng = np.random.default_rng(seed)
    layout = {"EU": 10, "NA": 9, "SA": 8, "AS": 8, "AF": 5, "OC": 5}
    rows = []
    for cont, n_regions in layout.items():
        for r in range(n_regions):
            region = f"{cont}{r}"
            for i in range(int(rng.integers(20, 101))):
                rows.append((region, f"{region}_c{i % 4}", cont))
    return pd.DataFrame(rows, columns=["region_id", "creator_id", "continent"])


@pytest.mark.parametrize("seed", range(5))
def test_every_continent_is_in_every_split(seed):
    df = _pilot_like(seed)
    df["split"] = assign_splits(df, FRACS, seed=seed)
    check_no_leakage(df)
    regions = df.groupby(["continent", "split"])["region_id"].nunique().unstack(fill_value=0)
    assert (regions[["train", "val", "test"]] >= 1).all().all(), regions
    share = df["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.1)


def test_creator_spanning_continents_keeps_regions_together():
    df = _pilot_like(0)
    df.loc[df["region_id"].isin(["EU0", "AS0"]), "creator_id"] = "globetrotter"
    df["split"] = assign_splits(df, FRACS)
    check_no_leakage(df)  # EU0 and AS0 share a creator, so they must share a split
    assert (
        df.loc[df["region_id"] == "EU0", "split"].iloc[0]
        == (df.loc[df["region_id"] == "AS0", "split"].iloc[0])
    )


@pytest.mark.parametrize("seed", range(5))
def test_a_creator_chaining_many_regions_lands_in_train(seed):
    """The real pilot: one creator links 9 regions on 3 continents (21% of images). Processed in
    random order, that group once landed in test and train fell to 34%."""
    df = _pilot_like(seed)
    chained = ["SA0", "SA1", "SA2", "SA3", "SA4", "SA5", "NA0", "NA1", "AS0"]
    mask = df["region_id"].isin(chained) & (df.index % 2 == 0)  # shares some images per region
    df.loc[mask, "creator_id"] = "prolific"
    df["split"] = assign_splits(df, FRACS, seed=seed)
    check_no_leakage(df)
    assert df.loc[df["region_id"].isin(chained), "split"].eq("train").all()
    share = df["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.1), share
    assert share["test"] == pytest.approx(0.15, abs=0.07), share


def test_fixture_splits_cover_each_continent_with_three_regions(fixture_images):
    df = fixture_images.copy()
    df["split"] = assign_splits(df, FRACS)
    per_cont = df.groupby("continent")["region_id"].nunique()
    covered = df.groupby("continent")["split"].nunique()
    assert (covered[per_cont >= 3] == 3).all()
