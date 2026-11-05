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
