"""Leakage-safe train/val/test splits (ground rule 2): by region AND by creator, never by image.

Regions that share a creator are merged into one group (union-find), and whole groups are
assigned to splits. So no region and no creator ever appears in two splits.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from core.schemas import SPLITS


def assign_splits(df: pd.DataFrame, fractions: dict[str, float], seed: int = 0) -> pd.Series:
    """Return a ``split`` Series aligned with ``df`` (needs region_id and creator_id)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for region in df["region_id"].unique():
        find(f"r:{region}")
    for creator, regions in df.groupby("creator_id")["region_id"].unique().items():
        for region in regions:
            union(f"c:{creator}", f"r:{region}")

    group_of_region = {r: find(f"r:{r}") for r in df["region_id"].unique()}
    group_sizes = df["region_id"].map(group_of_region).value_counts()

    # Deterministic shuffle by hash, then greedily fill splits by image count.
    def key(g: str) -> str:
        return hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()

    total = len(df)
    targets = {s: fractions.get(s, 0.0) * total for s in SPLITS}
    filled = dict.fromkeys(SPLITS, 0.0)
    split_of_group: dict[str, str] = {}
    for g in sorted(group_sizes.index, key=key):
        # Put the group where it leaves the largest remaining deficit relative to the target.
        s = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
        split_of_group[g] = s
        filled[s] += group_sizes[g]

    return df["region_id"].map(lambda r: split_of_group[group_of_region[r]]).astype("string")


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any region or creator appears in more than one split."""
    for col in ("region_id", "creator_id"):
        n = df.groupby(col)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise AssertionError(f"{col} in multiple splits: {list(bad.index[:5])}")
