"""Leakage-safe train/val/test splits (ground rule 2): by region AND by creator, never by image.

Regions that share a creator are merged into one group (union-find), and whole groups are
assigned to splits. So no region and no creator ever appears in two splits.

When ``df`` has a ``continent`` column, splits are also balanced per continent: each continent's
groups are divided among train/val/test on their own, so every continent with three or more
groups is in every split and per-continent metrics (PLAN.md 6.5) exist on val and test. A
continent with two groups gets train + val; with one, train only. (Unstratified, the 45-region
pilot put no North American, South American or Oceanian region in test.)
"""

from __future__ import annotations

import hashlib

import pandas as pd

from core.schemas import SPLITS


def assign_splits(df: pd.DataFrame, fractions: dict[str, float], seed: int = 0) -> pd.Series:
    """Return a ``split`` Series aligned with ``df``.

    Needs ``region_id`` and ``creator_id``; uses ``continent`` to balance splits when present.
    """
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
    group = df["region_id"].map(group_of_region).rename("group")
    group_sizes = group.value_counts()

    # Stratum of a group = the continent holding most of its images (a creator who captured on
    # two continents merges their regions into one group). No continent column: one stratum.
    if "continent" in df.columns:
        counts = df.groupby([group, df["continent"]]).size().reset_index(name="n")
        counts = counts.sort_values(["n", "continent"], ascending=[False, True])
        stratum_of_group = counts.drop_duplicates("group").set_index("group")["continent"]
    else:
        stratum_of_group = pd.Series("all", index=group_sizes.index)

    # Greedy fill per stratum, LARGEST group first (seeded hash breaks ties between equal sizes).
    # Largest-first matters: shared creators can chain many regions into one group (in the pilot,
    # one creator links 9 regions, 21% of all images). Placed late, such a group lands in val or
    # test and wrecks the proportions; placed first, it goes to train.
    def key(g: str) -> tuple[int, str]:
        return (-int(group_sizes[g]), hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())

    split_of_group: dict[str, str] = {}
    for _, groups in stratum_of_group.groupby(stratum_of_group):
        sizes = group_sizes[groups.index]
        targets = {s: fractions.get(s, 0.0) * sizes.sum() for s in SPLITS}
        filled = dict.fromkeys(SPLITS, 0.0)
        for g in sorted(sizes.index, key=key):
            # The split with the largest remaining deficit relative to its target. Ties go to
            # SPLITS order, so a stratum's first three groups land in train, val, test.
            s = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
            split_of_group[g] = s
            filled[s] += sizes[g]

    return group.map(split_of_group).astype("string")


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any region or creator appears in more than one split."""
    for col in ("region_id", "creator_id"):
        n = df.groupby(col)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise AssertionError(f"{col} in multiple splits: {list(bad.index[:5])}")
