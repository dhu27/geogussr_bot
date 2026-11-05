"""Download a dataset into <data_root>/<name>/ (images.parquet + images/).

    uv run python scripts/download.py --name pilot_v1

M0 PLACEHOLDER: ``data.downloader.download`` still copies tests/fixtures/. The M1 downloader
replaces its body; this entry point and its output contract (core/schemas.py) stay the same.
See docs/DATA.md for the full download -> embed -> push path.
"""

from __future__ import annotations

import argparse

from core.config import load_config
from data.downloader import download
from data.hub import dataset_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="dataset name = version, e.g. pilot_v1")
    args = ap.parse_args()
    cfg = load_config("data")
    out = dataset_dir(args.name)
    if (out / "images.parquet").exists():
        raise SystemExit(f"{out} already has images.parquet; pick a new --name or delete it")
    fracs = {k: v for k, v in cfg["splits"].items() if k != "seed"}
    images = download(out, fracs, cfg["splits"]["seed"])
    print(f"{len(images)} images, {images['region_id'].nunique()} regions -> {out}")
    print(f"splits: {images['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
