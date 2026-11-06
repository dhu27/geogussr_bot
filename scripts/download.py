"""Download a dataset into <data_root>/<name>/ (images.parquet + images/).

    uv run python scripts/download.py --name smoke_test                 # copies tests/fixtures/
    uv run python scripts/download.py --name pilot_v1 --source mapillary  # the real downloader

Try a couple of regions first, then the full list:

    uv run python scripts/download.py --name trial --source mapillary \\
        --regions eu_london,sa_lima --images-per-region 10

``--source mapillary`` needs MAPILLARY_TOKEN in .env. It resumes if interrupted: run the same
command again. See docs/DATA.md for the full download -> embed -> push path.
"""

from __future__ import annotations

import argparse
import copy

from core.config import load_config
from data.downloader import download
from data.hub import dataset_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="dataset name = version, e.g. pilot_v1")
    ap.add_argument(
        "--source",
        choices=["fixtures", "mapillary"],
        default="fixtures",
        help="fixtures (default): copy tests/fixtures/; mapillary: the real downloader",
    )
    ap.add_argument("--regions", help="mapillary only: comma-separated region_ids to download")
    ap.add_argument("--images-per-region", type=int, help="mapillary only: override the config")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config("data"))
    pilot = cfg.get("pilot") or {}
    if args.source == "fixtures" and (args.regions or args.images_per_region):
        raise SystemExit("--regions and --images-per-region need --source mapillary")
    if args.regions:
        wanted = [r.strip() for r in args.regions.split(",") if r.strip()]
        known = {r["region_id"] for r in pilot.get("regions", [])}
        unknown = [r for r in wanted if r not in known]
        if unknown:
            raise SystemExit(
                f"unknown region_id(s) {unknown}; see pilot.regions in configs/data.yaml"
            )
        pilot["regions"] = [r for r in pilot["regions"] if r["region_id"] in wanted]
    if args.images_per_region:
        pilot["images_per_region"] = args.images_per_region

    out = dataset_dir(args.name)
    if (out / "images.parquet").exists():
        raise SystemExit(f"{out} already has images.parquet; pick a new --name or delete it")
    fracs = {k: v for k, v in cfg["splits"].items() if k != "seed"}
    images = download(out, fracs, cfg["splits"]["seed"], source=args.source, cfg=cfg)
    print(f"{len(images)} images, {images['region_id'].nunique()} regions -> {out}")
    print(f"splits: {images['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
