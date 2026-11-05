"""Crop every panorama at the 8 standard headings and embed the views (one-time pass).

    uv run python scripts/embed.py --name pilot_v1      # encoder from configs/env.yaml
    uv sync --extra ml
    uv run python scripts/embed.py --name pilot_v1 --encoder streetclip

Writes views.parquet + embeddings.npy next to images.parquet in <data_root>/<name>/.
"""

from __future__ import annotations

import argparse
import time

from core.config import load_config
from data.dataset import load_images
from data.hub import dataset_dir
from env.embed import embed_dataset
from env.encoders import make_encoder


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", choices=["fake", "streetclip"], help="override configs/env.yaml")
    args = ap.parse_args()
    cfg = load_config("env")
    if args.encoder:
        cfg["encoder"] = args.encoder
    root = dataset_dir(args.name)
    images = load_images(root)
    t0 = time.monotonic()
    views, emb = embed_dataset(images, root, make_encoder(cfg))
    print(
        f"{len(views)} views embedded with {cfg['encoder']} in {time.monotonic() - t0:.0f}s"
        f" -> {root}"
    )


if __name__ == "__main__":
    main()
