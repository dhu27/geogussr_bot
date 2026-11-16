"""Crop every panorama at the 8 standard headings and embed the views (one-time pass).

    uv run python scripts/embed.py --name pilot_v2_sc   # encoder from configs/env.yaml
    uv sync --extra ml                                   # StreetCLIP needs torch + transformers
    uv run --extra ml python scripts/embed.py --name pilot_v2_sc --encoder streetclip

Writes views.parquet + embeddings.npy next to images.parquet in <data_root>/<name>/.
Interrupted? Run the same command again: finished chunks are kept and reused.
StreetCLIP speed on an M3 Pro laptop: about 8 views/s (the pilot takes ~70 min).
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
    n_views = int(images["is_pano"].sum()) * 8
    print(f"embedding {n_views} views of {args.name} with {cfg['encoder']}", flush=True)
    t0 = time.monotonic()
    views, emb = embed_dataset(images, root, make_encoder(cfg), log=lambda m: print(m, flush=True))
    print(
        f"{len(views)} views embedded with {cfg['encoder']} in {time.monotonic() - t0:.0f}s"
        f" -> {root}"
    )


if __name__ == "__main__":
    main()
