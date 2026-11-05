"""Share datasets through one private Hugging Face dataset repo (ground rule 3: no data in git).

Layout on the hub (one folder per dataset; a dataset name is a version and is never rewritten):

    <hub_repo>/
      README.md                     # dataset card (license: CC BY-SA 4.0)
      <name>/
        images.parquet
        views.parquet, embeddings.npy   # if embedded
        edges.parquet                    # Phase 2, if built
        shards/images-00000.tar ...      # images/<id>.jpg packed ~1000 per tar
        MANIFEST.json                    # written LAST: its presence = dataset complete

Locally a dataset lives at ``dataset_dir(name)`` = <data_root>/<name>/ (see configs/data.yaml).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from core import view_spec
from core.config import REPO_ROOT, load_config, load_dotenv, resolve
from data.dataset import load_images
from data.splits import check_no_leakage

SHARD_SIZE = 1000
MANIFEST = "MANIFEST.json"
DATA_FILES = ("images.parquet", "views.parquet", "embeddings.npy", "edges.parquet")
REPO_TYPE = "dataset"

DATASET_CARD = """---
license: cc-by-sa-4.0
viewer: false
---
# GeoGuessr RL agent: Mapillary panoramas

Private working data for the GeoGuessr RL project. Each top-level folder is one immutable
dataset version (`images.parquet`, image tar shards, optional embeddings). Pull with
`uv run python scripts/data_hub.py pull <name>`.

Imagery from [Mapillary](https://www.mapillary.com), licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Creator ids are in
`images.parquet` (`creator_id`); image pages are `https://www.mapillary.com/app/?pKey=<image_id>`.
"""


class HubError(RuntimeError):
    pass


# --- local paths ------------------------------------------------------------------------
def data_root() -> Path:
    return resolve(load_config("data")["data_root"])


def dataset_dir(name: str) -> Path:
    return data_root() / name


# --- hub access ---------------------------------------------------------------------------
def hub_repo() -> str:
    repo = load_config("data").get("hub_repo")
    if not repo:
        raise HubError(
            "configs/data.yaml has no hub_repo. Set it to '<hf-org>/<repo-name>' "
            "(see docs/DATA.md, 'One-time setup')."
        )
    return repo


def _api():
    load_dotenv()  # HF_TOKEN may live in .env
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN") or None)  # None -> `hf auth login` token


def create_repo() -> str:
    """Create the private dataset repo (no-op if it exists) and its dataset card."""
    api, repo = _api(), hub_repo()
    url = api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
    if not api.file_exists(repo, "README.md", repo_type=REPO_TYPE):
        api.upload_file(
            path_or_fileobj=DATASET_CARD.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type=REPO_TYPE,
            commit_message="Add dataset card",
        )
    return str(url)


def list_datasets() -> list[dict]:
    """Complete datasets on the hub, with their manifests."""
    from huggingface_hub import RepoFolder

    api, repo = _api(), hub_repo()
    out = []
    for entry in api.list_repo_tree(repo, repo_type=REPO_TYPE):
        if not isinstance(entry, RepoFolder):  # files at the root (README.md)
            continue
        name = entry.path
        if api.file_exists(repo, f"{name}/{MANIFEST}", repo_type=REPO_TYPE):
            out.append(json.loads(_read_hub_file(api, repo, f"{name}/{MANIFEST}")))
        else:
            out.append({"name": name, "incomplete": True})
    return out


def _read_hub_file(api, repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, path, repo_type=REPO_TYPE, token=api.token)
    return Path(p).read_text()


# --- sharding (pure, tested offline) ------------------------------------------------------
def iter_shards(paths: list[str], size: int | None = None) -> Iterator[tuple[int, list[str]]]:
    size = size or SHARD_SIZE
    for i in range(0, len(paths), size):
        yield i // size, paths[i : i + size]


def write_shard(root: Path, rel_paths: list[str], out: Path) -> None:
    """Pack ``root/rel`` files into an uncompressed tar (JPEGs are already compressed)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tar.partial")
    with tarfile.open(tmp, "w") as tar:
        for rel in rel_paths:
            tar.add(root / rel, arcname=rel)
    tmp.replace(out)


def extract_shard(shard: Path, dest: Path) -> int:
    """Extract one shard into ``dest``; refuses paths that escape it (tarfile 'data' filter)."""
    with tarfile.open(shard) as tar:
        members = tar.getmembers()
        tar.extractall(dest, filter="data")
    return len(members)


def build_manifest(name: str, images, root: Path, n_shards: int) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit": commit,
        "n_images": int(len(images)),
        "n_panos": int(images["is_pano"].sum()),
        "n_regions": int(images["region_id"].nunique()),
        "splits": {k: int(v) for k, v in images["split"].value_counts().items()},
        "n_shards": n_shards,
        "files": [f for f in DATA_FILES if (root / f).is_file()],
        "view_spec": current_view_spec(),
    }


def current_view_spec() -> dict:
    return {
        "encoder": view_spec.ENCODER_NAME,
        "embed_dim": view_spec.EMBED_DIM,
        "view_size": view_spec.VIEW_SIZE,
        "hfov_deg": view_spec.HFOV_DEG,
        "headings_deg": list(view_spec.HEADINGS_DEG),
    }


# --- push / pull --------------------------------------------------------------------------
def push(name: str, *, log=print) -> dict:
    """Validate and upload ``dataset_dir(name)``. Resumable: re-running skips shards that are
    already on the hub. Refuses to overwrite a completed dataset (bump the name instead)."""
    root = dataset_dir(name)
    if not (root / "images.parquet").is_file():
        raise HubError(f"{root}/images.parquet not found")
    images = load_images(root, check_files=True)  # schema + every image file present
    check_no_leakage(images)

    api, repo = _api(), hub_repo()
    if api.file_exists(repo, f"{name}/{MANIFEST}", repo_type=REPO_TYPE):
        raise HubError(f"'{name}' already exists on {repo}; datasets are immutable, use a new name")

    paths = sorted(images["path"].tolist())
    shards = list(iter_shards(paths))
    with tempfile.TemporaryDirectory(prefix=f"hub-{name}-") as tmp:
        for idx, rels in shards:
            hub_path = f"{name}/shards/images-{idx:05d}.tar"
            if api.file_exists(repo, hub_path, repo_type=REPO_TYPE):
                log(f"  shard {idx + 1}/{len(shards)} already uploaded, skipping")
                continue
            local = Path(tmp) / f"images-{idx:05d}.tar"
            write_shard(root, rels, local)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=hub_path,
                repo_id=repo,
                repo_type=REPO_TYPE,
                commit_message=f"{name}: shard {idx}",
            )
            local.unlink()
            log(f"  shard {idx + 1}/{len(shards)} uploaded ({len(rels)} images)")

        stage = Path(tmp) / "meta"
        stage.mkdir()
        manifest = build_manifest(name, images, root, len(shards))
        for f in manifest["files"]:
            shutil.copy2(root / f, stage / f)
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2))
        api.upload_folder(
            repo_id=repo,
            repo_type=REPO_TYPE,
            folder_path=str(stage),
            path_in_repo=name,
            commit_message=f"{name}: tables + manifest",
        )
    log(f"pushed {name}: {manifest['n_images']} images in {len(shards)} shards -> {repo}")
    return manifest


def pull(name: str, *, force: bool = False, log=print) -> Path:
    """Download ``name`` into ``dataset_dir(name)`` and verify it."""
    from huggingface_hub import snapshot_download

    dest = dataset_dir(name)
    if (dest / MANIFEST).is_file() and not force:
        log(f"{dest} already pulled (use --force to re-download)")
        return dest
    api, repo = _api(), hub_repo()
    if not api.file_exists(repo, f"{name}/{MANIFEST}", repo_type=REPO_TYPE):
        raise HubError(f"'{name}' is not a complete dataset on {repo} (run `list`)")

    cache = data_root() / "_hub_download"
    snapshot_download(
        repo,
        repo_type=REPO_TYPE,
        allow_patterns=[f"{name}/*", f"{name}/**"],
        local_dir=cache,
        token=api.token,
    )
    src = cache / name
    dest.mkdir(parents=True, exist_ok=True)
    shards = sorted((src / "shards").glob("*.tar"))
    for i, shard in enumerate(shards):
        extract_shard(shard, dest)
        log(f"  extracted shard {i + 1}/{len(shards)}")
    manifest = json.loads((src / MANIFEST).read_text())
    for f in manifest["files"]:
        shutil.copy2(src / f, dest / f)

    images = load_images(dest, check_files=True)
    if len(images) != manifest["n_images"]:
        raise HubError(f"pulled {len(images)} images, manifest says {manifest['n_images']}")
    current = current_view_spec()
    mismatch = {k: v for k, v in manifest["view_spec"].items() if current.get(k) != v}
    if "embeddings.npy" in manifest["files"] and mismatch:
        log(f"WARNING: embeddings were made with a different view spec: {mismatch}")
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2))  # marks the pull complete
    shutil.rmtree(src, ignore_errors=True)
    log(f"pulled {name}: {len(images)} images -> {dest}")
    return dest
