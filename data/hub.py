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
      _models/<model_name>/              # trained guessers (model.json + weights), also immutable

Locally a dataset lives at ``dataset_dir(name)`` = <data_root>/<name>/ (see configs/data.yaml);
a model at runs/models/<model_name>/ (models/guesser/registry.py).
"""

from __future__ import annotations

import contextlib
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
HUB_MODELS = "_models"

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


@contextlib.contextmanager
def _plain_downloads():
    """Download over plain HTTPS instead of HF's Xet transfer. On some connections Xet stalls for
    many minutes and then fails ("CAS Client Error"), where plain HTTPS fetched the same 377 MB
    file in seconds (Nov 2026). Uploads are unaffected."""
    from huggingface_hub import constants

    old = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = old


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
        if name == HUB_MODELS:  # trained guessers, not a dataset (see push_model)
            continue
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


def pull(name: str, *, force: bool = False, tables_only: bool = False, log=print) -> Path:
    """Download ``name`` into ``dataset_dir(name)`` and verify it.

    ``tables_only``: just images.parquet, views.parquet and embeddings.npy, no image shards.
    Enough to train and evaluate guessers (~400 MB instead of ~3 GB for world_v1_sc); the game
    and harness need the images (a full pull later fetches them)."""
    from huggingface_hub import snapshot_download

    dest = dataset_dir(name)
    if (dest / MANIFEST).is_file() and not force:
        log(f"{dest} already pulled (use --force to re-download)")
        return dest
    api, repo = _api(), hub_repo()
    if not api.file_exists(repo, f"{name}/{MANIFEST}", repo_type=REPO_TYPE):
        raise HubError(f"'{name}' is not a complete dataset on {repo} (run `list`)")
    if tables_only:
        return _pull_tables(name, dest, api, repo, log)

    # Per-dataset download area: kept if the pull is interrupted (re-running resumes),
    # removed once the dataset is extracted and verified.
    cache = data_root() / "_hub_download" / name
    with _plain_downloads():
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
    shutil.rmtree(cache, ignore_errors=True)
    if not any(cache.parent.iterdir()):
        cache.parent.rmdir()
    log(f"pulled {name}: {len(images)} images -> {dest}")
    return dest


def _pull_tables(name: str, dest: Path, api, repo: str, log) -> Path:
    """The dataset's tables without its images. Writes no MANIFEST.json, so a later full pull
    still runs (MANIFEST.json marks a complete local copy)."""
    from huggingface_hub import hf_hub_download

    manifest = json.loads(_read_hub_file(api, repo, f"{name}/{MANIFEST}"))
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tables-{name}-") as tmp:
        for f in manifest["files"]:
            for attempt in range(3):
                try:
                    with _plain_downloads():
                        p = hf_hub_download(
                            repo, f"{name}/{f}", repo_type=REPO_TYPE, token=api.token,
                            local_dir=tmp,
                        )  # fmt: skip
                    break
                except Exception as e:  # network errors surface as several exception types
                    if attempt == 2:
                        raise HubError(f"could not download {name}/{f}: {e}") from e
                    log(f"  {f}: {type(e).__name__}, retrying")
                    time.sleep(5 * (attempt + 1))
            shutil.move(p, dest / f)
            log(f"  {f}")
    images = load_images(dest, check_files=False)
    if len(images) != manifest["n_images"]:
        raise HubError(f"pulled {len(images)} rows, manifest says {manifest['n_images']}")
    log(f"pulled {name} tables ({', '.join(manifest['files'])}) -> {dest}; no images")
    return dest


# --- models (trained guessers) --------------------------------------------------------------
def push_model(name: str, *, log=print) -> str:
    """Upload runs/models/<name>/ to <hub_repo>/_models/<name>/ in one commit. Model names are
    immutable like dataset names: refuses to overwrite (train again under a new name)."""
    from models.guesser.registry import MODEL_JSON, find_model, load_model

    folder = find_model(name)
    load_model(folder)  # refuse to share a model that does not load
    api, repo = _api(), hub_repo()
    if api.file_exists(repo, f"{HUB_MODELS}/{name}/{MODEL_JSON}", repo_type=REPO_TYPE):
        raise HubError(f"model '{name}' already exists on {repo}; models are immutable")
    api.upload_folder(
        repo_id=repo,
        repo_type=REPO_TYPE,
        folder_path=str(folder),
        path_in_repo=f"{HUB_MODELS}/{name}",
        commit_message=f"model {name}",
    )
    log(f"pushed model {name} -> {repo}/{HUB_MODELS}/{name}")
    return name


def pull_model(name: str, *, force: bool = False, log=print) -> Path:
    """Download <hub_repo>/_models/<name>/ into runs/models/<name>/."""
    from huggingface_hub import snapshot_download

    from models.guesser.registry import MODEL_JSON, model_dir

    dest = model_dir(name)
    if (dest / MODEL_JSON).is_file() and not force:
        log(f"{dest} already present (use --force to re-download)")
        return dest
    api, repo = _api(), hub_repo()
    if not api.file_exists(repo, f"{HUB_MODELS}/{name}/{MODEL_JSON}", repo_type=REPO_TYPE):
        raise HubError(f"no model '{name}' on {repo} (run `list-models`)")
    with tempfile.TemporaryDirectory(prefix=f"model-{name}-") as tmp:
        with _plain_downloads():
            snapshot_download(
                repo,
                repo_type=REPO_TYPE,
                allow_patterns=[f"{HUB_MODELS}/{name}/**"],
                local_dir=tmp,
                token=api.token,
            )
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / HUB_MODELS / name, dest)
    log(f"pulled model {name} -> {dest}")
    return dest


def list_models() -> list[dict]:
    """model.json of every model on the hub."""
    from huggingface_hub import RepoFolder

    from models.guesser.registry import MODEL_JSON

    api, repo = _api(), hub_repo()
    if not any(e.path == HUB_MODELS for e in api.list_repo_tree(repo, repo_type=REPO_TYPE)):
        return []  # no model pushed yet
    out = []
    for entry in api.list_repo_tree(repo, path_in_repo=HUB_MODELS, repo_type=REPO_TYPE):
        if isinstance(entry, RepoFolder):
            path = f"{entry.path}/{MODEL_JSON}"
            if api.file_exists(repo, path, repo_type=REPO_TYPE):
                out.append(json.loads(_read_hub_file(api, repo, path)))
    return out
