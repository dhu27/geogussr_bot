"""Data hub logic, offline: sharding round trip and push/pull against an in-memory fake hub."""

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from core.config import FIXTURES_DIR
from data import hub
from data.dataset import load_images


def test_shards_roundtrip(tmp_path, fixture_images):
    paths = sorted(fixture_images["path"])
    shards = list(hub.iter_shards(paths, size=10))
    assert [i for i, _ in shards] == list(range(len(shards)))
    assert sum(len(p) for _, p in shards) == len(paths)
    dest = tmp_path / "out"
    for idx, rels in shards:
        hub.write_shard(FIXTURES_DIR, rels, tmp_path / f"s{idx}.tar")
        hub.extract_shard(tmp_path / f"s{idx}.tar", dest)
    for rel in paths:
        assert (dest / rel).read_bytes() == (FIXTURES_DIR / rel).read_bytes()


def test_extract_refuses_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar"
    (tmp_path / "x").write_text("x")
    with tarfile.open(evil, "w") as tar:
        tar.add(tmp_path / "x", arcname="../escaped.txt")
    with pytest.raises(tarfile.TarError):
        hub.extract_shard(evil, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


def test_missing_hub_repo_gives_clear_error(monkeypatch):
    monkeypatch.setattr(hub, "load_config", lambda m: {"hub_repo": None})
    with pytest.raises(hub.HubError, match="docs/DATA.md"):
        hub.hub_repo()


class FakeHub:
    """Just enough of HfApi / snapshot_download, backed by a local directory."""

    def __init__(self, root: Path):
        self.root, self.token, self.uploads = root, None, 0

    def file_exists(self, repo, path, repo_type=None):
        return (self.root / path).is_file()

    def upload_file(self, *, path_or_fileobj, path_in_repo, **_):
        out = self.root / path_in_repo
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path_or_fileobj, out)
        self.uploads += 1

    def upload_folder(self, *, folder_path, path_in_repo, **_):
        shutil.copytree(folder_path, self.root / path_in_repo, dirs_exist_ok=True)

    def snapshot_download(self, repo, *, allow_patterns, local_dir, **_):
        name = allow_patterns[0].split("/")[0]
        shutil.copytree(self.root / name, Path(local_dir) / name, dirs_exist_ok=True)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    import huggingface_hub

    fake = FakeHub(tmp_path / "hub")
    local = tmp_path / "local"
    monkeypatch.setattr(hub, "_api", lambda: fake)
    monkeypatch.setattr(hub, "hub_repo", lambda: "org/repo")
    monkeypatch.setattr(hub, "data_root", lambda: local)
    monkeypatch.setattr(hub, "SHARD_SIZE", 10)  # 44 fixtures -> 5 shards
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    shutil.copytree(FIXTURES_DIR, local / "smoke", ignore=shutil.ignore_patterns("ATTR*"))
    return fake, local


def test_push_then_pull(fake_hub):
    fake, local = fake_hub
    manifest = hub.push("smoke", log=lambda *_: None)
    assert manifest["n_images"] == len(load_images(local / "smoke"))
    assert set(manifest["files"]) == {"images.parquet", "views.parquet", "embeddings.npy"}
    assert (fake.root / "smoke" / hub.MANIFEST).is_file()

    with pytest.raises(hub.HubError, match="immutable"):
        hub.push("smoke", log=lambda *_: None)

    shutil.rmtree(local / "smoke")
    dest = hub.pull("smoke", log=lambda *_: None)
    pulled = load_images(dest, check_files=True)
    assert len(pulled) == manifest["n_images"]
    assert json.loads((dest / hub.MANIFEST).read_text())["name"] == "smoke"
    assert (dest / "embeddings.npy").is_file()
    assert sorted(p.name for p in local.iterdir()) == ["smoke"]  # download area cleaned up


def test_push_resumes_after_interruption(fake_hub):
    fake, _ = fake_hub
    real_upload = fake.upload_file
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ConnectionError("network dropped")
        real_upload(**kw)

    fake.upload_file = flaky
    with pytest.raises(ConnectionError):
        hub.push("smoke", log=lambda *_: None)
    assert not (fake.root / "smoke" / hub.MANIFEST).exists()  # incomplete, not listed as done

    fake.upload_file = real_upload
    before = fake.uploads
    hub.push("smoke", log=lambda *_: None)
    n_shards = len(list((fake.root / "smoke" / "shards").glob("*.tar")))
    assert fake.uploads - before == n_shards - 2  # the two finished shards were skipped
