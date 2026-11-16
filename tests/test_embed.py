"""The embedding pass: resumable chunks, and the real StreetCLIP encoder (only where torch is)."""

import numpy as np
import pytest

from core.config import FIXTURES_DIR
from core.view_spec import EMBED_DIM, VIEW_SIZE
from env.embed import PARTS_DIR, embed_dataset
from env.encoders import FakeEncoder


class CountingEncoder(FakeEncoder):
    def __init__(self, fail_after_calls: int | None = None):
        self.calls, self.fail_after_calls = 0, fail_after_calls

    def embed(self, images):
        if self.fail_after_calls is not None and self.calls >= self.fail_after_calls:
            raise KeyboardInterrupt("simulated interruption")
        self.calls += 1
        return super().embed(images)


class OtherEncoder(CountingEncoder):
    """Same output as FakeEncoder but a different identity: its chunks must not be reused."""


def test_resumes_after_interruption_with_identical_output(tmp_path, fixture_images):
    ref_v, ref_e = embed_dataset(
        fixture_images, FIXTURES_DIR, FakeEncoder(), tmp_path / "ref", chunk_panos=10
    )
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        embed_dataset(
            fixture_images, FIXTURES_DIR, CountingEncoder(fail_after_calls=2), out, chunk_panos=10
        )
    assert len(list((out / PARTS_DIR).glob("part-*.npz"))) == 2  # two chunks were saved

    again = CountingEncoder()
    v, e = embed_dataset(fixture_images, FIXTURES_DIR, again, out, chunk_panos=10)
    n_chunks = -(-len(fixture_images) // 10)
    assert again.calls == n_chunks - 2  # only the unfinished chunks were embedded
    assert np.array_equal(e, ref_e) and v.equals(ref_v)
    assert not (out / PARTS_DIR).exists()  # cleaned up after a complete run


def test_chunks_from_another_encoder_are_not_reused(tmp_path, fixture_images):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        embed_dataset(
            fixture_images, FIXTURES_DIR, CountingEncoder(fail_after_calls=2), out, chunk_panos=10
        )
    other = OtherEncoder()
    embed_dataset(fixture_images, FIXTURES_DIR, other, out, chunk_panos=10)
    assert other.calls == -(-len(fixture_images) // 10)  # every chunk redone


def test_corrupt_chunk_is_redone(tmp_path, fixture_images):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        embed_dataset(
            fixture_images, FIXTURES_DIR, CountingEncoder(fail_after_calls=1), out, chunk_panos=10
        )
    (out / PARTS_DIR / "part-00000.npz").write_bytes(b"not an npz")
    again = CountingEncoder()
    embed_dataset(fixture_images, FIXTURES_DIR, again, out, chunk_panos=10)
    assert again.calls == -(-len(fixture_images) // 10)


@pytest.mark.ml
def test_streetclip_encoder_outputs_unit_vectors():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from env.encoders import StreetCLIPEncoder

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (VIEW_SIZE, VIEW_SIZE, 3), dtype=np.uint8) for _ in range(3)]
    e = StreetCLIPEncoder(batch_size=2).embed(imgs)  # batch of 2 + a batch of 1
    assert e.shape == (3, EMBED_DIM) and e.dtype == np.float32
    assert np.all(np.isfinite(e)) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    again = StreetCLIPEncoder(batch_size=3).embed(imgs)
    assert np.allclose(e, again, atol=1e-3)  # batching doesn't change the embedding
