"""Encoders implementing core.interfaces.Encoder."""

from __future__ import annotations

import numpy as np
from PIL import Image

from core.view_spec import EMBED_DIM, ENCODER_NAME


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class FakeEncoder:
    """Deterministic, torch-free stand-in for StreetCLIP: a 16x16 RGB thumbnail (= 768 values),
    mean-centred and L2-normalised. Similar pixels give similar embeddings, so it is good
    enough for fixtures, tests, CI and the crop-vs-screenshot consistency check."""

    dim = EMBED_DIM
    _side = 16  # 16 * 16 * 3 == 768

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        thumbs = [
            np.asarray(
                Image.fromarray(im).resize((self._side, self._side), Image.BILINEAR), np.float32
            ).reshape(-1)
            for im in images
        ]
        x = np.stack(thumbs) / 255.0
        return _l2norm(x - x.mean(axis=1, keepdims=True)).astype(np.float32)


class StreetCLIPEncoder:
    """Frozen StreetCLIP image tower. Needs the ``ml`` extra: ``uv sync --extra ml``."""

    dim = EMBED_DIM

    def __init__(self, device: str = "auto", batch_size: int = 32, model_name: str = ENCODER_NAME):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = self.processor(images=images[i : i + self.batch_size], return_tensors="pt")
                feats = self.model.get_image_features(
                    pixel_values=batch["pixel_values"].to(self.device)
                )
                # transformers < 5 returns the projected embedding as a tensor; 5.x returns an
                # output object whose pooler_output is that same projection (checked equal).
                if not isinstance(feats, self.torch.Tensor):
                    feats = feats.pooler_output
                out.append(feats.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return _l2norm(np.concatenate(out)).astype(np.float32)


def make_encoder(cfg: dict) -> FakeEncoder | StreetCLIPEncoder:
    name = cfg.get("encoder", "fake")
    if name == "fake":
        return FakeEncoder()
    if name == "streetclip":
        return StreetCLIPEncoder(
            device=cfg.get("device", "auto"), batch_size=cfg.get("batch_size", 32)
        )
    raise ValueError(f"unknown encoder {name!r}")
