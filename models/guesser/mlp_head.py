"""MLP head: a small neural network over geocells in place of the linear softmax. Owner: Daniel Hu.

p(k | v) = softmax((W2 relu(W1 standardise(v) + b1) + b2) / T), on the same geocells as the
baseline (spherical k-means with the same seed), so the comparison isolates the head.

* ``temperature`` T: calibration, chosen on val after training (no retraining). The guess maximises
  the expected score, which needs honest probabilities.
* ``country_weight``: optional second training target. A country head on the same hidden layer
  adds ``country_weight * cross-entropy(country)`` to the loss, pushing the shared layer towards
  country-level cues that carry over to unseen cities. Inference uses the cell head only.

Trained in PyTorch (``uv run --extra ml``), but the weights are stored as numpy arrays and
inference is numpy, so a saved model runs without torch (evaluate.py, play.py, the game).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from models.guesser.classifier import _GeocellGuesser, _prior, _softmax
from models.guesser.geocells import fit_geocells
from models.guesser.registry import register


@register("mlp_head")
class MLPGuesser(_GeocellGuesser):
    def __init__(self, centroids, prior, mean, scale, w1, b1, w2, b2, temperature=1.0):
        super().__init__(centroids, prior)
        self.temperature = float(temperature)
        f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
        self.mean, self.scale = f32(mean), f32(scale)  # (D,) training standardisation
        self.w1, self.b1 = f32(w1), f32(b1)  # (D, H), (H,)
        self.w2, self.b2 = f32(w2), f32(b2)  # (H, K), (K,)

    def _per_view_probs(self, view_embs):
        """(t, 768) view embeddings -> (t, K) cell probabilities, each row summing to 1."""
        x = (np.atleast_2d(view_embs).astype(np.float32) - self.mean) / self.scale
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return _softmax((h @ self.w2 + self.b2) / self.temperature)

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        n_cells: int = 256,
        hidden: int = 1024,
        dropout: float = 0.3,
        label_smoothing: float = 0.0,
        country: np.ndarray | None = None,
        country_weight: float = 0.0,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 1024,
        seed: int = 0,
        log: Callable[[str], None] | None = print,
    ) -> MLPGuesser:
        """x: (n_views, 768) TRAIN-split view embeddings; lat, lon: (n_views,) their locations;
        country: (n_views,) country codes, needed when country_weight > 0."""
        import torch

        centroids, labels = fit_geocells(lat, lon, n_cells, 50, 0)  # same cells as the baseline
        k = len(centroids)
        x = x.astype(np.float32)
        mean, scale = x.mean(0), x.std(0) + 1e-6

        torch.manual_seed(seed)
        dev = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        xs = torch.from_numpy((x - mean) / scale).to(dev)
        ys = torch.from_numpy(labels.astype(np.int64)).to(dev)
        n, d = xs.shape
        body = torch.nn.Sequential(
            torch.nn.Dropout(dropout / 2),  # light input dropout
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ).to(dev)
        cell_head = torch.nn.Linear(hidden, k).to(dev)
        with torch.no_grad():  # start from the class prior, like the baseline's bias
            cell_head.bias.copy_(torch.from_numpy(np.log(_prior(labels, k)).astype(np.float32)))
        params = list(body.parameters()) + list(cell_head.parameters())
        if country_weight > 0:
            if country is None:
                raise ValueError("country_weight > 0 needs country labels")
            codes, country_ids = np.unique(np.asarray(country), return_inverse=True)
            cs = torch.from_numpy(country_ids.astype(np.int64)).to(dev)
            country_head = torch.nn.Linear(hidden, len(codes)).to(dev)
            params += list(country_head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        steps = epochs * math.ceil(n / batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        body.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, generator=gen).to(dev)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                hid = body(xs[idx])
                loss = loss_fn(cell_head(hid), ys[idx])
                if country_weight > 0:
                    loss = loss + country_weight * loss_fn(country_head(hid), cs[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            if log and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1):
                log(f"    epoch {epoch + 1}/{epochs}: train loss {total / n:.3f}")
        body.eval()

        def to_np(t):
            return t.detach().cpu().numpy()

        return cls(
            centroids,
            _prior(labels, k),
            mean,
            scale,
            to_np(body[1].weight).T,
            to_np(body[1].bias),
            to_np(cell_head.weight).T,
            to_np(cell_head.bias),
        )

    def _arrays(self):
        return {
            "mean": self.mean, "scale": self.scale,
            "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2,
            "temperature": np.float64(self.temperature),
        }  # fmt: skip

    @classmethod
    def _from_arrays(cls, a):
        t = float(a["temperature"]) if "temperature" in a else 1.0
        return cls(
            a["centroids"], a["prior"], a["mean"], a["scale"],
            a["w1"], a["b1"], a["w2"], a["b2"], t,
        )  # fmt: skip
