"""Save and load any guesser variant the same way, so evaluate.py and play.py work with all of them.

A saved model is a folder, runs/models/<name>/ (gitignored; share it with
``scripts/data_hub.py push-model <name>``)::

    model.json      name, kind, dataset, author, description, config
    ...             whatever the class writes (e.g. guesser.npz)

To add a variant, write a class in any module under models/guesser/ that implements
core.interfaces.Guesser plus ``save_to(folder)`` and a classmethod ``load_from(folder)``, and
decorate it with ``@register("my_kind")``. Save a trained one with ``save_model``.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import subprocess
from pathlib import Path
from typing import Any

from core.config import resolve
from core.interfaces import Guesser

MODEL_JSON = "model.json"
REGISTRY: dict[str, type] = {}


def register(kind: str):
    """Class decorator: makes ``kind`` loadable by ``load_model``."""

    def deco(cls: type) -> type:
        cls.kind = kind
        REGISTRY[kind] = cls
        return cls

    return deco


def model_dir(name: str) -> Path:
    return resolve("runs/models") / name


def save_model(
    guesser,
    *,
    name: str,
    dataset: str,
    description: str = "",
    config: dict[str, Any] | None = None,
    author: str | None = None,
) -> Path:
    """Write ``guesser`` and its model.json into runs/models/<name>/."""
    if author is None:
        r = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True)
        author = r.stdout.strip()
    out = model_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    guesser.save_to(out)
    info = {
        "name": name,
        "kind": type(guesser).kind,
        "dataset": dataset,
        "author": author,
        "description": description,
        "config": config or {},
    }
    (out / MODEL_JSON).write_text(json.dumps(info, indent=2, default=str) + "\n")
    return out


def find_model(name_or_path: str | Path) -> Path:
    """The folder of a model, given its name or its path."""
    for folder in (resolve(name_or_path), model_dir(str(name_or_path))):
        if (folder / MODEL_JSON).is_file():
            return folder
    raise FileNotFoundError(
        f"no model {str(name_or_path)!r} in runs/models/. If a teammate trained it: "
        f"uv run python scripts/data_hub.py pull-model {name_or_path}"
    )


def model_info(name_or_path: str | Path) -> dict[str, Any]:
    return json.loads((find_model(name_or_path) / MODEL_JSON).read_text())


def load_model(name_or_path: str | Path) -> Guesser:
    """Load a saved model by name or folder."""
    import models.guesser as pkg

    for m in pkgutil.iter_modules(pkg.__path__):  # runs every @register in the package
        importlib.import_module(f"{pkg.__name__}.{m.name}")
    folder = find_model(name_or_path)
    kind = model_info(folder)["kind"]
    if kind not in REGISTRY:
        raise ValueError(
            f"{folder}: unknown kind {kind!r}; is the module defining it on your branch?"
        )
    return REGISTRY[kind].load_from(folder)
