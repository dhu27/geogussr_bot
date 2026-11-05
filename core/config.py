"""Per-module config loading. One YAML file per module in configs/; no shared global config.

Paths in configs are relative to the repo root unless absolute. ``GEO_DATA_ROOT`` overrides
``data_root`` so each person can point at their own copy of the shared drive.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def load_config(module: str) -> dict[str, Any]:
    """Load configs/<module>.yaml."""
    with open(CONFIG_DIR / f"{module}.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    if module == "data" and os.environ.get("GEO_DATA_ROOT"):
        cfg["data_root"] = os.environ["GEO_DATA_ROOT"]
    return cfg


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so secrets like MAPILLARY_TOKEN stay out of git."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
