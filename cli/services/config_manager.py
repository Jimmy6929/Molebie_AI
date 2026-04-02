"""Load and save the CLI configuration file (.molebie/config.json)."""

from __future__ import annotations

import json
from pathlib import Path

from cli.models.config import MolebieConfig


def _find_project_root() -> Path:
    """Walk up from cwd looking for the project root (contains .env.example + gateway/)."""
    current = Path.cwd().resolve()
    for directory in [current, *current.parents]:
        if (directory / ".env.example").exists() and (directory / "gateway").is_dir():
            return directory
    raise FileNotFoundError(
        "Could not find Molebie AI project root. "
        "Run this command from inside the project directory."
    )


def get_project_root() -> Path:
    return _find_project_root()


def get_config_dir() -> Path:
    return get_project_root() / ".molebie"


def get_config_path() -> Path:
    return get_config_dir() / "config.json"


def ensure_config_dir() -> Path:
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def load_config() -> MolebieConfig:
    """Load config from disk, returning defaults if file doesn't exist."""
    path = get_config_path()
    if not path.exists():
        return MolebieConfig()
    data = json.loads(path.read_text())
    return MolebieConfig.model_validate(data)


def save_config(config: MolebieConfig) -> None:
    """Write config to disk as pretty-printed JSON."""
    ensure_config_dir()
    path = get_config_path()
    path.write_text(config.model_dump_json(indent=2) + "\n")


def config_exists() -> bool:
    return get_config_path().exists()


# Default matches gateway/app/config.py → Settings.embedding_model
_DEFAULT_EMBEDDING_MODEL = "Orange/orange-nomic-v1.5-1536"


def read_embedding_model() -> str:
    """Read EMBEDDING_MODEL from .env.local, fall back to project default.

    Single source of truth for the embedding model name across CLI commands.
    The default here must match gateway/app/config.py → Settings.embedding_model.
    """
    env_path = get_project_root() / ".env.local"
    if not env_path.exists():
        return _DEFAULT_EMBEDDING_MODEL
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("EMBEDDING_MODEL=") and not stripped.startswith("#"):
            value = stripped.split("=", 1)[1].strip()
            return value or _DEFAULT_EMBEDDING_MODEL
    return _DEFAULT_EMBEDDING_MODEL
