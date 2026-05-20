"""Load and save the CLI configuration file (.molebie/config.json)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


def _migrate_v2_to_v3(data: dict) -> dict:
    """Migrate old two-machine config to per-service distributed format."""
    if data.get("version", 2) >= 3:
        return data

    data["version"] = 3
    old_setup = data.get("setup_type", "single")

    if old_setup == "two-machine":
        data["setup_type"] = "distributed"
        gpu_ip = data.pop("gpu_ip", "localhost")
        data.pop("server_ip", None)
        # Old model: inference on remote GPU machine, gateway+webapp on this machine
        data["run_inference"] = False
        data["run_gateway"] = True
        data["run_webapp"] = True
        data["inference_host"] = gpu_ip
        data["gateway_host"] = "localhost"
        data["webapp_host"] = "localhost"
    else:
        data.pop("gpu_ip", None)
        data.pop("server_ip", None)

    return data


def _migrate_v3_to_v4(data: dict) -> dict:
    """Add the fleet satellites list. Synthesize one COMPUTE satellite for
    v3 distributed setups that pointed at a remote LLM machine; other v3 shapes
    get an empty satellites list and re-install on the intended primary."""
    if data.get("version", 3) >= 4:
        return data

    data["version"] = 4
    data.setdefault("satellites", [])

    if (
        data.get("setup_type") == "distributed"
        and data.get("run_inference") is False
        and data.get("run_gateway") is True
        and data.get("run_webapp") is True
    ):
        host = (data.get("inference_host") or "").strip()
        if host and host != "localhost":
            data["satellites"] = [
                {
                    "host": host,
                    "role": "compute",
                    "capabilities": {},
                    "status": "active",
                    "joined_at": datetime.now(timezone.utc).isoformat(),
                    "label": "migrated-from-v3",
                }
            ]

    return data


def load_config() -> MolebieConfig:
    """Load config from disk, returning defaults if file doesn't exist."""
    path = get_config_path()
    if not path.exists():
        return MolebieConfig()
    data = json.loads(path.read_text())
    original_version = data.get("version", 2)
    data = _migrate_v2_to_v3(data)
    data = _migrate_v3_to_v4(data)
    config = MolebieConfig.model_validate(data)
    if data.get("version") != original_version:
        save_config(config)
    return config


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
