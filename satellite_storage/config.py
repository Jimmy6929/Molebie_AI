"""Settings for the satellite-side blob storage service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime settings, env-overridable.

    Single-process service, single operator — no per-tenant config.
    """

    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".molebie" / "satellite-storage",
        description="Where blob files live on disk.",
    )
    bind_host: str = "0.0.0.0"
    bind_port: int = 8090


@lru_cache
def get_settings() -> Settings:
    """Cached Settings instance. Honors MOLEBIE_STORAGE_DATA_DIR and
    MOLEBIE_STORAGE_PORT for the two knobs operators are most likely to
    want to override."""
    data_dir = os.getenv("MOLEBIE_STORAGE_DATA_DIR")
    bind_port = os.getenv("MOLEBIE_STORAGE_PORT")
    kwargs: dict = {}
    if data_dir:
        kwargs["data_dir"] = Path(data_dir).expanduser()
    if bind_port:
        kwargs["bind_port"] = int(bind_port)
    return Settings(**kwargs)
