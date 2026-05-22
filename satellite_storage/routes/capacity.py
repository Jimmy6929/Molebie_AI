"""Capacity endpoint — disk space, used by the primary's capacity poller."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends

from satellite_storage.config import get_settings
from satellite_storage.middleware.tailscale_identity import (
    TailscaleIdentity,
    get_tailscale_identity,
)

router = APIRouter(prefix="/v1/storage", tags=["Storage"])


@router.get("/capacity")
async def capacity(
    _identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> dict:
    """Disk usage for the data directory.

    Reports filesystem-level numbers, not just blob-directory numbers —
    if the operator put the data_dir on a small partition, that's the
    real constraint and the primary needs to see it.
    """
    settings = get_settings()
    # disk_usage walks up to find the mountpoint; if data_dir doesn't
    # exist yet we look at its parent (the satellite may have just been
    # installed and not received any blobs yet).
    target = settings.data_dir if settings.data_dir.exists() else settings.data_dir.parent
    usage = shutil.disk_usage(target)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "data_dir": str(settings.data_dir),
    }
