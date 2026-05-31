"""Manifest endpoint — what blobs does this satellite actually hold?

The primary's ``satellite_blobs`` table is the *authoritative* record of
which blobs the primary *expects* to live on each satellite. This
endpoint lets the primary discover what's *actually* on disk so it can
detect drift: blobs the primary recorded that vanished, blobs on disk
the primary never recorded, or size mismatches (the only corruption
signature catchable by light scrub).

The walk is sync because (a) the CAS layout is just a directory tree —
no I/O bottleneck at v0.2 scale (thousands of blobs), and (b) the route
is otherwise read-only with no concurrency benefits from aiofiles.

``.partial`` filenames are skipped — they're in-flight writes from CAS
``write_blob`` and not yet a real blob.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from satellite_storage.config import get_settings
from satellite_storage.middleware.tailscale_identity import (
    TailscaleIdentity,
    get_tailscale_identity,
)

router = APIRouter(prefix="/v1/storage", tags=["Storage"])


@router.get("/manifest")
async def manifest(
    _identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> dict:
    """List every blob this satellite currently holds.

    Returns ``{generated_at, blobs: [{sha256, size_bytes}, ...]}``, sorted
    by sha256 for deterministic primary-side diffs.
    """
    settings = get_settings()
    blobs_root = settings.data_dir / "blobs"
    entries: list[dict] = []
    if blobs_root.exists():
        for path in blobs_root.glob("*/*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            entries.append({"sha256": path.name, "size_bytes": path.stat().st_size})
    entries.sort(key=lambda e: e["sha256"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blobs": entries,
    }
