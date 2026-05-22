"""Health endpoint — liveness check, intentionally unauthenticated.

Matches the gateway's ``/health`` pattern: anyone on the tailnet can
ping this to verify the service is alive, even before Tailscale
identity headers are end-to-end configured. The capacity endpoint
requires identity because it leaks system info; ``/health`` is just
"yes, the process is running."
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from satellite_storage import __version__

router = APIRouter(prefix="/v1/storage", tags=["Storage"])

_STARTED_AT = time.monotonic()


@router.get("/health")
async def health() -> dict:
    return {
        "version": __version__,
        "uptime_sec": int(time.monotonic() - _STARTED_AT),
    }
