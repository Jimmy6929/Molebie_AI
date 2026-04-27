"""Live metrics endpoint for the `molebie-ai monitor` terminal dashboard.

Design calls baked in:

  * **Loopback-only.** The endpoint rejects any request whose client IP is
    not 127.0.0.1 / ::1 / localhost. No JWT required on loopback — the
    operator running `molebie-ai monitor` on the same host gets zero-
    config access, while remote clients can't reach it at all. This is
    strictly safer than JWT-with-wide-access (no token can leak off-box).
  * **Metadata only.** Nothing in the response can expose prompt or
    response text. Session IDs are not included. v2 can add an opt-in
    prompt-preview path behind an explicit header gate.
  * **Cheap.** Reads cached probe snapshots; does not trigger any
    blocking work. Safe to poll at 2 Hz.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from app.services.backend_probe import get_backend_probe
from app.services.metrics_registry import get_metrics_registry
from app.services.system_probe import get_system_probe

router = APIRouter(prefix="/metrics", tags=["Metrics"])

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in _LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Metrics endpoint is loopback-only",
        )


@router.get("/live")
async def get_live_metrics(request: Request) -> dict:
    """Return a single snapshot of gateway + system + backend state."""
    _require_loopback(request)

    registry_snap = await get_metrics_registry().snapshot()
    sys_snap = get_system_probe().latest()
    backend_snap = get_backend_probe().latest()

    return {
        "ts": time.time(),
        "system": {
            "cpu_percent": sys_snap.cpu_percent,
            "cpu_cores": sys_snap.cpu_cores,
            "cpu_cores_physical": sys_snap.cpu_cores_physical,
            "ram_used_gb": sys_snap.ram_used_gb,
            "ram_total_gb": sys_snap.ram_total_gb,
            "ram_percent": sys_snap.ram_percent,
            "gpu_percent": sys_snap.gpu_percent,
            "gpu_temp_c": sys_snap.gpu_temp_c,
            "power_w": sys_snap.power_w,
            "note": sys_snap.note,
        },
        "backends": [
            {
                "tier": b.tier,
                "url": b.url,
                "model": b.model,
                "status": b.status,
                "last_latency_ms": b.last_latency_ms,
                "last_error": b.last_error,
            }
            for b in backend_snap.values()
        ],
        "requests": {
            "uptime_sec": registry_snap["uptime_sec"],
            "ttft_p50_ms": registry_snap["ttft_p50_ms"],
            "tpot_mean_ms": registry_snap["tpot_mean_ms"],
            "total_p50_ms": registry_snap["total_p50_ms"],
            "total_p95_ms": registry_snap["total_p95_ms"],
            "total_p99_ms": registry_snap["total_p99_ms"],
            "req_per_sec": registry_snap["req_per_sec"],
            "fallback_count": registry_snap["fallback_count"],
            "errors_60s": registry_snap["errors_60s"],
            "tiers": registry_snap["tiers"],
            "ttft_series": registry_snap["ttft_series"],
            "tpot_series": registry_snap["tpot_series"],
            "recent_events": registry_snap["recent_events"],
        },
    }
