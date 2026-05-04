"""HTTP-level observability middleware.

Wraps every incoming request and records it as a `http.{normalized_path}`
subsystem call in the metrics registry. The Subsystems panel in the
monitor then shows constant motion at every API call — chat, sessions,
documents, voice, TTS — without per-route hand-instrumentation.

Why a middleware (not per-route decorators):
  * Coverage by default: any route added later is observed automatically.
  * One place to filter noisy paths (the monitor itself polls /metrics/live
    at 2 Hz; logging that creates feedback noise).
  * Sub-microsecond overhead — `time.monotonic()` + one async lock acquire
    on the registry's existing structures.

Path normalization collapses session-id-like segments to `:id` so we
don't end up with thousands of unique subsystem names — `GET
/chat/sessions/abc123/messages` and `GET /chat/sessions/xyz789/messages`
both record under `http.GET.chat.sessions.:id.messages`.
"""

from __future__ import annotations

import re
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.services.metrics_registry import get_metrics_registry

# Routes we never record — they'd create feedback noise (the monitor
# itself polls /metrics/live at 2 Hz, /health is hit by liveness probes).
_SKIP_PATHS: set[str] = {"/metrics/live", "/health"}

# Path-segment patterns we collapse to a placeholder so per-id routes
# don't blow up the subsystem dict. UUIDs, hex tokens, numeric IDs.
_ID_SEGMENT_RE = re.compile(
    r"^("
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"   # UUID
    r"|[0-9a-fA-F]{16,}"                                                              # long hex
    r"|\d+"                                                                           # bare numeric
    r")$"
)


def _normalize_path(method: str, path: str) -> str:
    """Turn ('GET', '/chat/sessions/abc-def/messages') into
    'http.GET.chat.sessions.:id.messages'."""
    parts = [p for p in path.split("/") if p]
    normalized = [
        ":id" if _ID_SEGMENT_RE.match(p) else p
        for p in parts
    ]
    return "http." + method + "." + ".".join(normalized) if normalized else f"http.{method}.root"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Record every request's method+path+status+latency into the registry."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in _SKIP_PATHS:
            return await call_next(request)

        registry = get_metrics_registry()
        name = _normalize_path(request.method, path)
        start = time.monotonic()
        ok = True
        status_code: int | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            ok = status_code < 500
            return response
        except Exception:
            ok = False
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000.0
            note = f"{status_code}" if status_code is not None else "exception"
            try:
                await registry.record_subsystem(name, latency_ms, ok=ok, note=note)
            except Exception:
                # Observability failure must never affect the user-facing response.
                pass
