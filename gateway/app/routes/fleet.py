"""Fleet management endpoints (Plan B).

This module currently exposes one read endpoint — ``GET /fleet/audit`` —
which streams recent audit-log rows. Future slices will add satellite
registration, fleet inventory, and per-satellite health endpoints; they
all share the loopback gate and metadata-only contract used here.

Design calls:

* **Loopback-only**, same gate the ``/metrics`` route uses. The operator
  running ``molebie-ai extend audit`` on the primary's machine gets
  zero-config access; remote clients cannot reach this endpoint.
* **Metadata only.** ``metadata_json`` is parsed back into a dict in the
  response; the schema-level invariant (no prompt/response content) is
  the caller's responsibility, enforced at ``audit.record`` time.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.services.database import get_database_service

router = APIRouter(prefix="/fleet", tags=["Fleet"])

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_AUDIT_DEFAULT_LIMIT = 100
_AUDIT_MAX_LIMIT = 500


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in _LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fleet endpoint is loopback-only",
        )


def _parse_metadata(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


@router.get("/audit")
async def list_audit_events(
    request: Request,
    limit: int = Query(_AUDIT_DEFAULT_LIMIT, ge=1, le=_AUDIT_MAX_LIMIT),
    event_type: str | None = Query(default=None),
    since: str | None = Query(
        default=None,
        description="ISO 8601 timestamp; returns events with created_at > since",
    ),
) -> dict:
    """Return the most recent audit-log entries."""
    _require_loopback(request)

    where: list[str] = []
    args: list[Any] = []
    if event_type is not None:
        where.append("event_type = ?")
        args.append(event_type)
    if since is not None:
        where.append("created_at > ?")
        args.append(since)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    args.append(limit)

    db = get_database_service()
    conn = await db._get_conn()
    rows = await conn.execute_fetchall(
        f"SELECT id, event_type, actor, target, metadata_json, created_at "
        f"FROM audit_events{where_sql} "
        f"ORDER BY created_at DESC, id DESC LIMIT ?",
        tuple(args),
    )

    events = [
        {
            "id": r["id"],
            "event_type": r["event_type"],
            "actor": r["actor"],
            "target": r["target"],
            "metadata": _parse_metadata(r["metadata_json"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return {"events": events, "count": len(events)}
