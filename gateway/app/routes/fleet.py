"""Fleet management endpoints (Plan B).

Three endpoints share this router:

* ``GET /fleet/audit`` — loopback-gated; reads recent ``audit_events``
  rows for the operator on the primary's machine.
* ``POST /fleet/satellites/register`` — Tailscale-identity-gated;
  satellites call this from remote tailnet IPs to upsert themselves
  into the fleet inventory.
* ``GET /fleet/inventory`` — loopback-gated; lists every registered
  satellite for the operator.

Two trust gates, used independently:

* ``_require_loopback`` — for operator-side reads (no remote access).
* ``get_tailscale_identity`` — for satellite-side writes (Tailscale
  daemon attests the caller's identity).

Metadata-only invariant: nothing in the response can leak prompt or
response content. ``tailscale_user`` is deliberately excluded from the
inventory response — who registered a satellite belongs in
``audit_events``, not in every inventory reader's view.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.middleware.tailscale_identity import (
    TailscaleIdentity,
    get_tailscale_identity,
)
from app.services import audit
from app.services.database import get_database_service

router = APIRouter(prefix="/fleet", tags=["Fleet"])

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_AUDIT_DEFAULT_LIMIT = 100
_AUDIT_MAX_LIMIT = 500

SatelliteRole = Literal["storage", "compute", "both"]


class SatelliteRegistration(BaseModel):
    """Request body for ``POST /fleet/satellites/register``.

    Field shape mirrors ``cli/models/config.py:SatelliteNode`` so the
    future ``molebie-ai join`` command's payload serializes cleanly.
    """

    host: str = Field(..., min_length=1, description="Tailscale hostname or IP of this satellite")
    role: SatelliteRole
    label: str | None = None
    capabilities: dict[str, Any] | None = None


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


# ─────────────────────────── satellite lifecycle ───────────────────────────


def _row_to_inventory_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "host": row["host"],
        "role": row["role"],
        "status": row["status"],
        "label": row["label"],
        "capabilities": _parse_metadata(row["capabilities_json"]),
        "joined_at": row["joined_at"],
        "updated_at": row["updated_at"],
    }


@router.post("/satellites/register")
async def register_satellite(
    payload: SatelliteRegistration,
    identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> dict[str, Any]:
    """Upsert a satellite into the fleet inventory.

    Called by satellites from remote tailnet IPs; identity is provided
    by the Tailscale daemon via injected headers. Not loopback-gated.
    Re-registration of an existing host is an UPDATE (preserves the
    server-assigned ``id``); the audit event distinguishes the two
    cases via ``satellite.join`` vs ``satellite.update``.
    """
    db = get_database_service()
    conn = await db._get_conn()
    now = datetime.now(timezone.utc).isoformat()
    capabilities_json = (
        json.dumps(payload.capabilities) if payload.capabilities is not None else None
    )

    existing = await conn.execute_fetchall(
        "SELECT id, joined_at FROM fleet_satellites WHERE host = ?",
        (payload.host,),
    )
    if existing:
        satellite_id = existing[0]["id"]
        await conn.execute(
            "UPDATE fleet_satellites "
            "SET role = ?, status = ?, label = ?, capabilities_json = ?, "
            "    tailscale_user = ?, updated_at = ? "
            "WHERE id = ?",
            (
                payload.role,
                "active",
                payload.label,
                capabilities_json,
                identity.user_login,
                now,
                satellite_id,
            ),
        )
        await conn.commit()
        await audit.record(
            "satellite.update",
            actor=identity.user_login,
            target=payload.host,
            metadata={"role": payload.role, "from_ip": identity.peer_ip},
        )
    else:
        satellite_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO fleet_satellites "
            "(id, host, role, status, label, capabilities_json, "
            " tailscale_user, joined_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                satellite_id,
                payload.host,
                payload.role,
                "active",
                payload.label,
                capabilities_json,
                identity.user_login,
                now,  # joined_at — same as updated_at on first registration
                now,
            ),
        )
        await conn.commit()
        await audit.record(
            "satellite.join",
            actor=identity.user_login,
            target=payload.host,
            metadata={"role": payload.role, "from_ip": identity.peer_ip},
        )

    rows = await conn.execute_fetchall(
        "SELECT id, host, role, status, label, capabilities_json, "
        "       joined_at, updated_at "
        "FROM fleet_satellites WHERE id = ?",
        (satellite_id,),
    )
    return _row_to_inventory_item(dict(rows[0]))


@router.get("/inventory")
async def list_inventory(request: Request) -> dict[str, Any]:
    """List every registered satellite (loopback-gated, metadata only)."""
    _require_loopback(request)

    db = get_database_service()
    conn = await db._get_conn()
    rows = await conn.execute_fetchall(
        "SELECT id, host, role, status, label, capabilities_json, "
        "       joined_at, updated_at "
        "FROM fleet_satellites "
        "ORDER BY joined_at ASC"
    )
    satellites = [_row_to_inventory_item(dict(r)) for r in rows]
    return {"satellites": satellites, "count": len(satellites)}


@router.post("/storage/migrate")
async def migrate_storage(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """Migrate up to ``limit`` locally-stored documents to a satellite.

    Loopback-gated (operator-only). Manual trigger for the slice-9.4a
    storage mover; the slice-9.4b scheduler will call the same engine
    automatically. Runs the (sync) mover in a worker thread so a large
    blob upload doesn't block the event loop.
    """
    _require_loopback(request)

    import asyncio

    from app.config import get_settings
    from app.services.storage import LocalStorageService
    from app.services.storage_mover import StorageMover
    from app.services.tailscale_outbound import get_operator_identity

    identity = get_operator_identity()
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No Tailscale identity available — cannot authenticate to satellites.",
        )

    data_dir = getattr(get_settings(), "data_dir", "data")
    mover = StorageMover(LocalStorageService(data_dir), identity, data_dir)
    results = await asyncio.to_thread(mover.migrate_documents, limit)

    migrated = sum(1 for r in results if r.migrated)
    return {
        "migrated": migrated,
        "skipped": len(results) - migrated,
        "results": [
            {
                "doc_id": r.doc_id,
                "migrated": r.migrated,
                "satellite_node_id": r.satellite_node_id,
                "sha256": r.sha256,
                "reason": r.reason,
            }
            for r in results
        ],
    }
