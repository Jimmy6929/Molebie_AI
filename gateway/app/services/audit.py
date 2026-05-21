"""Append-only audit log for fleet operations.

A single public function — ``record`` — writes one row to the ``audit_events``
table. Callers pass an ``event_type`` (e.g. ``"storage.upload"``, ``"routing.fallback"``,
``"satellite.join"``) plus optional actor / target / metadata. The table and
its indexes are created by the schema migration in ``app/schema.py``.

Design calls:

* **Always-on, opt-out is not a thing.** Audit recording is part of the
  baseline security posture (Plan A and Plan B CLI Flow doc §"Always on,
  no configuration"). No env flag, no config knob.
* **Never crashes the caller.** Audit recording is telemetry-shaped: a
  failure to write must not propagate. The function catches and logs.
* **Metadata only, never content.** ``metadata`` is a small JSON-serializable
  dict — never prompt text, response text, document content, or PII. Same
  invariant the metrics registry enforces.
* **No new connection pool.** Reuses ``DatabaseService``'s singleton aiosqlite
  connection so audit writes ride the same WAL-mode pipe as the rest of the
  gateway's DB activity. Cheap on the happy path; bounded on failure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.services.database import get_database_service


async def record(
    event_type: str,
    *,
    actor: str | None = None,
    target: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one event to ``audit_events``. Never raises."""
    try:
        db = get_database_service()
        conn = await db._get_conn()
        await conn.execute(
            "INSERT INTO audit_events "
            "(event_type, actor, target, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event_type,
                actor,
                target,
                json.dumps(metadata) if metadata is not None else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await conn.commit()
    except Exception as exc:  # pragma: no cover - defensive
        # Telemetry-style: never break the caller's code path on a log
        # write failure. Surface via stderr so operators can spot it in
        # gateway logs.
        print(f"[audit] failed to record {event_type!r}: {exc}")
