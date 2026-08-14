"""Tests for the per-turn pipeline stage metrics (T6a).

Covers: contextvar-driven attribution of subsystem timers into the
per-request pipeline log, stage-event shaping, best-effort persistence,
and the schema for the new pipeline_stage_metrics table.
"""

from __future__ import annotations

import json
import sqlite3

from app.routes.chat import _persist_stage_metrics, _shape_stage_events
from app.schema import init_database_sync
from app.services.metrics_registry import MetricsRegistry, current_request_id

# ── contextvar attribution ─────────────────────────────────────────────────


async def test_record_subsystem_attributes_to_current_request():
    reg = MetricsRegistry()
    token = current_request_id.set("req-abc")
    try:
        await reg.record_subsystem("rag.embed", 12.5)
        await reg.record_subsystem("memory.embed", 3.0, ok=False, note="timeout")
    finally:
        current_request_id.reset(token)

    events = await reg.get_request_events("req-abc")
    assert [e["stage"] for e in events] == ["rag.embed", "memory.embed"]
    assert events[0]["status"] == "ok" and events[0]["ms"] == 12.5
    assert events[1]["status"] == "fail" and events[1]["note"] == "timeout"


async def test_record_subsystem_without_request_id_stays_out_of_pipeline():
    reg = MetricsRegistry()
    assert current_request_id.get() is None
    await reg.record_subsystem("rag.embed", 5.0)
    snap = await reg.pipeline_snapshot()
    assert snap["events"] == []


async def test_get_request_events_filters_by_req_id():
    reg = MetricsRegistry()
    await reg.pipeline_event("req-1", "request.start", status="ok")
    await reg.pipeline_event("req-2", "request.start", status="ok")
    await reg.pipeline_event("req-1", "rag.retrieve", ms=40.0, status="ok")
    events = await reg.get_request_events("req-1")
    assert [e["stage"] for e in events] == ["request.start", "rag.retrieve"]
    assert all(e["req_id"] == "req-1" for e in events)


# ── stage shaping ──────────────────────────────────────────────────────────


def test_shape_drops_running_and_computes_wall_span():
    events = [
        {"ts": 100.0, "req_id": "r", "stage": "request.start", "ms": None,
         "status": "ok", "note": None},
        {"ts": 100.1, "req_id": "r", "stage": "inference.instant", "ms": None,
         "status": "running", "note": "warm"},
        {"ts": 102.5, "req_id": "r", "stage": "inference.instant", "ms": 2400.0,
         "status": "ok", "note": "12 deltas"},
    ]
    stages, total_ms = _shape_stage_events(events)
    assert [s["stage"] for s in stages] == ["request.start", "inference.instant"]
    assert stages[1]["note"] == "12 deltas"
    # Wall span first→last event, not the sum of (nested) stage ms.
    assert total_ms == 2500.0


def test_shape_empty_events():
    stages, total_ms = _shape_stage_events([])
    assert stages == [] and total_ms is None


# ── persistence helper ─────────────────────────────────────────────────────


class _FakeDB:
    def __init__(self):
        self.calls = []

    async def insert_pipeline_stage_metrics(self, user_id, metrics):
        self.calls.append((user_id, metrics))


async def test_persist_writes_row_with_message_join():
    reg = MetricsRegistry()
    await reg.pipeline_event("req-9", "request.start", status="ok")
    await reg.pipeline_event("req-9", "rag.retrieve", ms=55.0, status="ok")
    db = _FakeDB()
    await _persist_stage_metrics(
        db, reg, "req-9",
        user_id="u1", session_id="s1", message_id="m1",
        route="chat", mode="instant",
    )
    assert len(db.calls) == 1
    uid, m = db.calls[0]
    assert uid == "u1"
    assert m["message_id"] == "m1" and m["request_id"] == "req-9"
    assert m["route"] == "chat" and m["mode"] == "instant"
    stages = json.loads(m["stages"])
    assert [s["stage"] for s in stages] == ["request.start", "rag.retrieve"]


async def test_persist_noop_when_no_events():
    db = _FakeDB()
    await _persist_stage_metrics(
        db, MetricsRegistry(), "req-unknown",
        user_id="u1", session_id=None, message_id=None,
        route="chat", mode=None,
    )
    assert db.calls == []


async def test_persist_swallows_db_errors():
    class _BoomDB:
        async def insert_pipeline_stage_metrics(self, *a):
            raise RuntimeError("boom")

    reg = MetricsRegistry()
    await reg.pipeline_event("req-x", "request.start", status="ok")
    # Must not raise — a metrics-write failure cannot break the chat.
    await _persist_stage_metrics(
        _BoomDB(), reg, "req-x",
        user_id="u1", session_id=None, message_id=None,
        route="chat_stream", mode="thinking",
    )


# ── schema ─────────────────────────────────────────────────────────────────


def test_schema_creates_table_fresh_and_on_migrate(tmp_path):
    # Fresh DB gets the table.
    db_path = init_database_sync(str(tmp_path))
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='pipeline_stage_metrics'"
        ).fetchone()
        assert row is not None
        # Simulate a pre-T6a database: drop, then re-run init (migrate path).
        conn.execute("DROP TABLE pipeline_stage_metrics")
        conn.commit()
    finally:
        conn.close()

    init_database_sync(str(tmp_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO pipeline_stage_metrics "
            "(id, user_id, session_id, message_id, request_id, route, mode, "
            "total_ms, stages, created_at) "
            "VALUES ('p1', 'u1', 's1', 'm1', 'r1', 'chat', 'instant', "
            "123.4, '[]', '2026-08-14T00:00:00Z')"
        )
        got = conn.execute(
            "SELECT message_id, total_ms FROM pipeline_stage_metrics "
            "WHERE request_id='r1'"
        ).fetchone()
        assert got == ("m1", 123.4)
    finally:
        conn.close()
