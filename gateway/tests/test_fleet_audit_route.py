"""Integration tests for ``GET /fleet/audit``.

Mounts the real FastAPI app and exercises the route via httpx's ASGI
transport so the loopback gate is honored as in production. The
``client=("127.0.0.1", 0)`` arg on ASGITransport spoofs the client address
so ``request.client.host`` reads as a loopback IP — the same trick the
test_cors_regex test relies on.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync


@pytest.fixture
def isolated_data_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        get_settings.cache_clear()
        init_database_sync(td, embedding_dim=1024, auth_mode="single")

        from app.services import database
        database._db_service = None  # type: ignore[attr-defined]
        database.get_database_service.cache_clear()

        yield td

        async def _close():
            from app.services.database import get_database_service
            await get_database_service().close()
        asyncio.run(_close())


def _build_app():
    from app.main import create_app
    return create_app()


def _loopback_client(app):
    """httpx.AsyncClient that the route will see as coming from 127.0.0.1."""
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 0))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _remote_client(app):
    """httpx.AsyncClient that the route will see as coming from a remote IP."""
    transport = httpx.ASGITransport(app=app, client=("10.0.0.5", 0))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def _seed_events():
    """Plant a handful of audit events spanning multiple types and times."""
    from app.services.audit import record
    await record("storage.upload", actor="user-a", target="blob-1",
                 metadata={"size": 1024})
    await record("routing.fallback", actor="local-thinking",
                 target="local-instant", metadata={"reason": "circuit_open"})
    await record("storage.upload", actor="user-a", target="blob-2",
                 metadata={"size": 4096})
    await record("satellite.join", actor="home-server", target="primary")


class TestFleetAuditRoute:
    def test_loopback_returns_recent_events_desc(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            await _seed_events()
            async with _loopback_client(app) as client:
                resp = await client.get("/fleet/audit")
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 4
        assert len(body["events"]) == 4
        # Most-recent first — the last seeded event should be index 0.
        assert body["events"][0]["event_type"] == "satellite.join"
        # Metadata is parsed back into a dict.
        upload_events = [e for e in body["events"] if e["event_type"] == "storage.upload"]
        assert all(isinstance(e["metadata"], dict) for e in upload_events)
        assert any(e["metadata"].get("size") == 1024 for e in upload_events)

    def test_remote_client_gets_403(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _remote_client(app) as client:
                return await client.get("/fleet/audit")

        resp = asyncio.run(_run())
        assert resp.status_code == 403
        assert "loopback" in resp.json()["detail"].lower()

    def test_event_type_filter(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            await _seed_events()
            async with _loopback_client(app) as client:
                return await client.get(
                    "/fleet/audit",
                    params={"event_type": "storage.upload"},
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert all(e["event_type"] == "storage.upload" for e in body["events"])

    def test_since_filter(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            from app.services.audit import record
            await record("event.early")
            # Capture a timestamp between the two records.
            from datetime import datetime, timezone
            cutoff = datetime.now(timezone.utc).isoformat()
            await asyncio.sleep(0.01)  # ensure created_at differs
            await record("event.late")

            async with _loopback_client(app) as client:
                return await client.get(
                    "/fleet/audit", params={"since": cutoff}
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        # Only the "late" event should remain.
        assert body["count"] == 1
        assert body["events"][0]["event_type"] == "event.late"

    def test_limit_is_honored(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            from app.services.audit import record
            for i in range(20):
                await record("burst.event", metadata={"i": i})
            async with _loopback_client(app) as client:
                return await client.get("/fleet/audit", params={"limit": 5})

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 5

    def test_limit_capped_at_500(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _loopback_client(app) as client:
                return await client.get("/fleet/audit", params={"limit": 5000})

        resp = asyncio.run(_run())
        # FastAPI's Query(le=500) rejects with 422 — the cap is enforced at
        # parameter validation time, not silently clamped. Either is fine;
        # we just need the contract.
        assert resp.status_code == 422
