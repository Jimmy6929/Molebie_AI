"""Integration tests for ``POST /fleet/storage/migrate``.

Loopback-gated trigger for the storage mover. The mover engine itself is
covered by test_storage_mover.py; here we test the route wiring — gate,
identity check, and response shape — with the mover monkeypatched.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync
from app.services.storage_mover import MigrationResult


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
        import asyncio
        asyncio.run(_close())


def _build_app():
    from app.main import create_app
    return create_app()


def _loopback_client(app):
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 0))
    return httpx.AsyncClient(transport=transport, base_url="http://primary")


def _remote_client(app):
    transport = httpx.ASGITransport(app=app, client=("10.0.0.5", 0))
    return httpx.AsyncClient(transport=transport, base_url="http://primary")


class TestMigrateRoute:
    def test_loopback_returns_counts(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        # Identity present.
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )
        # Mover returns a canned mix of migrated + skipped.
        def _fake_migrate(self, limit):
            return [
                MigrationResult("d1", migrated=True, satellite_node_id="n", sha256="abc"),
                MigrationResult("d2", migrated=False, reason="no_satellite"),
            ]
        monkeypatch.setattr(
            "app.services.storage_mover.StorageMover.migrate_documents", _fake_migrate
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/migrate?limit=5")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["migrated"] == 1
        assert body["skipped"] == 1
        assert len(body["results"]) == 2

    def test_remote_client_gets_403(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        async def _run():
            async with _remote_client(app) as client:
                return await client.post("/fleet/storage/migrate")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 403

    def test_no_identity_returns_503(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: None
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/migrate")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 503

    def test_limit_capped_at_100(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/migrate?limit=5000")

        import asyncio
        resp = asyncio.run(_run())
        # FastAPI Query(le=100) → 422 at validation.
        assert resp.status_code == 422
