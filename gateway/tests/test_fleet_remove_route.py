"""Integration tests for ``DELETE /fleet/satellites/{node_id}`` — slice 9.6.

Three paths covered: graceful (drained-clean), force=true (skips drain
+ accepts data loss), refused (409 when blobs still pending without
force).
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync
from tests.test_storage_drainer import _seed_satellite_doc
from tests.test_storage_mover import _register_satellite


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


def _satellite_row_count(data_dir: str, node_id: str) -> int:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM fleet_satellites WHERE id = ?", (node_id,)
        ).fetchone()[0]
    finally:
        conn.close()


class TestRemoveRoute:
    def test_graceful_success_when_drained(self, isolated_data_dir):
        _register_satellite(isolated_data_dir, "node-a", "100.64.0.9")
        # No satellite_blobs rows for node-a — drained clean.

        async def _run():
            async with _loopback_client(_build_app()) as client:
                return await client.delete("/fleet/satellites/node-a")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["removed"] is True
        assert body["forced"] is False
        assert body["lost_blobs"] == 0
        assert _satellite_row_count(isolated_data_dir, "node-a") == 0

    def test_refuses_409_when_blobs_still_present(self, isolated_data_dir):
        _register_satellite(isolated_data_dir, "node-a", "100.64.0.9")
        _seed_satellite_doc(isolated_data_dir, "node-a", "a" * 64, 10)

        async def _run():
            async with _loopback_client(_build_app()) as client:
                return await client.delete("/fleet/satellites/node-a")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 409
        # Satellite untouched.
        assert _satellite_row_count(isolated_data_dir, "node-a") == 1

    def test_force_succeeds_even_with_blobs_present(self, isolated_data_dir, monkeypatch):
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )
        _register_satellite(isolated_data_dir, "node-a", "100.64.0.9")
        _seed_satellite_doc(isolated_data_dir, "node-a", "a" * 64, 100)
        _seed_satellite_doc(isolated_data_dir, "node-a", "b" * 64, 250,
                            filename="b.pdf")

        async def _run():
            async with _loopback_client(_build_app()) as client:
                return await client.delete("/fleet/satellites/node-a?force=true")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["removed"] is True
        assert body["forced"] is True
        assert body["lost_blobs"] == 2
        assert body["lost_bytes"] == 350
        assert _satellite_row_count(isolated_data_dir, "node-a") == 0

    def test_404_unknown_satellite(self, isolated_data_dir):
        async def _run():
            async with _loopback_client(_build_app()) as client:
                return await client.delete("/fleet/satellites/ghost")

        import asyncio
        assert asyncio.run(_run()).status_code == 404
