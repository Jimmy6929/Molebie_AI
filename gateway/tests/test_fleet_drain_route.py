"""Integration tests for ``POST /fleet/storage/drain`` and
``GET /fleet/extend/drain-preview`` — slice 9.6.

Drainer engine itself is covered by test_storage_drainer.py; here we
verify route wiring — loopback gate, identity check, response shape.
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
from app.services.storage_drainer import (
    DrainBatchReport,
    DrainPreview,
    DrainResult,
)
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


def _remote_client(app):
    transport = httpx.ASGITransport(app=app, client=("10.0.0.5", 0))
    return httpx.AsyncClient(transport=transport, base_url="http://primary")


class TestDrainRoute:
    def test_loopback_returns_counts(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )
        _register_satellite(isolated_data_dir, "node-a", "100.64.0.9")

        def _fake_drain(self, node_id, limit):
            return DrainBatchReport(
                node_id=node_id, drained=1, skipped=0, remaining=0,
                bytes_drained=42,
                results=[DrainResult("a" * 64, drained=True, docs_relocated=1,
                                     bytes_drained=42)],
            )
        monkeypatch.setattr(
            "app.services.storage_drainer.StorageDrainer.drain", _fake_drain
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/drain?node=node-a&limit=10")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["drained"] == 1
        assert body["remaining"] == 0
        assert body["bytes_drained"] == 42
        assert len(body["results"]) == 1
        assert body["results"][0]["docs_relocated"] == 1

    def test_remote_403(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        async def _run():
            async with _remote_client(app) as client:
                return await client.post("/fleet/storage/drain?node=node-a")

        import asyncio
        assert asyncio.run(_run()).status_code == 403

    def test_no_identity_returns_503(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: None
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/drain?node=node-a")

        import asyncio
        assert asyncio.run(_run()).status_code == 503


class TestDrainPreviewRoute:
    def test_preview_returns_feasibility(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )
        _register_satellite(isolated_data_dir, "node-a", "100.64.0.9")

        def _fake_preview(self, node_id):
            return DrainPreview(
                node_id=node_id, blob_count=3, total_bytes=1000,
                primary_free_bytes=10**9, feasible=True, satellite_reachable=True,
            )
        monkeypatch.setattr(
            "app.services.storage_drainer.StorageDrainer.preview", _fake_preview
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.get("/fleet/extend/drain-preview?node=node-a")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["blob_count"] == 3
        assert body["feasible"] is True
        assert body["satellite_reachable"] is True

    def test_preview_404_unknown_node(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        # No satellite registered — route resolves preview but then 404s on
        # the existence check.
        def _fake_preview(self, node_id):
            return DrainPreview(node_id=node_id)
        monkeypatch.setattr(
            "app.services.storage_drainer.StorageDrainer.preview", _fake_preview
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.get("/fleet/extend/drain-preview?node=ghost")

        import asyncio
        assert asyncio.run(_run()).status_code == 404
