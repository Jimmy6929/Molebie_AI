"""Integration tests for ``POST /fleet/storage/reconcile``.

Reconciler engine is covered by test_storage_reconciler.py; here we
test the route wiring — loopback gate, identity check, single-node vs
reconcile-all dispatch, and response shape — with the reconciler
monkeypatched.
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
from app.services.storage_reconciler import DriftEntry, ReconciliationReport


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


class TestReconcileRoute:
    def test_single_node_returns_report_with_drift(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        def _fake_reconcile(self, node_id):
            return ReconciliationReport(
                node_id=node_id, verified=2,
                drift=[DriftEntry("d" * 64, "orphan_on_satellite", actual_size=99)],
            )
        monkeypatch.setattr(
            "app.services.storage_reconciler.ManifestReconciler.reconcile",
            _fake_reconcile,
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/reconcile?node=node-a")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["reports"]) == 1
        r = body["reports"][0]
        assert r["node_id"] == "node-a"
        assert r["verified"] == 2
        assert r["fetch_error"] is None
        assert len(r["drift"]) == 1
        assert r["drift"][0]["kind"] == "orphan_on_satellite"
        assert r["drift"][0]["sha256"] == "d" * 64
        assert r["drift"][0]["actual_size"] == 99

    def test_reconcile_all_when_node_omitted(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        def _fake_reconcile_all(self):
            return [
                ReconciliationReport(node_id="node-a", verified=1),
                ReconciliationReport(node_id="node-b", verified=0, fetch_error="manifest_fetch_failed"),
            ]
        monkeypatch.setattr(
            "app.services.storage_reconciler.ManifestReconciler.reconcile_all",
            _fake_reconcile_all,
        )

        async def _run():
            async with _loopback_client(app) as client:
                return await client.post("/fleet/storage/reconcile")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        ids = sorted(r["node_id"] for r in body["reports"])
        assert ids == ["node-a", "node-b"]
        # fetch_error surfaces through the response
        node_b = next(r for r in body["reports"] if r["node_id"] == "node-b")
        assert node_b["fetch_error"] == "manifest_fetch_failed"

    def test_remote_client_gets_403(self, isolated_data_dir, monkeypatch):
        app = _build_app()
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity", lambda: "op@x"
        )

        async def _run():
            async with _remote_client(app) as client:
                return await client.post("/fleet/storage/reconcile")

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
                return await client.post("/fleet/storage/reconcile")

        import asyncio
        resp = asyncio.run(_run())
        assert resp.status_code == 503
