"""Integration tests for ``GET /fleet/inventory``.

Loopback-gated; lists every satellite registered via
``POST /fleet/satellites/register``. The ``tailscale_user`` column is
intentionally excluded from the response — that lives in the audit log.
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
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 0))
    return httpx.AsyncClient(transport=transport, base_url="http://primary")


def _tailnet_client(app, peer_ip: str = "100.64.0.5"):
    transport = httpx.ASGITransport(app=app, client=(peer_ip, 0))
    return httpx.AsyncClient(transport=transport, base_url="http://primary")


_AUTH = {"Tailscale-User-Login": "jimmy@github"}


class TestFleetInventory:
    def test_returns_all_registered_satellites(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as sat_client:
                await sat_client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH,
                    json={"host": "home-server", "role": "storage",
                          "capabilities": {"disk_gb": 500}},
                )
                await sat_client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH,
                    json={"host": "gpu-box", "role": "compute",
                          "capabilities": {"vram_gb": 24}},
                )
            async with _loopback_client(app) as ops_client:
                return await ops_client.get("/fleet/inventory")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        hosts = {s["host"] for s in body["satellites"]}
        assert hosts == {"home-server", "gpu-box"}

    def test_empty_fleet_returns_zero(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _loopback_client(app) as client:
                return await client.get("/fleet/inventory")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        assert resp.json() == {"satellites": [], "count": 0}

    def test_remote_client_gets_403(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as client:
                return await client.get("/fleet/inventory")

        resp = asyncio.run(_run())
        assert resp.status_code == 403
        assert "loopback" in resp.json()["detail"].lower()

    def test_capabilities_parsed_as_dict(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as sat_client:
                await sat_client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH,
                    json={"host": "h", "role": "both",
                          "capabilities": {"disk_gb": 1000, "models": ["a", "b"]}},
                )
            async with _loopback_client(app) as ops_client:
                return await ops_client.get("/fleet/inventory")

        resp = asyncio.run(_run())
        body = resp.json()
        caps = body["satellites"][0]["capabilities"]
        assert caps == {"disk_gb": 1000, "models": ["a", "b"]}

    def test_tailscale_user_absent_from_response(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as sat_client:
                await sat_client.post(
                    "/fleet/satellites/register",
                    headers={"Tailscale-User-Login": "secret-user@example.com"},
                    json={"host": "h", "role": "storage"},
                )
            async with _loopback_client(app) as ops_client:
                return await ops_client.get("/fleet/inventory")

        resp = asyncio.run(_run())
        body = resp.json()
        # Metadata-only invariant: who registered the satellite is in the
        # audit log, not surfaced to every inventory reader.
        assert "tailscale_user" not in body["satellites"][0]
        # And the value definitely doesn't leak via any other field.
        flat = str(body)
        assert "secret-user@example.com" not in flat
