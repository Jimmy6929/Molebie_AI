"""Integration tests for ``POST /fleet/satellites/register``.

Mounts the FastAPI app and exercises register via httpx's ASGI
transport. Identity is injected via the ``Tailscale-User-Login`` header;
the route is NOT loopback-gated, so non-loopback client IPs are valid.
"""

from __future__ import annotations

import asyncio
import sqlite3
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


def _tailnet_client(app, peer_ip: str = "100.64.0.5"):
    """Client with a non-loopback IP (simulates a satellite over Tailscale)."""
    transport = httpx.ASGITransport(app=app, client=(peer_ip, 0))
    return httpx.AsyncClient(transport=transport, base_url="http://primary")


def _audit_rows(data_dir: str) -> list[dict]:
    db = sqlite3.connect(Path(data_dir) / "molebie.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT * FROM audit_events ORDER BY id ASC"
    )]
    db.close()
    return rows


def _fleet_rows(data_dir: str) -> list[dict]:
    db = sqlite3.connect(Path(data_dir) / "molebie.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT * FROM fleet_satellites ORDER BY joined_at ASC"
    )]
    db.close()
    return rows


_AUTH_HEADERS = {
    "Tailscale-User-Login": "jimmy@github",
    "Tailscale-User-Name": "Jimmy Z",
}


class TestRegisterSatellite:
    def test_first_registration_creates_row_and_audit_event(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as client:
                return await client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH_HEADERS,
                    json={
                        "host": "home-server",
                        "role": "storage",
                        "label": "Living-room NAS",
                        "capabilities": {"disk_gb": 500},
                    },
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["host"] == "home-server"
        assert body["role"] == "storage"
        assert body["status"] == "active"
        assert body["label"] == "Living-room NAS"
        assert body["capabilities"] == {"disk_gb": 500}
        assert body["id"]  # UUID assigned
        assert body["joined_at"]
        assert body["updated_at"] == body["joined_at"]  # first registration

        rows = _fleet_rows(isolated_data_dir)
        assert len(rows) == 1
        assert rows[0]["tailscale_user"] == "jimmy@github"

        audit = _audit_rows(isolated_data_dir)
        assert len(audit) == 1
        assert audit[0]["event_type"] == "satellite.join"
        assert audit[0]["actor"] == "jimmy@github"
        assert audit[0]["target"] == "home-server"

    def test_reregistration_updates_row_and_emits_update_event(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as client:
                first = await client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH_HEADERS,
                    json={
                        "host": "home-server",
                        "role": "storage",
                        "capabilities": {"disk_gb": 500},
                    },
                )
                second = await client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH_HEADERS,
                    json={
                        "host": "home-server",
                        "role": "both",
                        "capabilities": {"disk_gb": 750, "models_loaded": ["qwen3-thinking"]},
                    },
                )
                return first, second

        first, second = asyncio.run(_run())
        assert first.status_code == 200
        assert second.status_code == 200

        first_body = first.json()
        second_body = second.json()
        assert second_body["id"] == first_body["id"]  # id preserved
        assert second_body["role"] == "both"  # role updated
        assert second_body["capabilities"]["disk_gb"] == 750  # capabilities refreshed
        assert second_body["joined_at"] == first_body["joined_at"]
        assert second_body["updated_at"] >= first_body["updated_at"]

        rows = _fleet_rows(isolated_data_dir)
        assert len(rows) == 1  # still one row

        audit = _audit_rows(isolated_data_dir)
        event_types = [a["event_type"] for a in audit]
        assert event_types == ["satellite.join", "satellite.update"]

    def test_missing_identity_header_returns_401(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as client:
                return await client.post(
                    "/fleet/satellites/register",
                    json={"host": "x", "role": "storage"},
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 401
        assert "Tailscale-User-Login" in resp.json()["detail"]

    def test_invalid_role_returns_422(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as client:
                return await client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH_HEADERS,
                    json={"host": "x", "role": "router"},  # not in {storage, compute, both}
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 422

    def test_capabilities_round_trip_as_dict(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            payload = {
                "host": "gpu-box",
                "role": "compute",
                "capabilities": {
                    "gpu": "RTX 4090",
                    "vram_gb": 24,
                    "models_loaded": ["qwen3-thinking", "qwen3-instant"],
                },
            }
            async with _tailnet_client(app) as client:
                return await client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH_HEADERS,
                    json=payload,
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        caps = resp.json()["capabilities"]
        assert caps["gpu"] == "RTX 4090"
        assert caps["vram_gb"] == 24
        assert caps["models_loaded"] == ["qwen3-thinking", "qwen3-instant"]

    def test_register_works_from_non_loopback_ip(self, isolated_data_dir):
        """Register MUST be reachable from remote tailnet IPs — that's the
        whole point. Verify a non-loopback peer is not blocked."""
        app = _build_app()

        async def _run():
            async with _tailnet_client(app, peer_ip="100.103.5.42") as client:
                return await client.post(
                    "/fleet/satellites/register",
                    headers=_AUTH_HEADERS,
                    json={"host": "remote", "role": "storage"},
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200

    def test_tailscale_user_persisted_on_row(self, isolated_data_dir):
        app = _build_app()

        async def _run():
            async with _tailnet_client(app) as client:
                return await client.post(
                    "/fleet/satellites/register",
                    headers={"Tailscale-User-Login": "alice@example.com"},
                    json={"host": "alice-box", "role": "compute"},
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        rows = _fleet_rows(isolated_data_dir)
        assert rows[0]["tailscale_user"] == "alice@example.com"
        # And inventory response must NOT leak it.
        assert "tailscale_user" not in resp.json()
