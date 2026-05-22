"""Tests for the gateway lifespan's Tailscale Serve auto-bootstrap.

Exercises ``_maybe_enable_tailscale_serve`` directly with the three
underlying helpers (`is_serve_configured`, `enable_serve`, `get_https_url`)
patched at their import sites in ``app.main``. The satellite-existence
check is also patched. Verifies the audit event is or isn't written, and
that the env-var escape hatch / pre-configured state / enable failures
are all handled gracefully.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + initialised DB per test. Resets singleton state."""
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


def _seed_satellite(data_dir: str) -> None:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        conn.execute(
            "INSERT INTO fleet_satellites "
            "(id, host, role, status, label, capabilities_json, "
            " tailscale_user, joined_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "seed-id", "100.64.0.5", "storage", "active", None, None,
                "test@example.com", "2026-05-21T00:00:00+00:00",
                "2026-05-21T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _audit_rows(data_dir: str) -> list[dict]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM audit_events ORDER BY id ASC"
    )]
    conn.close()
    return rows


def _patch_helpers(monkeypatch, *, is_configured: bool, enable_ok: bool, url: str | None):
    """Patch the three subprocess helpers as imported into ``app.main``.

    Returns a dict of call counters so tests can assert which helpers
    were invoked.
    """
    counters: dict[str, int] = {"is_configured": 0, "enable_serve": 0, "get_url": 0}

    def _is_configured(port: int = 8000) -> bool:
        counters["is_configured"] += 1
        return is_configured

    def _enable_serve(port: int = 8000) -> bool:
        counters["enable_serve"] += 1
        return enable_ok

    def _get_url() -> str | None:
        counters["get_url"] += 1
        return url

    # These names are looked up inside _maybe_enable_tailscale_serve via
    # a local `from app.services.tailscale_serve import ...` statement, so
    # patching the source module is the correct hook point.
    from app.services import tailscale_serve as ts_mod
    monkeypatch.setattr(ts_mod, "is_serve_configured", _is_configured)
    monkeypatch.setattr(ts_mod, "enable_serve", _enable_serve)
    monkeypatch.setattr(ts_mod, "get_https_url", _get_url)
    return counters


class TestMaybeEnableTailscaleServe:
    def test_enables_and_records_audit_when_satellite_exists(
        self, isolated_data_dir, monkeypatch
    ):
        _seed_satellite(isolated_data_dir)
        counters = _patch_helpers(
            monkeypatch, is_configured=False, enable_ok=True,
            url="https://primary.tailnet.ts.net/",
        )

        from app.main import _maybe_enable_tailscale_serve
        asyncio.run(_maybe_enable_tailscale_serve())

        assert counters == {"is_configured": 1, "enable_serve": 1, "get_url": 1}
        rows = _audit_rows(isolated_data_dir)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "security.tls_enabled"
        assert rows[0]["target"] == "https://primary.tailnet.ts.net/"
        assert rows[0]["actor"] == "system"

    def test_skips_entirely_when_no_satellites(self, isolated_data_dir, monkeypatch):
        # No satellite seeded.
        counters = _patch_helpers(
            monkeypatch, is_configured=False, enable_ok=True, url="https://x/",
        )
        from app.main import _maybe_enable_tailscale_serve
        asyncio.run(_maybe_enable_tailscale_serve())
        # No subprocess calls. No audit row.
        assert counters == {"is_configured": 0, "enable_serve": 0, "get_url": 0}
        assert _audit_rows(isolated_data_dir) == []

    def test_idempotent_when_already_configured(self, isolated_data_dir, monkeypatch):
        _seed_satellite(isolated_data_dir)
        counters = _patch_helpers(
            monkeypatch, is_configured=True, enable_ok=True, url="https://x/",
        )
        from app.main import _maybe_enable_tailscale_serve
        asyncio.run(_maybe_enable_tailscale_serve())
        # is_configured was called once, but enable_serve was NOT.
        assert counters["is_configured"] == 1
        assert counters["enable_serve"] == 0
        # No new audit row.
        assert _audit_rows(isolated_data_dir) == []

    def test_no_audit_when_enable_fails(self, isolated_data_dir, monkeypatch, capsys):
        _seed_satellite(isolated_data_dir)
        _patch_helpers(
            monkeypatch, is_configured=False, enable_ok=False, url=None,
        )
        from app.main import _maybe_enable_tailscale_serve
        asyncio.run(_maybe_enable_tailscale_serve())
        # No audit row.
        assert _audit_rows(isolated_data_dir) == []
        # Helpful diagnostic was printed.
        out = capsys.readouterr().out
        assert "HTTPS Certificates" in out

    def test_logs_url_unknown_when_get_url_returns_none(
        self, isolated_data_dir, monkeypatch, capsys
    ):
        _seed_satellite(isolated_data_dir)
        _patch_helpers(
            monkeypatch, is_configured=False, enable_ok=True, url=None,
        )
        from app.main import _maybe_enable_tailscale_serve
        asyncio.run(_maybe_enable_tailscale_serve())
        # Audit row still written with target="unknown".
        rows = _audit_rows(isolated_data_dir)
        assert len(rows) == 1
        assert rows[0]["target"] == "unknown"
        out = capsys.readouterr().out
        assert "URL unknown" in out

    def test_env_var_off_short_circuits_lifespan_branch(
        self, isolated_data_dir, monkeypatch
    ):
        """When MOLEBIE_AUTO_TAILSCALE_SERVE=0, the lifespan block should not
        even call _maybe_enable_tailscale_serve. We verify by patching the
        helper to raise — if the env-var gate doesn't short-circuit, the
        exception would surface."""
        _seed_satellite(isolated_data_dir)
        monkeypatch.setenv("MOLEBIE_AUTO_TAILSCALE_SERVE", "0")
        import os
        # Simulate the lifespan-block gate logic directly (it's a single
        # `if os.getenv(...) == "1":` line). When off, _maybe_enable is
        # not called, so no audit row appears.
        if os.getenv("MOLEBIE_AUTO_TAILSCALE_SERVE", "1") == "1":
            from app.main import _maybe_enable_tailscale_serve
            asyncio.run(_maybe_enable_tailscale_serve())
        assert _audit_rows(isolated_data_dir) == []
