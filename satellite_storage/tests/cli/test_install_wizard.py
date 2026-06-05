"""Tests for ``molebie-satellite install`` — the 6-phase wizard."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import typer

from satellite_storage.cli import install as install_mod
from satellite_storage.cli.network import TailscaleWhoamiInfo


@dataclass
class _FakeResp:
    status_code: int
    text: str = ""
    _json: dict | None = None

    def json(self):
        return self._json or {}


@pytest.fixture
def good_identity(monkeypatch):
    monkeypatch.setattr(
        install_mod, "get_tailscale_whoami",
        lambda: TailscaleWhoamiInfo("ops@example.com", "Ops"),
    )
    monkeypatch.setattr(install_mod, "get_tailscale_ip", lambda: "100.64.0.5")


@pytest.fixture
def primary_reachable(monkeypatch):
    """Make /health probes return 200 from the primary, and the register POST 200."""
    def _get(url, **kwargs):
        if "/health" in url:
            return _FakeResp(200)
        if "/fleet/inventory" in url:
            return _FakeResp(200, _json={"satellites": [{"host": "100.64.0.5"}]})
        return _FakeResp(404, text="not configured")

    def _post(url, **kwargs):
        return _FakeResp(
            200, _json={"id": "abc1234567890", "host": "100.64.0.5", "role": "storage"},
        )

    monkeypatch.setattr(httpx, "get", _get)
    monkeypatch.setattr(httpx, "post", _post)


class TestInstallWizardForeground:
    def test_foreground_skips_service_install(
        self, good_identity, primary_reachable, monkeypatch, tmp_path
    ):
        # Service-install dispatcher would fail on this host without mocking;
        # foreground path avoids touching it at all.
        boom_called = [False]

        def _boom(*_a, **_kw):
            boom_called[0] = True
            raise RuntimeError("install_service should NOT be called in --foreground")

        monkeypatch.setattr(install_mod, "install_service", _boom)
        monkeypatch.setattr(install_mod, "find_satellite_binary", lambda: Path("/usr/bin/molebie-satellite"))

        install_mod.install_command(
            primary="100.64.0.1", port=8000, role="storage",
            label="nas", data_dir=tmp_path / "data",
            foreground=True, yes=True,
        )

        assert boom_called[0] is False


class TestInstallWizardServiceMode:
    def test_happy_path_calls_install_service_and_registers(
        self, good_identity, primary_reachable, monkeypatch, tmp_path,
    ):
        installed = [False]

        def _install(cfg):
            installed[0] = True
            return f"{tmp_path}/com.molebieai.satellite.plist"

        monkeypatch.setattr(install_mod, "install_service", _install)
        monkeypatch.setattr(install_mod, "find_satellite_binary", lambda: Path("/usr/bin/molebie-satellite"))
        # Skip the real health-poll loop.
        monkeypatch.setattr(install_mod, "_wait_for_health", lambda *a, **k: True)

        install_mod.install_command(
            primary="100.64.0.1", port=8000, role="storage",
            label="nas", data_dir=tmp_path / "data",
            foreground=False, yes=True,
        )

        assert installed[0] is True


class TestInstallWizardRefusals:
    def test_invalid_role_exits(self, tmp_path):
        with pytest.raises(typer.Exit) as exc:
            install_mod.install_command(
                primary="100.64.0.1", port=8000, role="not-real",
                label=None, data_dir=tmp_path / "data",
                foreground=False, yes=True,
            )
        assert exc.value.exit_code == 1

    def test_no_tailscale_identity_exits(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(install_mod, "get_tailscale_whoami", lambda: None)
        with pytest.raises(typer.Exit) as exc:
            install_mod.install_command(
                primary="100.64.0.1", port=8000, role="storage",
                label=None, data_dir=tmp_path / "data",
                foreground=False, yes=True,
            )
        assert exc.value.exit_code == 1

    def test_unreachable_primary_exits(
        self, good_identity, monkeypatch, tmp_path
    ):
        def _get(*a, **k):
            raise httpx.ConnectError("primary down")
        monkeypatch.setattr(httpx, "get", _get)

        with pytest.raises(typer.Exit) as exc:
            install_mod.install_command(
                primary="100.64.0.1", port=8000, role="storage",
                label=None, data_dir=tmp_path / "data",
                foreground=False, yes=True,
            )
        assert exc.value.exit_code == 1

    def test_missing_satellite_binary_exits(
        self, good_identity, primary_reachable, monkeypatch, tmp_path
    ):
        # Skip the prereq + data dir phases; fail at find_satellite_binary.
        monkeypatch.setattr(install_mod, "find_satellite_binary", lambda: None)
        with pytest.raises(typer.Exit) as exc:
            install_mod.install_command(
                primary="100.64.0.1", port=8000, role="storage",
                label=None, data_dir=tmp_path / "data",
                foreground=False, yes=True,
            )
        assert exc.value.exit_code == 1

    def test_service_install_failure_exits(
        self, good_identity, primary_reachable, monkeypatch, tmp_path
    ):
        from satellite_storage.cli.service import ServiceInstallError

        def _install(_cfg):
            raise ServiceInstallError("launchctl bootstrap failed")

        monkeypatch.setattr(install_mod, "install_service", _install)
        monkeypatch.setattr(install_mod, "find_satellite_binary", lambda: Path("/usr/bin/molebie-satellite"))

        with pytest.raises(typer.Exit) as exc:
            install_mod.install_command(
                primary="100.64.0.1", port=8000, role="storage",
                label=None, data_dir=tmp_path / "data",
                foreground=False, yes=True,
            )
        assert exc.value.exit_code == 1


def test_python_version_check_passes_on_current(
    good_identity, primary_reachable, monkeypatch, tmp_path,
):
    """Sanity: we run on Python 3.13+, so the prereq check should not trip."""
    monkeypatch.setattr(install_mod, "install_service", lambda _c: "/tmp/x.plist")
    monkeypatch.setattr(install_mod, "find_satellite_binary", lambda: Path("/usr/bin/molebie-satellite"))
    monkeypatch.setattr(install_mod, "_wait_for_health", lambda *a, **k: True)

    install_mod.install_command(
        primary="100.64.0.1", port=8000, role="storage",
        label=None, data_dir=tmp_path / "data",
        foreground=True, yes=True,
    )
    # Reaching here means the sys.version_info >= (3,13) gate didn't reject us.
    assert sys.version_info >= (3, 13)
