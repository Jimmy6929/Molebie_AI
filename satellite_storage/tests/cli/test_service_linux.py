"""Tests for ``service_linux.py`` — systemd user unit install + uninstall."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from satellite_storage.cli import service_linux
from satellite_storage.cli.service import (
    SATELLITE_SERVICE_LABEL,
    ServiceConfig,
    ServiceInstallError,
)


@pytest.fixture
def cfg(tmp_path):
    return ServiceConfig(
        satellite_bin=Path("/usr/bin/molebie-satellite"),
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        home_dir=tmp_path / "home",
    )


@pytest.fixture
def patched_units_dir(tmp_path, monkeypatch):
    units = tmp_path / "systemd" / "user"
    monkeypatch.setattr(service_linux, "_USER_UNITS_DIR", units)
    return units


def _capture_run(calls: list):
    def _run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
    return _run


class TestInstall:
    def test_writes_unit_and_enables(self, cfg, patched_units_dir, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _capture_run(calls))
        result = service_linux.install_service(cfg)

        unit = patched_units_dir / f"{service_linux.LINUX_UNIT_NAME}.service"
        assert unit.exists()
        assert result == str(unit)
        # daemon-reload + enable --now both fired.
        assert any("daemon-reload" in c for c in calls)
        assert any("enable" in c and "--now" in c for c in calls)

    def test_raises_on_enable_failure(self, cfg, patched_units_dir, monkeypatch):
        def _run(args, **kwargs):
            # First call (daemon-reload) succeeds; second call (enable) fails.
            if "enable" in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=1, stdout="", stderr="systemd: nope",
                )
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr="",
            )

        monkeypatch.setattr(subprocess, "run", _run)
        with pytest.raises(ServiceInstallError, match="systemd: nope"):
            service_linux.install_service(cfg)


class TestUninstall:
    def test_ignores_label_param(self, patched_units_dir, monkeypatch):
        """The systemd module always operates on LINUX_UNIT_NAME regardless
        of what the dispatcher's macOS-style default passes in."""
        # Even when called with the macOS-style label, it should look up the
        # systemd unit file at the LINUX_UNIT_NAME path.
        unit = patched_units_dir / f"{service_linux.LINUX_UNIT_NAME}.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text("[Unit]\n")

        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _capture_run(calls))
        service_linux.uninstall_service(SATELLITE_SERVICE_LABEL)

        assert not unit.exists()
        assert any("disable" in c and "--now" in c for c in calls)

    def test_idempotent_on_missing(self, patched_units_dir, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _capture_run([]))
        service_linux.uninstall_service(SATELLITE_SERVICE_LABEL)


def test_is_service_installed_reflects_file(patched_units_dir):
    unit = patched_units_dir / f"{service_linux.LINUX_UNIT_NAME}.service"
    assert service_linux.is_service_installed(SATELLITE_SERVICE_LABEL) is False
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("[Unit]\n")
    assert service_linux.is_service_installed(SATELLITE_SERVICE_LABEL) is True
