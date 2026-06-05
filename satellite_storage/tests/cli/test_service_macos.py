"""Tests for ``service_macos.py`` — launchd plist install + uninstall."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from satellite_storage.cli import service_macos
from satellite_storage.cli.service import (
    SATELLITE_SERVICE_LABEL,
    ServiceConfig,
    ServiceInstallError,
)


@pytest.fixture
def cfg(tmp_path):
    return ServiceConfig(
        satellite_bin=Path("/usr/local/bin/molebie-satellite"),
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        home_dir=tmp_path / "home",
    )


@pytest.fixture
def patched_launch_agents_dir(tmp_path, monkeypatch):
    """Redirect LaunchAgents writes to a tempdir so we don't touch the real one."""
    agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(service_macos, "_LAUNCH_AGENTS_DIR", agents)
    return agents


def _success_subprocess(*args, **kwargs):
    return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")


def _failure_subprocess(stderr_msg: str):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr=stderr_msg
        )
    return _run


class TestInstall:
    def test_writes_plist_and_bootstraps(
        self, cfg, patched_launch_agents_dir, monkeypatch
    ):
        calls: list[list[str]] = []

        def _run(args, **kwargs):
            calls.append(list(args))
            return _success_subprocess(args)

        monkeypatch.setattr(subprocess, "run", _run)
        result = service_macos.install_service(cfg)

        plist_path = patched_launch_agents_dir / f"{SATELLITE_SERVICE_LABEL}.plist"
        assert plist_path.exists()
        assert result == str(plist_path)
        # The bootstrap call must have happened; bootout is best-effort first.
        bootstrap_calls = [c for c in calls if "bootstrap" in c]
        assert len(bootstrap_calls) == 1
        assert str(plist_path) in bootstrap_calls[0]

    def test_raises_on_bootstrap_failure(
        self, cfg, patched_launch_agents_dir, monkeypatch
    ):
        # Make bootout pass and bootstrap fail.
        def _run(args, **kwargs):
            if "bootstrap" in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=1, stdout="", stderr="boom",
                )
            return _success_subprocess(args)

        monkeypatch.setattr(subprocess, "run", _run)
        with pytest.raises(ServiceInstallError, match="boom"):
            service_macos.install_service(cfg)


class TestUninstall:
    def test_idempotent_on_missing(self, patched_launch_agents_dir, monkeypatch):
        # Nothing on disk; uninstall must not raise.
        monkeypatch.setattr(subprocess, "run", _success_subprocess)
        service_macos.uninstall_service(SATELLITE_SERVICE_LABEL)

    def test_deletes_plist_and_calls_bootout(
        self, patched_launch_agents_dir, monkeypatch
    ):
        plist = patched_launch_agents_dir / f"{SATELLITE_SERVICE_LABEL}.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_text("<plist/>")

        called: list[list[str]] = []

        def _run(args, **kwargs):
            called.append(list(args))
            return _success_subprocess(args)

        monkeypatch.setattr(subprocess, "run", _run)
        service_macos.uninstall_service(SATELLITE_SERVICE_LABEL)

        assert not plist.exists()
        assert any("bootout" in c for c in called)


def test_is_service_installed_reflects_file(patched_launch_agents_dir):
    plist = patched_launch_agents_dir / f"{SATELLITE_SERVICE_LABEL}.plist"
    assert service_macos.is_service_installed(SATELLITE_SERVICE_LABEL) is False
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text("<plist/>")
    assert service_macos.is_service_installed(SATELLITE_SERVICE_LABEL) is True
