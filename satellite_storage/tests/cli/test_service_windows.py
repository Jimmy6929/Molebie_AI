"""Tests for ``service_windows.py`` — Windows Task Scheduler install/uninstall.

These tests run on any host (we mock subprocess); the goal is to verify
that the right ``schtasks.exe`` commands are issued, that XML rendering
works, and — most importantly — that ``install_service`` returns a stable
identifier rather than the unlinked temp XML path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from satellite_storage.cli import service_windows
from satellite_storage.cli.service import (
    SATELLITE_SERVICE_LABEL,
    ServiceConfig,
    ServiceInstallError,
)


@pytest.fixture
def cfg(tmp_path):
    return ServiceConfig(
        satellite_bin=Path("C:/molebie/molebie-satellite.exe"),
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        home_dir=tmp_path / "home",
    )


def _capture_run(calls: list):
    def _run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
    return _run


class TestInstall:
    def test_returns_scheduler_identifier_not_temp_path(self, cfg, monkeypatch):
        """The headline behavior from the previous-review fix: even though
        ``install_service`` writes a temp XML and then unlinks it, the
        returned string must be a stable Task Scheduler identifier."""
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _capture_run(calls))
        result = service_windows.install_service(cfg)
        assert result == f"Scheduled Task: {service_windows.WINDOWS_TASK_NAME}"

    def test_calls_schtasks_create_and_run(self, cfg, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _capture_run(calls))
        service_windows.install_service(cfg)

        # /Create came with /XML + /F flags.
        creates = [c for c in calls if "/Create" in c]
        assert len(creates) == 1
        create = creates[0]
        assert "/TN" in create and service_windows.WINDOWS_TASK_NAME in create
        assert "/XML" in create
        assert "/F" in create
        # The task is kicked immediately so the operator doesn't wait for re-logon.
        assert any("/Run" in c for c in calls)

    def test_raises_on_schtasks_create_failure(self, cfg, monkeypatch):
        def _run(args, **kwargs):
            if "/Create" in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=1, stdout="", stderr="ACCESS DENIED",
                )
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr="",
            )

        monkeypatch.setattr(subprocess, "run", _run)
        with pytest.raises(ServiceInstallError, match="ACCESS DENIED"):
            service_windows.install_service(cfg)


class TestUninstall:
    def test_calls_schtasks_delete(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _capture_run(calls))
        service_windows.uninstall_service(SATELLITE_SERVICE_LABEL)
        deletes = [c for c in calls if "/Delete" in c]
        assert len(deletes) == 1
        assert service_windows.WINDOWS_TASK_NAME in deletes[0]
        assert "/F" in deletes[0]  # force, no confirmation prompt


class TestIsInstalled:
    def test_returns_true_on_query_zero_exit(self, monkeypatch):
        def _run(args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        monkeypatch.setattr(subprocess, "run", _run)
        assert service_windows.is_service_installed(SATELLITE_SERVICE_LABEL) is True

    def test_returns_false_on_query_nonzero_exit(self, monkeypatch):
        def _run(args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="not found")
        monkeypatch.setattr(subprocess, "run", _run)
        assert service_windows.is_service_installed(SATELLITE_SERVICE_LABEL) is False
