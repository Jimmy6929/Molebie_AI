"""Windows Task Scheduler implementation for the satellite service.

Installs the satellite as a Scheduled Task named ``MolebieSatellite`` with
an ``ONLOGON`` trigger, using the built-in ``schtasks.exe`` — no third-party
service manager (NSSM) required.

The task XML lives at
``satellite_storage/templates/molebie-satellite-task.xml`` with
``__USER__`` / ``__SATELLITE_BIN__`` / ``__DATA_DIR__`` placeholders.
The data dir's ``MOLEBIE_STORAGE_DATA_DIR`` env var is set at the task's
working-directory level via the satellite's own config (``WorkingDirectory``
ensures the relative-path fallback hits the right dir).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from importlib import resources
from pathlib import Path

from satellite_storage.cli._service_base import (
    ServiceConfig,
    ServiceInstallError,
    render_template,
)

# Windows tasks use a flat PascalCase identifier, not reverse-DNS.
WINDOWS_TASK_NAME = "MolebieSatellite"


def _read_template() -> str:
    return (
        resources.files("satellite_storage")
        .joinpath("templates", "molebie-satellite-task.xml")
        .read_text(encoding="utf-8")
    )


def _current_user_qualified() -> str:
    """``DOMAIN\\user`` form schtasks expects. Falls back to ``USERNAME`` when
    USERDOMAIN isn't set (rare, mostly local accounts)."""
    domain = os.environ.get("USERDOMAIN")
    user = os.environ.get("USERNAME") or os.environ.get("USER") or "Unknown"
    return f"{domain}\\{user}" if domain else user


def _schtasks(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["schtasks.exe", *args],
        capture_output=True, text=True,
    )


def install_service(config: ServiceConfig) -> str:
    """Render the task XML, write it to a temp file, register via ``schtasks /create /xml``.

    Returns a stable description of the registered task (the Task Scheduler
    identifier, not a file path — the source XML is unlinked after
    registration because the task lives in the scheduler registry).
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    xml_text = render_template(
        _read_template(),
        mapping={
            "USER": _current_user_qualified(),
            "SATELLITE_BIN": str(config.satellite_bin),
            "DATA_DIR": str(config.data_dir),
        },
    )

    # schtasks needs a file on disk; the task is stored in the registry so we
    # can drop the file after registration. Use UTF-16 (required by schtasks
    # for the XML schema's encoding declaration).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, encoding="utf-16",
    ) as f:
        f.write(xml_text)
        xml_path = Path(f.name)

    try:
        result = _schtasks(
            "/Create", "/TN", WINDOWS_TASK_NAME,
            "/XML", str(xml_path), "/F",  # /F = overwrite if exists
        )
        if result.returncode != 0:
            raise ServiceInstallError(
                f"schtasks /Create failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        # Kick the task immediately so the satellite is up without waiting for re-logon.
        _schtasks("/Run", "/TN", WINDOWS_TASK_NAME)
    finally:
        try:
            xml_path.unlink()
        except OSError:
            # Best-effort temp cleanup — the task is already registered in
            # the scheduler registry, so a leftover XML must not fail install.
            pass

    # Task lives in the scheduler registry, not on disk — return a stable
    # identifier the wizard can show the operator instead of a vanished temp path.
    return f"Scheduled Task: {WINDOWS_TASK_NAME}"


def uninstall_service(label: str) -> None:
    """``schtasks /Delete /F``. Idempotent: ignores 'task not found'.

    ``label`` is accepted for cross-platform dispatcher uniformity but
    ignored: Windows Scheduled Tasks use a flat PascalCase identifier
    (``MolebieSatellite``), not the dispatcher's macOS-style reverse-DNS
    default, so this module always uses ``WINDOWS_TASK_NAME``.
    """
    _schtasks("/Delete", "/TN", WINDOWS_TASK_NAME, "/F")


def is_service_installed(label: str) -> bool:
    """``schtasks /Query`` returns 0 if present, non-zero otherwise.
    ``label`` accepted for dispatcher uniformity but ignored."""
    result = _schtasks("/Query", "/TN", WINDOWS_TASK_NAME)
    return result.returncode == 0
