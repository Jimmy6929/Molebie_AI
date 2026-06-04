"""macOS launchd implementation for the satellite service.

Installs the satellite blob service as a per-user LaunchAgent at
``~/Library/LaunchAgents/<label>.plist``. Loads it via the modern
``launchctl bootstrap gui/<uid> <path>`` API (not the deprecated
``launchctl load``). KeepAlive is on, so the service auto-restarts on
crash and at login.

The plist template lives at ``satellite_storage/templates/com.molebieai.satellite.plist``
with ``__LABEL__`` / ``__SATELLITE_BIN__`` / ``__DATA_DIR__`` /
``__LOG_DIR__`` / ``__HOME_DIR__`` placeholders.
"""

from __future__ import annotations

import os
import subprocess
from importlib import resources
from pathlib import Path

from satellite_storage.cli.service import (
    ServiceConfig,
    ServiceInstallError,
    render_template,
)

_LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _plist_path(label: str) -> Path:
    return _LAUNCH_AGENTS_DIR / f"{label}.plist"


def _domain_target() -> str:
    return f"gui/{os.getuid()}"


def _service_target(label: str) -> str:
    return f"{_domain_target()}/{label}"


def _read_template() -> str:
    """Load the plist template shipped with the package."""
    return (
        resources.files("satellite_storage")
        .joinpath("templates", "com.molebieai.satellite.plist")
        .read_text(encoding="utf-8")
    )


def install_service(config: ServiceConfig) -> str:
    """Render the plist, write it under ~/Library/LaunchAgents, bootstrap.

    If the service is already loaded, bootout first (idempotent reinstall).
    Returns the plist's absolute path as a string — survives on disk so
    operators can ``cat`` it for debugging.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    _LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    plist_text = render_template(
        _read_template(),
        mapping={
            "LABEL": config.label,
            "SATELLITE_BIN": str(config.satellite_bin),
            "DATA_DIR": str(config.data_dir),
            "LOG_DIR": str(config.log_dir),
            "HOME_DIR": str(config.home_dir),
        },
    )

    plist = _plist_path(config.label)
    plist.write_text(plist_text, encoding="utf-8")

    # Idempotent reinstall: bootout if loaded (ignore failure — likely not loaded).
    subprocess.run(
        ["launchctl", "bootout", _service_target(config.label)],
        capture_output=True, text=True,
    )

    result = subprocess.run(
        ["launchctl", "bootstrap", _domain_target(), str(plist)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ServiceInstallError(
            f"launchctl bootstrap failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return str(plist)


def uninstall_service(label: str) -> None:
    """Bootout + delete the plist file. Idempotent — both steps tolerate absence."""
    subprocess.run(
        ["launchctl", "bootout", _service_target(label)],
        capture_output=True, text=True,
    )
    plist = _plist_path(label)
    if plist.exists():
        plist.unlink()


def is_service_installed(label: str) -> bool:
    """True if the plist file is on disk (sufficient signal at v0.2 scale)."""
    return _plist_path(label).exists()
