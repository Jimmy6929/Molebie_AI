"""Cross-platform service install dispatcher for the satellite blob service.

Detects ``sys.platform`` at install time and routes to the platform-specific
module:

* **macOS** (``darwin``) — launchd user agent at
  ``~/Library/LaunchAgents/com.molebieai.satellite.plist``.
* **Linux** (``linux*``) — systemd user unit at
  ``~/.config/systemd/user/molebie-satellite.service``.
* **Windows** (``win32``) — Scheduled Task ``MolebieSatellite`` with an
  ``ONLOGON`` trigger via the built-in ``schtasks.exe``.

A single ``ServiceConfig`` dataclass carries everything any platform module
needs (label, satellite binary path, data dir, log dir, home dir). Each
module is responsible for its own template rendering + subprocess invocations.

The dispatcher uses lazy imports so a Windows-only or Linux-only module
never gets loaded on macOS — keeps the import cost low and avoids accidental
circular references in test setup.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Re-exported from the leaf module so callers can keep importing these names
# from satellite_storage.cli.service unchanged. The shared types live in
# _service_base to break the static cycle between this dispatcher and the
# platform-specific modules.
from satellite_storage.cli._service_base import (
    SATELLITE_SERVICE_LABEL,
    ServiceConfig,
    ServiceInstallError,
    render_template,
)


class PlatformNotSupportedError(Exception):
    """Raised when sys.platform isn't one of darwin/linux*/win32."""


def find_satellite_binary() -> Path | None:
    """Locate the ``molebie-satellite`` executable on PATH (pipx put it there).

    Tests can monkeypatch this; callers that fail to find it should surface
    a clear error pointing the operator at ``pipx ensurepath``.
    """
    located = shutil.which("molebie-satellite")
    return Path(located) if located else None


def default_data_dir() -> Path:
    return Path.home() / ".molebie" / "satellite-storage"


def default_log_dir() -> Path:
    return Path.home() / ".molebie" / "logs"


# ── dispatcher API ────────────────────────────────────────────────────


def install_service(config: ServiceConfig) -> str:
    """Install the satellite as an OS service. Returns a stable description
    of the registered unit (a file path on macOS / Linux, a Task Scheduler
    identifier on Windows) — suitable for printing to the operator."""
    return _platform_module().install_service(config)


def uninstall_service(label: str = SATELLITE_SERVICE_LABEL) -> None:
    """Remove the OS service. Idempotent: no error if not installed."""
    _platform_module().uninstall_service(label)


def is_service_installed(label: str = SATELLITE_SERVICE_LABEL) -> bool:
    return _platform_module().is_service_installed(label)


def _platform_module():
    """Lazy import of the right platform module for ``sys.platform``."""
    if sys.platform == "darwin":
        from satellite_storage.cli import service_macos
        return service_macos
    if sys.platform.startswith("linux"):
        from satellite_storage.cli import service_linux
        return service_linux
    if sys.platform == "win32":
        from satellite_storage.cli import service_windows
        return service_windows
    raise PlatformNotSupportedError(
        f"sys.platform={sys.platform!r} — molebie-satellite currently "
        f"supports darwin / linux / win32 only."
    )
