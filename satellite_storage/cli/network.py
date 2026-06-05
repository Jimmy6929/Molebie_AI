"""Tailscale identity + IP discovery for the satellite CLI.

Cross-platform fallbacks for the ``tailscale`` binary location:

* **macOS** — the Mac App Store install puts the CLI inside the .app
  bundle and doesn't add it to PATH.
* **Linux** — usually on PATH via the package manager; ``shutil.which``
  handles ``/usr/bin/tailscale`` and ``/usr/local/bin/tailscale``.
* **Windows** — installs to ``C:\\Program Files\\Tailscale\\tailscale.exe``;
  not always on PATH for non-admin shells.

Deliberate copy of the primary's ``cli/services/network_info.py`` rather
than a cross-package import — the satellite ships as an independent
``pipx``-installable package and can't depend on the primary's tree.

The ``tailscale`` CLI's ``whoami`` subcommand does NOT exist; identity is
read from ``status --json`` via ``Self.UserID`` → ``User[<id>].LoginName``
(real-hardware smoke test of v0.2 caught this regression; tests in both
packages had mocked the subprocess).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_TAILSCALE_FALLBACK_PATHS_MACOS = [
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
]
_TAILSCALE_FALLBACK_PATHS_WINDOWS = [
    r"C:\Program Files\Tailscale\tailscale.exe",
    r"C:\Program Files (x86)\Tailscale\tailscale.exe",
]
# Linux installs land in PATH via apt/dnf/etc., so shutil.which covers them.
# Keep an empty list so the lookup loop is platform-uniform.
_TAILSCALE_FALLBACK_PATHS_LINUX: list[str] = []

_PROBE_TIMEOUT_SEC = 2.0


def _platform_fallback_paths() -> list[str]:
    if sys.platform == "darwin":
        return _TAILSCALE_FALLBACK_PATHS_MACOS
    if sys.platform == "win32":
        return _TAILSCALE_FALLBACK_PATHS_WINDOWS
    return _TAILSCALE_FALLBACK_PATHS_LINUX


def _find_tailscale_cli() -> str | None:
    """Locate the ``tailscale`` binary on this machine."""
    cli = shutil.which("tailscale")
    if cli:
        return cli
    for path in _platform_fallback_paths():
        if Path(path).exists():
            return path
    return None


@dataclass(frozen=True)
class TailscaleWhoamiInfo:
    """Identity of the locally-authenticated Tailscale user."""

    user_login: str
    display_name: str | None


@cache
def get_tailscale_whoami() -> TailscaleWhoamiInfo | None:
    """Return the local Tailscale identity, or None if unavailable.

    Never raises. Returns None when Tailscale isn't installed, the
    daemon isn't ``Running``, the subprocess times out, or the output
    can't be parsed — callers decide how to surface the failure.
    """
    cli = _find_tailscale_cli()
    if not cli:
        return None
    try:
        result = subprocess.run(
            [cli, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return _whoami_from_status(data)


def _whoami_from_status(data: object) -> TailscaleWhoamiInfo | None:
    """Pull ``User[Self.UserID]`` out of ``tailscale status --json``."""
    if not isinstance(data, dict):
        return None
    if data.get("BackendState") != "Running":
        return None
    self_info = data.get("Self")
    if not isinstance(self_info, dict):
        return None
    user_id = self_info.get("UserID")
    if not isinstance(user_id, int):
        return None
    users = data.get("User")
    if not isinstance(users, dict):
        return None
    profile = users.get(str(user_id))
    if not isinstance(profile, dict):
        return None
    login = profile.get("LoginName")
    if not isinstance(login, str) or not login:
        return None
    display = profile.get("DisplayName")
    return TailscaleWhoamiInfo(
        user_login=login,
        display_name=display if isinstance(display, str) and display else None,
    )


@cache
def get_tailscale_ip() -> str | None:
    """Return this machine's Tailscale IPv4 address, or None if unavailable."""
    cli = _find_tailscale_cli()
    if not cli:
        return None
    try:
        result = subprocess.run(
            [cli, "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None
