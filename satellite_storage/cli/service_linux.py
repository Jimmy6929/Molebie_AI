"""Linux systemd implementation for the satellite service.

Installs the satellite as a per-user systemd unit at
``~/.config/systemd/user/<label>.service`` (where ``label`` is
``molebie-satellite`` on Linux — systemd doesn't use dotted reverse-DNS).
Loads it via ``systemctl --user daemon-reload && systemctl --user enable
--now <unit>``, which both starts the service and enables on-login auto-start
once the user enables lingering via ``loginctl enable-linger`` (operator
responsibility — we surface the hint in the wizard).

The unit template lives at
``satellite_storage/templates/molebie-satellite.service`` with
``__SATELLITE_BIN__`` / ``__DATA_DIR__`` / ``__LOG_DIR__`` placeholders.
"""

from __future__ import annotations

import subprocess
from importlib import resources
from pathlib import Path

from satellite_storage.cli.service import (
    ServiceConfig,
    ServiceInstallError,
    render_template,
)

# systemd unit names don't use the reverse-DNS form; this overrides the
# macOS-style default ``com.molebieai.satellite`` label whenever the Linux
# module is the active dispatcher.
LINUX_UNIT_NAME = "molebie-satellite"

_USER_UNITS_DIR = Path.home() / ".config" / "systemd" / "user"


def _unit_path(label: str) -> Path:
    name = label if label.endswith(".service") else f"{label}.service"
    return _USER_UNITS_DIR / name


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True, text=True,
    )


def _read_template() -> str:
    return (
        resources.files("satellite_storage")
        .joinpath("templates", "molebie-satellite.service")
        .read_text(encoding="utf-8")
    )


def install_service(config: ServiceConfig) -> str:
    """Write the unit + ``systemctl --user enable --now`` it.

    Returns the unit file's absolute path as a string — the file survives
    on disk so operators can ``cat`` it for debugging.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    _USER_UNITS_DIR.mkdir(parents=True, exist_ok=True)

    unit_text = render_template(
        _read_template(),
        mapping={
            "SATELLITE_BIN": str(config.satellite_bin),
            "DATA_DIR": str(config.data_dir),
            "LOG_DIR": str(config.log_dir),
        },
    )

    # systemd uses its own name (molebie-satellite.service), not the macOS-style
    # com.molebieai.satellite label.
    unit = _unit_path(LINUX_UNIT_NAME)
    unit.write_text(unit_text, encoding="utf-8")

    reload_result = _systemctl("daemon-reload")
    if reload_result.returncode != 0:
        raise ServiceInstallError(
            f"systemctl --user daemon-reload failed: "
            f"{reload_result.stderr.strip()}"
        )

    enable_result = _systemctl("enable", "--now", f"{LINUX_UNIT_NAME}.service")
    if enable_result.returncode != 0:
        raise ServiceInstallError(
            f"systemctl --user enable --now failed: "
            f"{enable_result.stderr.strip()}"
        )
    return str(unit)


def uninstall_service(label: str) -> None:
    """``systemctl --user disable --now`` + delete the unit file. Idempotent.

    ``label`` is accepted for cross-platform dispatcher uniformity but is
    ignored: systemd unit names are flat (``molebie-satellite.service``),
    not reverse-DNS, so the Linux module always uses ``LINUX_UNIT_NAME``
    regardless of what the dispatcher's macOS-flavoured default sends in.
    """
    _systemctl("disable", "--now", f"{LINUX_UNIT_NAME}.service")
    unit = _unit_path(LINUX_UNIT_NAME)
    if unit.exists():
        unit.unlink()
    _systemctl("daemon-reload")


def is_service_installed(label: str) -> bool:
    """``label`` accepted for dispatcher uniformity but ignored (see
    ``uninstall_service`` docstring)."""
    return _unit_path(LINUX_UNIT_NAME).exists()
