"""Detect reachable network addresses (Tailscale, LAN) for cross-device access."""

from __future__ import annotations

import shutil
import socket
import subprocess
from functools import cache
from pathlib import Path


# Tailscale on macOS installs the CLI inside the .app bundle and doesn't
# always add it to PATH, so probe the known location as a fallback.
_TAILSCALE_FALLBACK_PATHS = [
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
]


def _find_tailscale_cli() -> str | None:
    cli = shutil.which("tailscale")
    if cli:
        return cli
    for path in _TAILSCALE_FALLBACK_PATHS:
        if Path(path).exists():
            return path
    return None


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
            timeout=2.0,
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


@cache
def get_lan_ip() -> str | None:
    """Return the primary outbound IPv4 address, or None if no route exists."""
    # Classic trick: UDP socket to a public IP doesn't send packets but
    # forces the kernel to pick the interface it would route through.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    if not ip or ip == "0.0.0.0" or ip.startswith("127."):
        return None
    return ip


def get_network_urls(port: int) -> list[tuple[str, str]]:
    """Return a list of (label, url) pairs for every reachable address on the given port.

    Always includes the Local entry; LAN and Tailscale entries are only included
    when detected.
    """
    urls: list[tuple[str, str]] = [("Local", f"http://localhost:{port}")]
    lan = get_lan_ip()
    if lan:
        urls.append(("LAN", f"http://{lan}:{port}"))
    tailscale = get_tailscale_ip()
    if tailscale:
        urls.append(("Tailscale", f"http://{tailscale}:{port}"))
    return urls
