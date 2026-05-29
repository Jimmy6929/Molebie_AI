"""Outbound Tailscale identity for primary → satellite calls.

The inbound sibling — ``gateway/app/middleware/tailscale_identity.py`` —
parses the ``Tailscale-User-Login`` header that satellites *send* the
primary. This module is the reverse: when the primary calls a satellite's
``molebie-storage`` service (to read or delete a blob), it must *send*
its own operator identity so the satellite's identity gate accepts the
request.

For v0.2 the primary reads its own Tailscale-authenticated user via
``tailscale whoami --json`` — same mechanism the CLI's ``molebie-ai
join`` command uses (cli/services/network_info.py). The value is cached
for the process lifetime; identity doesn't change while the gateway runs,
and re-forking a subprocess on every blob read would be wasteful.

Deliberate copy of the CLI's ``get_tailscale_whoami`` shape rather than a
cross-package import — the gateway and CLI are independently deployable
and shouldn't depend on each other (same call we made for the inbound
middleware and ``_find_tailscale_cli``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import cache
from pathlib import Path

_TAILSCALE_FALLBACK_PATHS = [
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
]
_PROBE_TIMEOUT_SEC = 2.0


def _find_tailscale_cli() -> str | None:
    cli = shutil.which("tailscale")
    if cli:
        return cli
    for path in _TAILSCALE_FALLBACK_PATHS:
        if Path(path).exists():
            return path
    return None


@cache
def get_operator_identity() -> str | None:
    """Return the primary operator's Tailscale login (e.g. ``jimmy@github``),
    or None if Tailscale isn't installed/authenticated or the output can't
    be parsed. Never raises. Cached for the process lifetime."""
    cli = _find_tailscale_cli()
    if not cli:
        return None
    try:
        result = subprocess.run(
            [cli, "whoami", "--json"],
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
    if not isinstance(data, dict):
        return None
    profile = data.get("UserProfile")
    if not isinstance(profile, dict):
        return None
    login = profile.get("LoginName")
    if not isinstance(login, str) or not login:
        return None
    return login
