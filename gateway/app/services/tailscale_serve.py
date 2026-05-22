"""Tailscale Serve auto-bootstrap helpers.

Tailscale Serve is Tailscale's built-in HTTPS reverse proxy: it fetches
and renews a TLS cert for the local node, terminates TLS, and forwards
the decrypted request to a local HTTP backend with cryptographically-
verified identity headers attached.

This module provides three subprocess-shaped helpers the gateway's
lifespan startup hook uses to ensure Tailscale Serve is proxying
``https://<host>.<tailnet>.ts.net/`` to the gateway's local HTTP port:

* ``is_serve_configured`` — am I already exposed?
* ``enable_serve``        — expose me.
* ``get_https_url``       — what URL does the operator use?

Trust model upgrade this enables (when satellites later switch to the
HTTPS path in slice 7c): the ``Tailscale-User-Login`` header on inbound
requests is injected by Tailscale Serve based on the requesting peer's
verified Tailscale identity, rather than self-set by the CLI as in
slice 7a. That's the actual hardening payoff.

All helpers are defensive — subprocess failures (missing binary, timeout,
non-zero exit, malformed JSON, missing fields) return safe values
(`False` or `None`). They never raise to callers; the lifespan hook
falls back to HTTP-only on any failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_TAILSCALE_FALLBACK_PATHS = [
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
]

_PROBE_TIMEOUT_SEC = 2.0
_ENABLE_TIMEOUT_SEC = 10.0


def _find_tailscale_cli() -> str | None:
    """Locate the `tailscale` binary on PATH or in the macOS .app bundle."""
    cli = shutil.which("tailscale")
    if cli:
        return cli
    for path in _TAILSCALE_FALLBACK_PATHS:
        if Path(path).exists():
            return path
    return None


def is_serve_configured(port: int = 8000) -> bool:
    """Return True if ``tailscale serve`` already proxies to ``localhost:<port>``.

    Parses ``tailscale serve status --json`` output looking for any
    handler entry whose ``Proxy`` field points at the local gateway port.
    """
    cli = _find_tailscale_cli()
    if not cli:
        return False
    try:
        result = subprocess.run(
            [cli, "serve", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    expected_proxy = f"http://127.0.0.1:{port}"
    # Tailscale's status JSON nests handlers under Web → "<host>:443" → Handlers
    # → "<path>" → Proxy. Walk defensively.
    web = data.get("Web") or {}
    if not isinstance(web, dict):
        return False
    for endpoint in web.values():
        if not isinstance(endpoint, dict):
            continue
        handlers = endpoint.get("Handlers") or {}
        if not isinstance(handlers, dict):
            continue
        for handler in handlers.values():
            if not isinstance(handler, dict):
                continue
            proxy = handler.get("Proxy")
            if isinstance(proxy, str) and proxy.rstrip("/") == expected_proxy:
                return True
    return False


def enable_serve(port: int = 8000) -> bool:
    """Run ``tailscale serve --bg --https=443 http://127.0.0.1:<port>``.

    Returns True on exit 0. On any failure (missing binary, non-zero
    exit, timeout, etc.) returns False and prints stderr to the gateway
    log so operators can diagnose. The most common failure is
    "HTTPS Certificates are not enabled for this tailnet" — fixable in
    the Tailscale admin console.
    """
    cli = _find_tailscale_cli()
    if not cli:
        print("[security] tailscale binary not found; can't enable Serve")
        return False
    proxy_target = f"http://127.0.0.1:{port}"
    try:
        result = subprocess.run(
            [cli, "serve", "--bg", "--https=443", proxy_target],
            capture_output=True,
            text=True,
            timeout=_ENABLE_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[security] tailscale serve invocation failed: {exc}")
        return False
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        print(f"[security] tailscale serve returned non-zero: {stderr}")
        return False
    return True


def get_https_url() -> str | None:
    """Compute the MagicDNS HTTPS URL Tailscale Serve exposes the gateway at.

    Parses ``tailscale status --json`` for ``Self.DNSName`` (Tailscale's
    fully-qualified MagicDNS name for this node) and returns
    ``https://<dns>/``. Returns None when Tailscale isn't authenticated,
    when the binary is missing, or when the response shape is
    unexpected — callers should treat None as "unknown URL" and surface
    a less specific message.
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
    if not isinstance(data, dict):
        return None
    self_node = data.get("Self")
    if not isinstance(self_node, dict):
        return None
    dns_name = self_node.get("DNSName")
    if not isinstance(dns_name, str) or not dns_name:
        return None
    # DNSName comes back with a trailing dot, e.g. "host.tailnet.ts.net."
    cleaned = dns_name.rstrip(".")
    return f"https://{cleaned}/"
