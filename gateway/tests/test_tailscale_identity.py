"""Tests for the Tailscale identity dependency.

Exercises the dependency directly with Starlette ``Request`` objects
constructed from minimal ASGI scopes — simpler than mounting a FastAPI
app for what is effectively a pure parser.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.middleware.tailscale_identity import (
    TailscaleIdentity,
    get_tailscale_identity,
)


def _make_request(
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("100.64.0.5", 1234),
) -> Request:
    """Build a minimal Starlette Request for dependency exercise."""
    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/anywhere",
        "headers": raw_headers,
        "client": client,
        "query_string": b"",
        "scheme": "http",
    }
    return Request(scope)


class TestTailscaleIdentityDependency:
    def test_returns_identity_when_login_present(self):
        request = _make_request(
            headers={"Tailscale-User-Login": "jimmy@github", "Tailscale-User-Name": "Jimmy Z"}
        )
        identity = asyncio.run(get_tailscale_identity(request))
        assert isinstance(identity, TailscaleIdentity)
        assert identity.user_login == "jimmy@github"
        assert identity.user_name == "Jimmy Z"
        assert identity.peer_ip == "100.64.0.5"

    def test_missing_login_raises_401(self):
        request = _make_request(headers={})
        with pytest.raises(HTTPException) as exc:
            asyncio.run(get_tailscale_identity(request))
        assert exc.value.status_code == 401
        assert "Tailscale-User-Login" in exc.value.detail

    def test_optional_user_name_absent(self):
        request = _make_request(headers={"Tailscale-User-Login": "alice@example.com"})
        identity = asyncio.run(get_tailscale_identity(request))
        assert identity.user_login == "alice@example.com"
        assert identity.user_name is None

    def test_peer_ip_captured_from_request_client(self):
        request = _make_request(
            headers={"Tailscale-User-Login": "x"},
            client=("100.64.0.99", 8080),
        )
        identity = asyncio.run(get_tailscale_identity(request))
        assert identity.peer_ip == "100.64.0.99"

    def test_peer_ip_falls_back_when_client_missing(self):
        request = _make_request(
            headers={"Tailscale-User-Login": "x"},
            client=None,
        )
        identity = asyncio.run(get_tailscale_identity(request))
        assert identity.peer_ip == "unknown"
