"""Tests for the satellite-side Tailscale identity dependency.

Identical shape to gateway's identity tests — kept in this package so
the satellite service's tests live alongside its code, and so the
gateway and satellite can drift independently if their trust models
ever diverge.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from satellite_storage.middleware.tailscale_identity import (
    TailscaleIdentity,
    get_tailscale_identity,
)


def _make_request(
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("100.64.0.5", 1234),
) -> Request:
    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "PUT",
        "path": "/v1/storage/blobs/abc",
        "headers": raw_headers,
        "client": client,
        "query_string": b"",
        "scheme": "http",
    }
    return Request(scope)


class TestTailscaleIdentity:
    def test_returns_identity_when_login_present(self):
        request = _make_request(
            headers={"Tailscale-User-Login": "operator@example.com",
                     "Tailscale-User-Name": "Operator"},
        )
        identity = asyncio.run(get_tailscale_identity(request))
        assert isinstance(identity, TailscaleIdentity)
        assert identity.user_login == "operator@example.com"
        assert identity.user_name == "Operator"
        assert identity.peer_ip == "100.64.0.5"

    def test_missing_login_raises_401(self):
        request = _make_request(headers={})
        with pytest.raises(HTTPException) as exc:
            asyncio.run(get_tailscale_identity(request))
        assert exc.value.status_code == 401

    def test_peer_ip_falls_back_when_client_missing(self):
        request = _make_request(
            headers={"Tailscale-User-Login": "x"},
            client=None,
        )
        identity = asyncio.run(get_tailscale_identity(request))
        assert identity.peer_ip == "unknown"
