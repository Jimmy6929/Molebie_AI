"""
Guard against regressions in the CORS allow-origin regex.

The regex in gateway/app/main.py grants LAN + Tailscale browsers the same CORS
treatment as localhost. A stray character here silently breaks remote access
(Failed to fetch with no server-side error), so pin the behaviour with tests.
"""

import re

import pytest

from app.main import create_app


def _get_cors_regex() -> str:
    app = create_app()
    for middleware in app.user_middleware:
        options = middleware.kwargs
        if "allow_origin_regex" in options:
            return options["allow_origin_regex"]
    raise AssertionError("CORS middleware with allow_origin_regex not found")


@pytest.mark.parametrize(
    "origin",
    [
        "http://192.168.1.198:3000",
        "http://192.168.5.100:3000",
        "http://10.0.0.5:3000",
        "http://172.16.0.10:3000",
        "http://172.31.255.254:3000",
        "http://100.104.193.59:3000",  # Tailscale CGNAT
        "http://100.64.0.1:3000",
    ],
)
def test_lan_and_tailscale_origins_match(test_env, origin):
    assert re.match(_get_cors_regex(), origin), f"{origin} should be allowed"


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",  # handled via explicit allow_origins list
        "http://8.8.8.8:3000",  # public IP — must be rejected
        "http://192.168.1.198:3001",  # wrong port
        "https://192.168.1.198:3000",  # wrong scheme
        "http://192.168.1.198:3000/evil",  # trailing path, not an origin
    ],
)
def test_non_lan_origins_do_not_match(test_env, origin):
    assert not re.match(_get_cors_regex(), origin), f"{origin} should NOT match regex"
