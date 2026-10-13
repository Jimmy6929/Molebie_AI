"""Token issue/verify round-trip and the rejections that keep it safe.

Pure functions only — no DB, no TestClient. AuthService never touches its
db for token work, so it is built with ``db=None``.
"""

from __future__ import annotations

import base64
import json
import time

import jwt
import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.middleware.auth import verify_jwt
from app.services.auth_service import AuthService

USER_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def auth() -> AuthService:
    return AuthService(get_settings(), db=None)


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def test_issued_token_verifies_on_both_paths(auth):
    token = auth.issue_token(USER_ID, "owner@example.com")
    assert auth.verify_token(token)["sub"] == USER_ID
    payload = verify_jwt(token, get_settings())
    assert payload.user_id == USER_ID
    assert payload.email == "owner@example.com"


def test_expired_token_is_rejected(auth):
    settings = get_settings()
    expired = jwt.encode(
        {"sub": USER_ID, "exp": int(time.time()) - 60},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(ValueError):
        auth.verify_token(expired)
    with pytest.raises(HTTPException) as exc:
        verify_jwt(expired, settings)
    assert exc.value.status_code == 401


def test_token_signed_with_another_secret_is_rejected(auth):
    settings = get_settings()
    forged = jwt.encode(
        {"sub": USER_ID, "exp": int(time.time()) + 60},
        "not-the-gateway-secret-not-the-gateway-secret",
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(ValueError):
        auth.verify_token(forged)
    with pytest.raises(HTTPException):
        verify_jwt(forged, settings)


def test_unsigned_alg_none_token_is_rejected(auth):
    """An attacker-built token claiming ``alg: none`` must not verify."""
    unsigned = (
        f"{_b64({'alg': 'none', 'typ': 'JWT'})}."
        f"{_b64({'sub': USER_ID, 'exp': int(time.time()) + 60})}."
    )
    with pytest.raises(ValueError):
        auth.verify_token(unsigned)
    with pytest.raises(HTTPException):
        verify_jwt(unsigned, get_settings())


def test_token_in_an_algorithm_we_did_not_configure_is_rejected(auth):
    """Correct secret, wrong algorithm: only ``jwt_algorithm`` is accepted."""
    settings = get_settings()
    other = jwt.encode(
        {"sub": USER_ID, "exp": int(time.time()) + 60},
        settings.jwt_secret,
        algorithm="HS512",
    )
    with pytest.raises(ValueError):
        auth.verify_token(other)
    with pytest.raises(HTTPException):
        verify_jwt(other, settings)


def test_garbage_token_is_a_401_not_a_500():
    with pytest.raises(HTTPException) as exc:
        verify_jwt("not.a.token", get_settings())
    assert exc.value.status_code == 401
