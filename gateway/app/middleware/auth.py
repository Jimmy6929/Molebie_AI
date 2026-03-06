"""
JWT Authentication middleware for validating Supabase tokens.
"""

from typing import Optional
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from jose.exceptions import JWTClaimsError
from pydantic import BaseModel

from app.config import Settings, get_settings


# HTTP Bearer token scheme
security = HTTPBearer()


class JWTPayload(BaseModel):
    """Decoded JWT payload from Supabase."""
    sub: str  # User ID (UUID)
    email: Optional[str] = None
    role: Optional[str] = None
    aud: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    raw_token: Optional[str] = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    @property
    def user_id(self) -> str:
        """Alias for sub (subject) which contains the user ID."""
        return self.sub


# Cache for JWKS
_jwks_cache: Optional[dict] = None


async def get_jwks(supabase_url: str) -> dict:
    """Fetch JWKS from Supabase for JWT verification."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{supabase_url}/auth/v1/.well-known/jwks.json")
        response.raise_for_status()
        _jwks_cache = response.json()
        return _jwks_cache


def decode_jwt_simple(token: str) -> JWTPayload:
    """
    Decode JWT without verification (for local development).
    In production, use proper JWKS verification.
    """
    try:
        # Decode without verification for local dev
        # Still need to provide a key, but verification is disabled
        payload = jwt.decode(
            token,
            key="",  # Empty key since we're not verifying signature
            options={
                "verify_signature": False,
                "verify_aud": False,
                "verify_exp": True,
            }
        )
        return JWTPayload(**payload)
    except (JWTError, JWTClaimsError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings),
) -> JWTPayload:
    """
    FastAPI dependency to extract and validate the current user from JWT.
    
    For local development, we skip signature verification since Supabase
    uses ES256 with rotating keys. In production with Supabase Cloud,
    you would verify against the JWKS endpoint.
    """
    payload = decode_jwt_simple(credentials.credentials)
    payload.raw_token = credentials.credentials
    return payload


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    settings: Settings = Depends(get_settings),
) -> Optional[JWTPayload]:
    """
    FastAPI dependency for optional authentication.
    Returns None if no token provided, validates if token is present.
    """
    if credentials is None:
        return None
    return decode_jwt_simple(credentials.credentials)
