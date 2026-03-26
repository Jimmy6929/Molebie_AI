"""
JWT Authentication middleware.

Validates gateway-issued JWT tokens signed with JWT_SECRET.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from jose.exceptions import JWTClaimsError
from pydantic import BaseModel

from app.config import Settings, get_settings


# HTTP Bearer token scheme
security = HTTPBearer()


class JWTPayload(BaseModel):
    """Decoded JWT payload."""
    sub: str  # User ID
    email: Optional[str] = None
    role: Optional[str] = None
    aud: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def user_id(self) -> str:
        """Alias for sub (subject) which contains the user ID."""
        return self.sub


def verify_jwt(token: str, settings: Settings) -> JWTPayload:
    """Decode and verify a JWT token using the gateway's secret."""
    try:
        payload = jwt.decode(
            token,
            key=settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={
                "verify_aud": False,
                "verify_exp": True,
            },
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
    """FastAPI dependency to extract and validate the current user from JWT."""
    return verify_jwt(credentials.credentials, settings)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    settings: Settings = Depends(get_settings),
) -> Optional[JWTPayload]:
    """FastAPI dependency for optional authentication."""
    if credentials is None:
        return None
    return verify_jwt(credentials.credentials, settings)
