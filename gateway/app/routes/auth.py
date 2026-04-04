"""
Authentication endpoints for the Gateway API.

Supports single-user mode (password only) and multi-user mode (email + password).
"""


from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.middleware.auth import JWTPayload, get_current_user
from app.services.auth_service import AuthService, get_auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class SimpleLoginRequest(BaseModel):
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class AuthModeResponse(BaseModel):
    mode: str
    setup_complete: bool


@router.get("/mode", response_model=AuthModeResponse)
async def get_auth_mode(
    auth: AuthService = Depends(get_auth_service),
):
    """Get the authentication mode and setup state."""
    return await auth.get_auth_mode_info()


@router.post("/register", response_model=AuthResponse)
async def register(
    request: LoginRequest,
    auth: AuthService = Depends(get_auth_service),
):
    """Register a new user (multi-user mode only)."""
    try:
        result = await auth.register(request.email, request.password)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    auth: AuthService = Depends(get_auth_service),
):
    """Login with email and password."""
    try:
        result = await auth.login(request.email, request.password)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post("/login-simple", response_model=AuthResponse)
async def login_simple(
    request: SimpleLoginRequest,
    auth: AuthService = Depends(get_auth_service),
):
    """Login with password only (single-user mode).

    On first call, the password is set. On subsequent calls, it's verified.
    """
    try:
        result = await auth.login_simple(request.password)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.get("/me")
async def get_current_user_info(
    user: JWTPayload = Depends(get_current_user),
):
    """Get info about the currently authenticated user."""
    return {
        "id": user.user_id,
        "email": user.email,
        "role": user.role,
    }
