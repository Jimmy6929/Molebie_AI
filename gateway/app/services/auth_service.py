"""
Authentication service using JWT + bcrypt.

Replaces Supabase Auth. Supports single-user (password only)
and multi-user (email + password) modes.
"""

import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import bcrypt
from jose import jwt, JWTError

from app.config import Settings, get_settings
from app.services.database import DatabaseService, get_database_service
from app.schema import DEFAULT_USER_ID, DEFAULT_USER_EMAIL


TOKEN_EXPIRY_HOURS = 168  # 7 days


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


class AuthService:
    """Gateway-managed authentication with JWT + bcrypt."""

    def __init__(self, settings: Settings, db: DatabaseService):
        self.settings = settings
        self.db = db
        self.jwt_secret = settings.jwt_secret
        self.jwt_algorithm = settings.jwt_algorithm
        self.auth_mode = getattr(settings, "auth_mode", "single")

    def issue_token(self, user_id: str, email: str) -> str:
        """Create a signed JWT token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "email": email,
            "role": "authenticated",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=TOKEN_EXPIRY_HOURS)).timestamp()),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token. Returns the payload dict."""
        try:
            return jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_aud": False},
            )
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")

    async def register(self, email: str, password: str) -> Dict[str, Any]:
        """Register a new user (multi-user mode). Returns {token, user}."""
        if self.auth_mode != "multi":
            raise ValueError("Registration is disabled in single-user mode")

        db = self.db
        # Check if user exists
        conn = await db._get_conn()
        rows = await conn.execute_fetchall(
            "SELECT id FROM users WHERE email = ?", (email,)
        )
        if rows:
            raise ValueError("Email already registered")

        # Create user
        from app.services.database import _uuid, _now
        user_id = _uuid()
        now = _now()
        password_hash = _hash_password(password)
        await conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, email, password_hash, now, now),
        )
        await conn.commit()

        token = self.issue_token(user_id, email)
        return {
            "token": token,
            "user": {"id": user_id, "email": email},
        }

    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login with email + password (multi-user mode). Returns {token, user}."""
        conn = await self.db._get_conn()
        rows = await conn.execute_fetchall(
            "SELECT id, email, password_hash FROM users WHERE email = ?", (email,)
        )
        if not rows:
            raise ValueError("Invalid email or password")

        user = dict(rows[0])
        if not user.get("password_hash") or not _verify_password(password, user["password_hash"]):
            raise ValueError("Invalid email or password")

        token = self.issue_token(user["id"], user["email"])
        return {
            "token": token,
            "user": {"id": user["id"], "email": user["email"]},
        }

    async def login_simple(self, password: str) -> Dict[str, Any]:
        """Login with password only (single-user mode). Returns {token, user}."""
        conn = await self.db._get_conn()
        rows = await conn.execute_fetchall(
            "SELECT id, email, password_hash FROM users WHERE id = ?",
            (DEFAULT_USER_ID,),
        )
        if not rows:
            raise ValueError("No user configured. Run setup first.")

        user = dict(rows[0])

        # If no password set yet, set it now (first login acts as setup)
        if not user.get("password_hash"):
            hashed = _hash_password(password)
            await conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (hashed, datetime.now(timezone.utc).isoformat(), DEFAULT_USER_ID),
            )
            await conn.commit()
        elif not _verify_password(password, user["password_hash"]):
            raise ValueError("Invalid password")

        token = self.issue_token(user["id"], user["email"])
        return {
            "token": token,
            "user": {"id": user["id"], "email": user["email"]},
        }

    async def get_auth_mode_info(self) -> Dict[str, Any]:
        """Return auth mode and setup state."""
        conn = await self.db._get_conn()
        rows = await conn.execute_fetchall(
            "SELECT password_hash FROM users WHERE id = ?", (DEFAULT_USER_ID,)
        )
        has_password = bool(rows and rows[0]["password_hash"])
        return {
            "mode": self.auth_mode,
            "setup_complete": has_password if self.auth_mode == "single" else True,
        }


_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get cached AuthService instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(get_settings(), get_database_service())
    return _auth_service
