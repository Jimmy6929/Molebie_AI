"""Tailscale identity dependency for satellite blob endpoints.

A deliberate copy of ``gateway/app/middleware/tailscale_identity.py``.
We don't share via a top-level ``shared/`` package because the satellite
runs as a separate process with its own deps and lifecycle — pulling the
gateway in as an import dependency would couple two independent
deployables. Same call we made for ``_find_tailscale_cli`` (copied
between ``cli/`` and ``gateway/``).

Trust model: identical to the gateway. The header is injected by
Tailscale Serve (on the primary's side), and we trust it because
off-tailnet clients cannot route to this satellite at all — the
Tailscale daemon is the network filter. Cryptographic verification via
``tailscale cert`` mTLS is a deferred sub-slice (7b).
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class TailscaleIdentity:
    """Identity of a Tailscale-authenticated caller."""

    user_login: str
    user_name: str | None
    peer_ip: str


async def get_tailscale_identity(request: Request) -> TailscaleIdentity:
    """FastAPI dependency: extract Tailscale-injected identity from request headers."""
    login = request.headers.get("Tailscale-User-Login")
    if not login:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Tailscale identity header (Tailscale-User-Login)",
        )
    return TailscaleIdentity(
        user_login=login,
        user_name=request.headers.get("Tailscale-User-Name"),
        peer_ip=request.client.host if request.client else "unknown",
    )
