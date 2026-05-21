"""Tailscale identity dependency for fleet endpoints.

Endpoints that satellites call from remote tailnet IPs (e.g.,
``POST /fleet/satellites/register``) authenticate via headers injected by
Tailscale Serve on the local Tailscale daemon. The dependency parses
those headers and returns a ``TailscaleIdentity`` for the caller.

Trust model — header presence, not cryptographic validation:

* Tailscale Serve injects ``Tailscale-User-Login``,
  ``Tailscale-User-Name``, etc. on every incoming request. Those headers
  cannot be forged by clients off the tailnet because off-tailnet
  clients can't reach the gateway at all — the Tailscale daemon is the
  network filter.
* Clients on the tailnet *could* in theory inject the header themselves
  if they bypassed Tailscale Serve and hit the gateway directly. That's
  what the future ``tailscale cert`` mTLS slice covers as defense in
  depth. For v0.2, the network-layer trust is the baseline.

Design notes:

* Shape mirrors ``gateway/app/middleware/auth.py``'s ``get_current_user``
  pattern (FastAPI dependency, not global middleware) — register/audit
  routes opt in via ``Depends(get_tailscale_identity)``.
* No coupling to the JWT auth path — Tailscale identity is the
  satellite-side analogue of the user-side JWT, distinct concept.
* Peer Node ID is not directly injected by Tailscale Serve as a header;
  we use the request's client IP (the Tailscale 100.x.x.x address) as
  the network-level identifier. The (user_login, peer_ip) tuple is the
  full identity for audit purposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class TailscaleIdentity:
    """Identity of a Tailscale-authenticated caller."""

    user_login: str         # e.g., "jimmy@github" — required
    user_name: str | None   # display name, optional
    peer_ip: str            # Tailscale 100.x.x.x IP, or "unknown" if request.client missing


async def get_tailscale_identity(request: Request) -> TailscaleIdentity:
    """FastAPI dependency: extract Tailscale-injected identity from request headers.

    Raises 401 when the required ``Tailscale-User-Login`` header is
    absent. Off-tailnet clients cannot route to the gateway in the first
    place, so a missing header generally means a misconfigured Tailscale
    Serve in front of the gateway — surface that loudly rather than
    silently treat as anonymous.
    """
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
