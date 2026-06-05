"""``molebie-satellite join`` — one-shot register with a Molebie primary.

Stateless: hits the primary's ``POST /fleet/satellites/register`` once and
prints the result. No local config file written — the primary's
``fleet_satellites`` table is the authoritative record of fleet membership
(satellites are passive blob stores; they don't need to remember which
primary they belong to). Re-running ``join`` against the same primary is
idempotent — the primary upserts on host match.

Trust model is the same as the primary's CLI ``join``: read the local
Tailscale identity via ``status --json``, send it as the
``Tailscale-User-Login`` header. Off-tailnet clients can't reach the
primary at all, so the network layer is the auth gate.
"""

from __future__ import annotations

import httpx
import typer

from satellite_storage.cli.network import (
    get_tailscale_ip,
    get_tailscale_whoami,
)
from satellite_storage.cli.ui import (
    console,
    print_fail,
    print_info,
    print_ok,
)

_VALID_ROLES = ("storage", "compute", "both")


def join_command(
    primary: str = typer.Option(
        ...,
        "--primary",
        help="Tailscale hostname or IP of the primary, e.g. 100.64.0.5",
    ),
    port: int = typer.Option(8000, help="Gateway port on the primary."),
    role: str = typer.Option(
        "storage",
        help=f"Role this satellite offers (one of: {', '.join(_VALID_ROLES)}).",
    ),
    label: str | None = typer.Option(
        None,
        help="Optional human-readable label for this satellite.",
    ),
) -> None:
    """Register this machine with a Molebie primary."""
    console.print()
    console.print("[bold]Molebie AI — Join Fleet[/bold]")
    console.print()

    if role not in _VALID_ROLES:
        print_fail(f"Invalid --role {role!r}; expected one of {_VALID_ROLES}.")
        raise typer.Exit(1)

    whoami = get_tailscale_whoami()
    if whoami is None:
        print_fail("Couldn't read your Tailscale identity.")
        console.print(
            "    Make sure Tailscale is installed and authenticated. "
            "Run [bold]tailscale up[/bold] and try again."
        )
        raise typer.Exit(1)
    print_ok(f"Tailscale identity: {whoami.user_login}")

    base_url = f"http://{primary}:{port}"
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
        if resp.status_code >= 400:
            print_fail(
                f"Primary at {primary}:{port} returned HTTP {resp.status_code} "
                f"on /health — is the gateway running?"
            )
            raise typer.Exit(1)
    except httpx.TimeoutException:
        print_fail(f"Primary at {primary}:{port} timed out on /health.")
        console.print("    Check Tailscale connectivity between the two machines.")
        raise typer.Exit(1)
    except httpx.TransportError as exc:
        print_fail(f"Primary at {primary}:{port} is unreachable: {exc}")
        console.print(
            "    Make sure the gateway is running on the primary and that "
            "this machine can reach it over Tailscale."
        )
        raise typer.Exit(1)
    print_ok(f"Primary reachable at {base_url}")

    local_ts_ip = get_tailscale_ip()
    if not local_ts_ip:
        print_fail("Couldn't determine this machine's Tailscale IP.")
        console.print(
            "    Run [bold]tailscale ip -4[/bold] to check; make sure "
            "Tailscale is up and authenticated."
        )
        raise typer.Exit(1)

    payload: dict = {"host": local_ts_ip, "role": role}
    cleaned_label = (label or "").strip() or None
    if cleaned_label:
        payload["label"] = cleaned_label

    headers = {"Tailscale-User-Login": whoami.user_login}
    try:
        resp = httpx.post(
            f"{base_url}/fleet/satellites/register",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        print_fail(f"Failed to reach the register endpoint: {exc}")
        raise typer.Exit(1)

    if resp.status_code == 401:
        print_fail("Primary rejected the Tailscale identity header (HTTP 401).")
        console.print(
            "    The primary's middleware didn't see a "
            "[bold]Tailscale-User-Login[/bold] header. Check that the "
            "primary gateway is on the latest version."
        )
        raise typer.Exit(1)
    if resp.status_code == 422:
        print_fail("Primary rejected the registration payload (HTTP 422).")
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        console.print(f"    Validation detail: {detail}")
        raise typer.Exit(1)
    if resp.status_code >= 400:
        print_fail(f"Primary returned HTTP {resp.status_code} on register.")
        console.print(f"    Body: {resp.text}")
        raise typer.Exit(1)

    body = resp.json()
    print_ok(
        f"Registered as [bold]{body['id'][:8]}…[/bold] "
        f"(host={body.get('host')}, role={body.get('role')})"
    )
    print_info("The primary's `/fleet/inventory` will now show this satellite.")
