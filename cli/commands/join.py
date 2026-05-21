"""molebie-ai join — register this machine as a satellite of a Molebie primary.

Runs on a satellite machine. Calls the primary's
``POST /fleet/satellites/register`` endpoint over Tailscale, then persists
the returned satellite info into the local ``MolebieConfig.satellites``
list. After this command succeeds, the primary's `/fleet/inventory` knows
about this satellite and the operator can manage the fleet from there.

Trust model (v0.2):

The CLI reads the local Tailscale-authenticated user via
``tailscale whoami --json`` and sends the ``LoginName`` as the
``Tailscale-User-Login`` header on the registration POST. The primary's
middleware (shipped in PR #47) reads it. Off-tailnet clients cannot reach
the primary at all, so the network layer is the auth gate. Cryptographic
verification via ``tailscale cert`` mTLS is a separate later slice that
will layer on top of this.
"""

from __future__ import annotations

from typing import Optional

import httpx
import typer

from cli.models.config import (
    SatelliteNode,
    SatelliteRole,
    SatelliteStatus,
)
from cli.services import config_manager
from cli.services.network_info import (
    get_tailscale_ip,
    get_tailscale_whoami,
)
from cli.ui.console import console, print_fail, print_info, print_ok
from cli.ui.prompts import ask_choice, ask_text

_VALID_ROLES = ("storage", "compute", "both")


def join(
    primary_host: str = typer.Argument(
        ...,
        help="Tailscale hostname or IP of the primary, e.g. 100.64.0.5 or my-primary",
    ),
    port: int = typer.Option(
        8000,
        help="Gateway port on the primary",
    ),
    role: Optional[str] = typer.Option(
        None,
        help="Pre-select role (storage/compute/both); if omitted, you'll be prompted",
    ),
    label: Optional[str] = typer.Option(
        None,
        help="Pre-set label; if omitted, you'll be prompted",
    ),
) -> None:
    """Join an existing Molebie primary as a satellite."""
    console.print()
    console.print("[heading]Molebie AI — Join Fleet[/heading]")
    console.print()

    # ── 1. Resolve local Tailscale identity ─────────────────────────────
    whoami = get_tailscale_whoami()
    if whoami is None:
        print_fail("Couldn't read your Tailscale identity.")
        console.print(
            "    Make sure Tailscale is installed and authenticated. "
            "Run [bold]tailscale up[/bold] and try again."
        )
        raise typer.Exit(1)
    print_ok(f"Tailscale identity: {whoami.user_login}")

    # ── 2. Reachability check ───────────────────────────────────────────
    base_url = f"http://{primary_host}:{port}"
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
        if resp.status_code >= 400:
            print_fail(
                f"Primary at {primary_host}:{port} returned HTTP {resp.status_code} "
                f"on /health — is the gateway running?"
            )
            raise typer.Exit(1)
    except httpx.TimeoutException:
        print_fail(f"Primary at {primary_host}:{port} timed out on /health.")
        console.print("    Check Tailscale connectivity between the two machines.")
        raise typer.Exit(1)
    except httpx.TransportError as exc:
        print_fail(f"Primary at {primary_host}:{port} is unreachable: {exc}")
        console.print(
            "    Make sure the gateway is running on the primary and that this "
            "machine can reach it over Tailscale."
        )
        raise typer.Exit(1)
    print_ok(f"Primary reachable at {base_url}")

    # ── 3. Resolve our own Tailscale IP — that's what the primary records ──
    local_ts_ip = get_tailscale_ip()
    if not local_ts_ip:
        print_fail("Couldn't determine this machine's Tailscale IP.")
        console.print(
            "    Run [bold]tailscale ip -4[/bold] to check; make sure Tailscale "
            "is up and authenticated on this machine."
        )
        raise typer.Exit(1)

    # ── 4. Prompt for role if not provided ──────────────────────────────
    if role is None:
        chosen_role = ask_choice(
            "What does this satellite offer?",
            list(_VALID_ROLES),
            default="storage",
        )
    else:
        if role not in _VALID_ROLES:
            print_fail(f"Invalid --role {role!r}; expected one of {_VALID_ROLES}.")
            raise typer.Exit(1)
        chosen_role = role

    # ── 5. Prompt for label if not provided ─────────────────────────────
    if label is None:
        raw_label = ask_text("Optional label (or leave empty)", default="")
        chosen_label: Optional[str] = raw_label.strip() or None
    else:
        chosen_label = label.strip() or None

    # ── 6. POST to the register endpoint ────────────────────────────────
    payload = {
        "host": local_ts_ip,
        "role": chosen_role,
    }
    if chosen_label:
        payload["label"] = chosen_label

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

    # ── 7. Handle response ──────────────────────────────────────────────
    if resp.status_code == 401:
        print_fail("Primary rejected the Tailscale identity header (HTTP 401).")
        console.print(
            "    This means the primary's middleware didn't see a "
            "[bold]Tailscale-User-Login[/bold] header. Check that the gateway "
            "is on the latest version (PR #47 or later)."
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

    # ── 8. Upsert into local config ─────────────────────────────────────
    new_node = SatelliteNode(
        host=body["host"],
        role=SatelliteRole(body["role"]),
        capabilities=body.get("capabilities") or {},
        status=SatelliteStatus(body["status"]),
        joined_at=body["joined_at"],
        label=body.get("label"),
    )

    config = config_manager.load_config()
    existing_index = next(
        (i for i, s in enumerate(config.satellites) if s.host == new_node.host),
        None,
    )
    if existing_index is None:
        config.satellites.append(new_node)
    else:
        config.satellites[existing_index] = new_node
    config_manager.save_config(config)

    # ── 9. Print success ────────────────────────────────────────────────
    print_ok(
        f"Registered with primary as satellite [bold]{body['id']}[/bold] "
        f"(host={new_node.host}, role={new_node.role.value})"
    )
    print_info("Saved to local config.")
