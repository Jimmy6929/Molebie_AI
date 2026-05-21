"""molebie-ai extend — operator-side fleet management commands.

Three read-only commands today, all hitting the primary's gateway over
loopback (the fleet routes are loopback-gated for operator access):

* ``extend list``   — pretty-print the fleet inventory
* ``extend audit``  — pretty-print recent audit events
* ``extend status`` — summary of fleet state by role / status, plus the
                       last few audit events

``extend remove`` is intentionally not in this slice — it requires a
drain protocol (move blobs off the satellite before deregistering) that
belongs with Storage Extension. Active per-satellite health probing is
also deferred until satellites expose a callable health endpoint.

This module is read-only by design: no writes, no config mutation, no
state persistence. Everything it shows comes from the gateway's `/fleet`
routes.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
import typer
from rich.table import Table

from cli.ui.console import console, print_fail, print_info

app = typer.Typer(no_args_is_help=True)

_GATEWAY_BASE = "http://localhost:8000"
_AUDIT_DEFAULT_LIMIT = 50
_AUDIT_MAX_LIMIT = 500
_STATUS_RECENT_AUDIT = 50  # how many audit rows to pull when summarising
_STATUS_SHOW_EVENTS = 5  # how many of those to display
_DETAIL_TRUNCATE = 40


def _fetch(path: str, params: dict | None = None) -> dict[str, Any]:
    """GET a fleet endpoint over loopback, exit cleanly on failure.

    Three call sites in this module justify pulling the friendly-error
    block out; below the rule-of-three threshold and inlining would
    triplicate ~10 lines per subcommand. Always returns a parsed JSON
    dict; never returns None — errors raise ``typer.Exit(1)``.
    """
    url = f"{_GATEWAY_BASE}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=5.0)
    except httpx.TimeoutException:
        print_fail(f"Gateway at {_GATEWAY_BASE} timed out.")
        console.print("    Is the gateway running on this machine?")
        raise typer.Exit(1)
    except httpx.TransportError as exc:
        print_fail(f"Gateway at {_GATEWAY_BASE} is unreachable: {exc}")
        console.print(
            "    Run [bold]molebie-ai run[/bold] on this machine to start the gateway."
        )
        raise typer.Exit(1)
    if resp.status_code >= 400:
        print_fail(f"Gateway returned HTTP {resp.status_code} on {path}.")
        console.print(f"    Body: {resp.text}")
        raise typer.Exit(1)
    return resp.json()


def _short_id(full: str) -> str:
    """Trim a UUID to its first 8 chars for human-readable display."""
    return full[:8] if full else ""


def _short_time(iso_ts: str) -> str:
    """Extract HH:MM:SS from an ISO 8601 timestamp; fall back to the original."""
    if not iso_ts:
        return ""
    # ISO is like "2026-05-21T16:30:00+00:00"; carve out the time portion.
    if "T" in iso_ts:
        time_part = iso_ts.split("T", 1)[1]
        return time_part[:8] if len(time_part) >= 8 else time_part
    return iso_ts


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


@app.command("list")
def list_satellites() -> None:
    """List every satellite registered with this primary."""
    body = _fetch("/fleet/inventory")
    satellites = body.get("satellites") or []
    if not satellites:
        print_info(
            "No satellites registered yet. Run [bold]molebie-ai join "
            "<primary>[/bold] on a satellite to add one."
        )
        return

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("ID", style="cyan", min_width=8, max_width=8)
    table.add_column("Host", min_width=14)
    table.add_column("Role", min_width=8)
    table.add_column("Status", min_width=8)
    table.add_column("Label", min_width=10)
    table.add_column("Joined", min_width=10)

    for sat in satellites:
        table.add_row(
            _short_id(sat.get("id", "")),
            sat.get("host", ""),
            sat.get("role", ""),
            sat.get("status", ""),
            sat.get("label") or "",
            _short_time(sat.get("joined_at", "")),
        )
    console.print()
    console.print(table)
    console.print()
    print_info(f"{len(satellites)} satellite(s) registered.")


@app.command("audit")
def audit_log(
    limit: int = typer.Option(
        _AUDIT_DEFAULT_LIMIT,
        help=f"Max events to show (1–{_AUDIT_MAX_LIMIT}).",
    ),
    event_type: Optional[str] = typer.Option(
        None,
        "--event-type",
        help="Only show events with this exact event_type (e.g. satellite.join).",
    ),
    since: Optional[str] = typer.Option(
        None,
        help="ISO 8601 timestamp; only show events created after this time.",
    ),
) -> None:
    """Show recent audit-log entries (newest first)."""
    if limit < 1 or limit > _AUDIT_MAX_LIMIT:
        print_fail(f"--limit must be between 1 and {_AUDIT_MAX_LIMIT}.")
        raise typer.Exit(1)

    params: dict[str, Any] = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    if since:
        params["since"] = since

    body = _fetch("/fleet/audit", params=params)
    events = body.get("events") or []
    if not events:
        print_info("No audit events recorded yet.")
        return

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Time", style="cyan", min_width=8)
    table.add_column("Event", min_width=18)
    table.add_column("Actor", min_width=14)
    table.add_column("Target", min_width=14)
    table.add_column("Detail", min_width=20)

    for ev in events:
        metadata = ev.get("metadata") or {}
        detail = ", ".join(f"{k}={v}" for k, v in metadata.items()) if metadata else ""
        table.add_row(
            _short_time(ev.get("created_at", "")),
            ev.get("event_type", ""),
            ev.get("actor") or "",
            ev.get("target") or "",
            _truncate(detail, _DETAIL_TRUNCATE),
        )
    console.print()
    console.print(table)
    console.print()
    print_info(f"{len(events)} event(s) shown.")


@app.command("status")
def fleet_status() -> None:
    """Summary of the fleet — satellites by role/status, recent audit events."""
    inv = _fetch("/fleet/inventory")
    audit = _fetch("/fleet/audit", params={"limit": _STATUS_RECENT_AUDIT})

    satellites = inv.get("satellites") or []
    events = audit.get("events") or []

    console.print()
    console.print("[heading]Molebie AI — Fleet Status[/heading]")
    console.print()

    if not satellites:
        print_info(
            "Fleet is empty. Run [bold]molebie-ai join <primary>[/bold] on a "
            "second machine to begin."
        )
        return

    # Aggregates
    by_role: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for sat in satellites:
        role = sat.get("role", "?")
        status = sat.get("status", "?")
        by_role[role] = by_role.get(role, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    role_str = " · ".join(f"{n} {r}" for r, n in sorted(by_role.items()))
    status_str = " · ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    console.print(f"  Satellites: [bold]{len(satellites)}[/bold]  ({role_str})")
    console.print(f"  Status:     {status_str}")

    if events:
        last_ev = events[0]
        console.print(
            f"  Last fleet activity: [cyan]{last_ev.get('created_at', '?')}[/cyan] "
            f"— {last_ev.get('event_type', '?')}"
        )
    console.print()

    # Recent events table — show the most recent N
    if events:
        console.print("[heading]Recent events[/heading]")
        table = Table(show_header=True, header_style="bold", expand=False)
        table.add_column("Time", style="cyan", min_width=8)
        table.add_column("Event", min_width=18)
        table.add_column("Target", min_width=14)
        for ev in events[:_STATUS_SHOW_EVENTS]:
            table.add_row(
                _short_time(ev.get("created_at", "")),
                ev.get("event_type", ""),
                ev.get("target") or "",
            )
        console.print(table)
        console.print()
