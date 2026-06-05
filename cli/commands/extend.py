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

from cli.services.network_info import get_tailscale_ip
from cli.ui.console import console, print_fail, print_info, print_ok

app = typer.Typer(no_args_is_help=True)

_GATEWAY_BASE = "http://localhost:8000"
_AUDIT_DEFAULT_LIMIT = 50
_AUDIT_MAX_LIMIT = 500
_STATUS_RECENT_AUDIT = 50  # how many audit rows to pull when summarising
_STATUS_SHOW_EVENTS = 5  # how many of those to display
_DETAIL_TRUNCATE = 40


def _request(
    method: str, path: str, *, params: dict | None = None, timeout: float = 5.0,
) -> dict[str, Any]:
    """Call a fleet endpoint over loopback, exit cleanly on failure.

    Shared by ``_fetch`` (GET), ``_post``, and ``_delete``. Always returns
    a parsed JSON dict; errors raise ``typer.Exit(1)`` with a friendly
    hint for the operator.

    Dispatches to ``httpx.get`` / ``httpx.post`` / ``httpx.delete`` rather
    than ``httpx.request`` so tests can monkeypatch each method independently
    (the existing extend-list/audit/status tests rely on patching ``httpx.get``).
    """
    url = f"{_GATEWAY_BASE}{path}"
    fn = {"GET": httpx.get, "POST": httpx.post, "DELETE": httpx.delete}.get(method)
    if fn is None:
        raise ValueError(f"unsupported HTTP method: {method!r}")
    try:
        resp = fn(url, params=params, timeout=timeout)
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


def _fetch(path: str, params: dict | None = None) -> dict[str, Any]:
    return _request("GET", path, params=params)


def _post(path: str, params: dict | None = None, *, timeout: float = 60.0) -> dict[str, Any]:
    """POST helper. Longer default timeout — a drain batch can take seconds."""
    return _request("POST", path, params=params, timeout=timeout)


def _delete(path: str, params: dict | None = None) -> dict[str, Any]:
    return _request("DELETE", path, params=params)


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


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
            "No satellites registered yet. Run [bold]molebie-ai extend "
            "invite[/bold] for a one-command setup to add one."
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


@app.command("remove")
def remove_satellite(
    host: str = typer.Argument(..., help="Tailscale hostname or IP of the satellite to remove."),
    force: bool = typer.Option(
        False, "--force",
        help="Satellite is gone — skip drain and accept data loss on documents that lived there.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts."),
) -> None:
    """Drain a satellite back to the primary and remove it from the fleet.

    Graceful (default): the satellite must be reachable. Bytes are pulled
    back, ``documents.storage_path`` is rewritten to ``local://...``, the
    satellite is deleted from the inventory.

    ``--force``: the satellite is assumed gone. No drain is attempted;
    inventory rows are deleted; documents that pointed at it surface as
    BlobUnreachableError on read.
    """
    # 1. Resolve host → node_id.
    inv = _fetch("/fleet/inventory")
    matches = [s for s in (inv.get("satellites") or []) if s.get("host") == host]
    if not matches:
        print_fail(f"No satellite registered with host {host!r}.")
        console.print(
            "    Run [bold]molebie-ai extend list[/bold] to see registered satellites."
        )
        raise typer.Exit(1)
    node_id = matches[0]["id"]

    # 2. Pre-flight.
    preview = _fetch("/fleet/extend/drain-preview", params={"node": node_id})

    if force:
        if preview["satellite_reachable"]:
            console.print()
            console.print(
                "[bold yellow]Warning:[/bold yellow] satellite responds — "
                "[bold]--force[/bold] will lose data that could have been drained."
            )
            if not yes and not typer.confirm("Continue anyway?", default=False):
                raise typer.Exit(0)
        body = _delete(f"/fleet/satellites/{node_id}", params={"force": "true"})
        console.print()
        print_info(
            f"Removed {host} (force): {body.get('lost_blobs', 0)} blob(s) "
            f"lost, {_fmt_bytes(body.get('lost_bytes', 0))}."
        )
        return

    # 3. Graceful path — refuse cleanly when prerequisites fail.
    if not preview["satellite_reachable"]:
        print_fail(f"Satellite {host} is unreachable.")
        console.print(
            "    Bring it back online and retry, or pass [bold]--force[/bold] "
            "to accept data loss for documents that lived on it."
        )
        raise typer.Exit(1)
    if not preview["feasible"]:
        print_fail(
            f"Primary has {_fmt_bytes(preview['primary_free_bytes'])} free, "
            f"needs at least {_fmt_bytes(preview['total_bytes'])} to drain."
        )
        console.print("    Free up space on the primary and retry.")
        raise typer.Exit(1)

    if preview["blob_count"] == 0:
        # Nothing to drain — just delete the inventory row.
        _delete(f"/fleet/satellites/{node_id}")
        console.print()
        print_info(f"Removed {host} (no blobs to drain).")
        return

    # 4. Confirm + drain loop.
    console.print()
    console.print(
        f"  Will drain [bold]{preview['blob_count']}[/bold] blob(s) "
        f"([cyan]{_fmt_bytes(preview['total_bytes'])}[/cyan]) "
        f"from [bold]{host}[/bold] to this primary."
    )
    console.print(
        f"  Primary free: [cyan]{_fmt_bytes(preview['primary_free_bytes'])}[/cyan]"
    )
    if not yes and not typer.confirm("Continue?", default=False):
        raise typer.Exit(0)

    total_drained = 0
    total_skipped = 0
    while True:
        report = _post(
            "/fleet/storage/drain", params={"node": node_id, "limit": 10}
        )
        total_drained += report["drained"]
        total_skipped += report["skipped"]
        console.print(
            f"  drained={report['drained']} skipped={report['skipped']} "
            f"remaining={report['remaining']}"
        )
        if report["remaining"] == 0:
            break
        # Guard against an infinite loop if every blob in a batch errors.
        if report["drained"] == 0 and report["skipped"] > 0:
            print_fail(
                f"Drain stalled: {report['skipped']} blob(s) in this batch errored, "
                f"{report['remaining']} still on satellite."
            )
            console.print(
                "    Check the audit log: [bold]molebie-ai extend audit[/bold]"
            )
            raise typer.Exit(1)

    # 5. Final inventory delete.
    _delete(f"/fleet/satellites/{node_id}")
    console.print()
    print_info(
        f"Removed {host}: {total_drained} blob(s) drained back to primary"
        + (f", {total_skipped} skipped." if total_skipped else ".")
    )


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
            "Fleet is empty. Run [bold]molebie-ai extend invite[/bold] for a "
            "one-command setup to add a second machine."
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


@app.command("invite")
def invite_satellite(
    role: str = typer.Option(
        "storage",
        help="Role the new satellite will offer (storage, compute, both).",
    ),
    label: Optional[str] = typer.Option(
        None,
        help="Optional human-readable label to pre-fill on the new satellite.",
    ),
    repo_ref: str = typer.Option(
        "main",
        "--repo-ref",
        help="Git ref / branch / tag to install satellite_storage from.",
    ),
) -> None:
    """Print a copyable one-liner to install + register a new satellite.

    Mirrors ``kubeadm token create --print-join-command``: run on the
    primary, paste the printed line on the new satellite. The new satellite
    needs only Tailscale (same tailnet), Python 3.13+, and pipx.
    """
    primary_ip = get_tailscale_ip()
    if primary_ip is None:
        print_fail("Couldn't determine this primary's Tailscale IP.")
        console.print(
            "    Make sure Tailscale is up on this machine: "
            "[bold]tailscale up[/bold]."
        )
        raise typer.Exit(1)

    pipx_target = (
        f"'git+https://github.com/Jimmy6929/Molebie_AI.git@{repo_ref}"
        f"#subdirectory=satellite_storage'"
    )
    install_cmd = (
        f"pipx install {pipx_target} \\\n"
        f"    && molebie-satellite install --primary {primary_ip} --role {role}"
    )
    if label:
        install_cmd += f" --label {label!r}"

    console.print()
    print_info("Copy this onto your new satellite machine:")
    console.print()
    rule = "─" * 72
    console.print(f"  {rule}")
    console.print(f"  {install_cmd}")
    console.print(f"  {rule}")
    console.print()
    print_ok(
        "Requirements on the satellite machine:\n"
        "    • Tailscale installed + signed in to the same tailnet\n"
        "    • Python 3.13+\n"
        "    • pipx (install with: [bold]python3 -m pip install --user pipx "
        "&& python3 -m pipx ensurepath[/bold])"
    )
