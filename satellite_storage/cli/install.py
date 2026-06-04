"""``molebie-satellite install`` — interactive wizard.

Six phases (modeled on the primary's ``cli/commands/install.py`` UX):

  1. Prerequisites: Python version, Tailscale identity, primary reachable
  2. Data directory: pick a writable path for blobs
  3. Service install: launchd / systemd / Task Scheduler via the dispatcher
  4. Start service: wait for ``/health`` to answer locally
  5. Register: POST to the primary's ``/fleet/satellites/register``
  6. Verify: confirm the satellite shows up in ``/fleet/inventory``

``--foreground`` skips the service install (steps 3+4) and tells the
operator to run ``molebie-satellite serve`` themselves; useful for dev.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import typer

from satellite_storage.cli.network import (
    get_tailscale_ip,
    get_tailscale_whoami,
)
from satellite_storage.cli.service import (
    ServiceConfig,
    ServiceInstallError,
    default_data_dir,
    default_log_dir,
    find_satellite_binary,
    install_service,
)
from satellite_storage.cli.ui import (
    ask_text,
    console,
    print_fail,
    print_info,
    print_ok,
    print_step_header,
    print_warn,
)

_VALID_ROLES = ("storage", "compute", "both")
_HEALTH_POLL_TIMEOUT_SEC = 30.0
_HEALTH_POLL_INTERVAL_SEC = 1.0


def install_command(
    primary: str = typer.Option(
        ...,
        "--primary",
        help="Tailscale hostname or IP of the primary, e.g. 100.64.0.5",
    ),
    port: int = typer.Option(8000, help="Gateway port on the primary."),
    role: str = typer.Option(
        "storage", help=f"Role this satellite offers ({', '.join(_VALID_ROLES)})."
    ),
    label: str | None = typer.Option(
        None, help="Human-readable label for this satellite."
    ),
    data_dir: Path | None = typer.Option(
        None, help="Override the default data directory."
    ),
    foreground: bool = typer.Option(
        False, "--foreground", help="Skip OS service install — run manually."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompts."
    ),
) -> None:
    """Install satellite_storage as a service and register with a primary."""
    if role not in _VALID_ROLES:
        print_fail(f"Invalid --role {role!r}; expected one of {_VALID_ROLES}.")
        raise typer.Exit(1)

    console.print()
    console.print("[bold]Molebie AI — Satellite Install Wizard[/bold]")
    total_steps = 4 if foreground else 6

    # ── 1/N — Prerequisites ─────────────────────────────────────────────
    print_step_header(1, total_steps, "Checking prerequisites…")
    whoami, base_url, local_ts_ip = _check_prerequisites(primary, port)

    # ── 2/N — Data directory ────────────────────────────────────────────
    print_step_header(2, total_steps, "Data directory")
    chosen_data_dir = _choose_data_dir(data_dir, yes)

    if foreground:
        # ── 3/4 — How to run ─────────────────────────────────────────────
        print_step_header(3, total_steps, "Register with primary")
        _register_with_primary(
            base_url=base_url, whoami_login=whoami.user_login,
            local_ts_ip=local_ts_ip, role=role, label=label,
        )
        print_step_header(4, total_steps, "Done")
        print_warn(
            "Service install skipped (--foreground). Run "
            "[bold]molebie-satellite serve[/bold] in a long-lived "
            f"terminal with [bold]MOLEBIE_STORAGE_DATA_DIR={chosen_data_dir}[/bold]."
        )
        return

    # ── 3/6 — Service install ───────────────────────────────────────────
    print_step_header(3, total_steps, "Service install")
    satellite_bin = find_satellite_binary()
    if satellite_bin is None:
        print_fail("Couldn't find the `molebie-satellite` binary on PATH.")
        console.print(
            "    Ensure pipx-installed scripts are on PATH: "
            "[bold]python3 -m pipx ensurepath[/bold] (then reopen the terminal)."
        )
        raise typer.Exit(1)

    log_dir = default_log_dir()
    config = ServiceConfig(
        satellite_bin=satellite_bin,
        data_dir=chosen_data_dir,
        log_dir=log_dir,
        home_dir=Path.home(),
    )
    try:
        unit_path = install_service(config)
    except ServiceInstallError as exc:
        print_fail(f"Service install failed: {exc}")
        raise typer.Exit(1)
    print_ok(f"Installed service unit at {unit_path}")

    # ── 4/6 — Start + health ────────────────────────────────────────────
    print_step_header(4, total_steps, "Start service")
    bind_port = _satellite_bind_port()
    health_url = f"http://localhost:{bind_port}/v1/storage/health"
    if not _wait_for_health(health_url, timeout_sec=_HEALTH_POLL_TIMEOUT_SEC):
        print_fail(
            f"Satellite service didn't respond on {health_url} "
            f"within {_HEALTH_POLL_TIMEOUT_SEC:.0f}s."
        )
        console.print(
            f"    Check the service logs at {log_dir / 'satellite-stderr.log'}"
        )
        raise typer.Exit(1)
    print_ok(f"Satellite service responding on :{bind_port}.")

    # ── 5/6 — Register ──────────────────────────────────────────────────
    print_step_header(5, total_steps, "Register with primary")
    _register_with_primary(
        base_url=base_url, whoami_login=whoami.user_login,
        local_ts_ip=local_ts_ip, role=role, label=label,
    )

    # ── 6/6 — Verify ────────────────────────────────────────────────────
    print_step_header(6, total_steps, "Verify")
    _verify_inventory(base_url, local_ts_ip)

    console.print()
    print_ok("Done. The satellite will auto-start on every login.")
    print_info("To remove: [bold]molebie-satellite uninstall[/bold]")


# ────────────────────── phase helpers ──────────────────────


def _check_prerequisites(primary: str, port: int):
    """Returns (whoami_info, base_url, local_ts_ip) or exits with error printed."""
    if sys.version_info < (3, 13):
        print_fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected; "
            f"need 3.13+."
        )
        raise typer.Exit(1)
    print_ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    whoami = get_tailscale_whoami()
    if whoami is None:
        print_fail("Couldn't read your Tailscale identity.")
        console.print(
            "    Make sure Tailscale is installed and signed in to the same "
            "tailnet as the primary."
        )
        raise typer.Exit(1)
    print_ok(f"Tailscale identity: {whoami.user_login}")

    local_ts_ip = get_tailscale_ip()
    if not local_ts_ip:
        print_fail("Couldn't determine this machine's Tailscale IP.")
        raise typer.Exit(1)

    base_url = f"http://{primary}:{port}"
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
        if resp.status_code >= 400:
            print_fail(
                f"Primary at {primary}:{port} returned HTTP {resp.status_code} "
                f"on /health — is the gateway running?"
            )
            raise typer.Exit(1)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        print_fail(f"Primary at {primary}:{port} unreachable: {exc}")
        console.print(
            "    Verify Tailscale connectivity between the two machines."
        )
        raise typer.Exit(1)
    print_ok(f"Primary reachable at {base_url} (/health → 200)")

    return whoami, base_url, local_ts_ip


def _choose_data_dir(override: Path | None, yes: bool) -> Path:
    """Pick a data directory (interactive default; --yes skips the prompt)."""
    if override is not None:
        chosen = override.expanduser().resolve()
    elif yes:
        chosen = default_data_dir()
    else:
        raw = ask_text(
            "Where should blobs live?", default=str(default_data_dir())
        )
        chosen = Path(raw).expanduser().resolve()

    chosen.mkdir(parents=True, exist_ok=True)
    print_ok(f"Data dir: {chosen}")
    return chosen


def _satellite_bind_port() -> int:
    """Return the port the satellite_storage app will bind to.

    Reads from ``satellite_storage.config.get_settings()`` so the wizard
    tracks operator overrides (``MOLEBIE_STORAGE_PORT``) rather than
    hardcoding 8090.
    """
    from satellite_storage.config import get_settings

    return get_settings().bind_port


def _wait_for_health(url: str, *, timeout_sec: float) -> bool:
    """Poll the given /health URL until it answers 200 or we time out."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except (httpx.TimeoutException, httpx.TransportError):
            pass
        time.sleep(_HEALTH_POLL_INTERVAL_SEC)
    return False


def _register_with_primary(
    *, base_url: str, whoami_login: str, local_ts_ip: str,
    role: str, label: str | None,
) -> None:
    """POST to ``/fleet/satellites/register`` and print the result.

    Exits with a friendly error on transport failure or non-2xx. Callers
    don't consume the returned satellite id — operators verify by running
    ``molebie-ai extend list`` on the primary — so this returns nothing.
    """
    payload: dict = {"host": local_ts_ip, "role": role}
    cleaned = (label or "").strip() or None
    if cleaned:
        payload["label"] = cleaned

    headers = {"Tailscale-User-Login": whoami_login}
    try:
        resp = httpx.post(
            f"{base_url}/fleet/satellites/register",
            headers=headers, json=payload, timeout=10.0,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        print_fail(f"Failed to reach the register endpoint: {exc}")
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


def _verify_inventory(base_url: str, local_ts_ip: str) -> None:
    """Confirm the primary's inventory now lists this satellite by host."""
    try:
        resp = httpx.get(f"{base_url}/fleet/inventory", timeout=5.0)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        print_warn(f"Couldn't re-fetch /fleet/inventory: {exc} (registration still succeeded).")
        return
    if resp.status_code != 200:
        print_warn(
            f"/fleet/inventory returned HTTP {resp.status_code} "
            f"(registration still succeeded)."
        )
        return
    body = resp.json()
    if any(s.get("host") == local_ts_ip for s in body.get("satellites", [])):
        print_ok(f"Primary's /fleet/inventory now lists {local_ts_ip}.")
    else:
        print_warn(
            f"/fleet/inventory didn't show {local_ts_ip} (registration responded OK "
            f"though — try `molebie-ai extend list` on the primary to verify)."
        )

