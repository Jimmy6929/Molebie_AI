"""``molebie-satellite uninstall`` — remove the OS service and optionally data.

Default: tear down the service unit (launchd plist / systemd unit / Task
Scheduler task) but leave the blob data dir intact. Pass ``--purge`` to
also delete the data dir — explicit and destructive, so it's never the
default.

Idempotent: re-running on an already-uninstalled host succeeds silently.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from satellite_storage.cli.service import (
    SATELLITE_SERVICE_LABEL,
    default_data_dir,
    is_service_installed,
    uninstall_service,
)
from satellite_storage.cli.ui import (
    ask_confirm,
    console,
    print_info,
    print_ok,
    print_warn,
)


def uninstall_command(
    purge: bool = typer.Option(
        False, "--purge",
        help="Also delete the data directory (destructive — blobs are lost).",
    ),
    data_dir: Path | None = typer.Option(
        None, help="Override the data directory location (only used with --purge)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompts."
    ),
) -> None:
    """Remove the satellite service unit (and optionally the data directory)."""
    console.print()
    console.print("[bold]Molebie AI — Satellite Uninstall[/bold]")
    console.print()

    if is_service_installed():
        uninstall_service(SATELLITE_SERVICE_LABEL)
        print_ok("Removed OS service unit.")
    else:
        print_info("No service unit found — nothing to remove.")

    if purge:
        target = (data_dir or default_data_dir()).expanduser().resolve()
        if not target.exists():
            print_info(f"Data dir {target} doesn't exist — nothing to purge.")
            return
        if not yes:
            console.print()
            console.print(
                f"  [red]About to delete:[/red] {target}\n"
                f"  This permanently destroys any blobs still on this satellite."
            )
            if not ask_confirm("Continue with purge?", default=False):
                print_warn("Skipped: data directory left intact.")
                return
        shutil.rmtree(target)
        print_ok(f"Purged data directory: {target}")
    else:
        print_info(
            "Data directory left intact. Pass [bold]--purge[/bold] to delete it."
        )
