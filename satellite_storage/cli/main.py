"""molebie-satellite — typer app + subcommand registration.

This is the entry point exposed via the ``[project.scripts]`` table in
``pyproject.toml`` (``molebie-satellite = "satellite_storage.cli.main:app"``).
Each subcommand lives in its own sibling module and is registered here so
the help output stays a single source of truth.
"""

from __future__ import annotations

import typer

from satellite_storage import __version__
from satellite_storage.cli.install import install_command
from satellite_storage.cli.join import join_command
from satellite_storage.cli.serve import serve_command
from satellite_storage.cli.uninstall import uninstall_command

app = typer.Typer(
    name="molebie-satellite",
    help="Standalone satellite-side blob storage service for Molebie AI fleets.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("version")
def version_command() -> None:
    """Print the molebie-satellite package version."""
    typer.echo(f"molebie-satellite {__version__}")


app.command("serve", help="Start the satellite blob service in the foreground.")(serve_command)
app.command("join", help="Register this machine with a Molebie primary.")(join_command)
app.command("install", help="Interactive wizard: install as a service + register.")(install_command)
app.command("uninstall", help="Remove the service unit and optionally the data directory.")(uninstall_command)
