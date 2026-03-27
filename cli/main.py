"""Molebie AI CLI — main entrypoint."""

from __future__ import annotations

import typer

from cli import __version__
from cli.commands import config_cmd, doctor, feature, install, model_cmd, run, status

app = typer.Typer(
    name="molebie-ai",
    help="Manage your Molebie AI installation.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Top-level commands
app.command(name="install", help="Interactive setup wizard")(install.install)
app.command(name="run", help="Start all configured services")(run.run)
app.command(name="doctor", help="Diagnose environment and setup")(doctor.doctor)
app.command(name="status", help="Show current config and service state")(status.status)

# Sub-apps
app.add_typer(config_cmd.app, name="config", help="View and manage configuration")
app.add_typer(feature.app, name="feature", help="Manage optional features")
app.add_typer(model_cmd.app, name="model", help="Manage LLM models (download, remove, start/stop)")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"molebie-ai {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Molebie AI — self-hosted AI assistant platform."""
