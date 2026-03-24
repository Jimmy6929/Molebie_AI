"""molebie-ai config — view and manage configuration."""

from __future__ import annotations

import json

import typer
from rich.syntax import Syntax

from cli.services import config_manager
from cli.ui.console import console

app = typer.Typer(no_args_is_help=True)


def _mask_sensitive(data: dict) -> dict:
    """Mask sensitive values like API keys."""
    masked = dict(data)
    for key in ("inference_api_key",):
        if masked.get(key):
            val = masked[key]
            if len(val) > 8:
                masked[key] = val[:4] + "..." + val[-4:]
            else:
                masked[key] = "****"
    return masked


@app.command(name="show")
def show() -> None:
    """Display the saved setup configuration."""
    console.print()

    if not config_manager.config_exists():
        console.print("[warn]No configuration found. Run [bold]molebie-ai install[/bold] first.[/warn]")
        console.print()
        raise typer.Exit(0)

    config = config_manager.load_config()
    data = _mask_sensitive(config.model_dump(mode="json"))
    formatted = json.dumps(data, indent=2)

    console.print(Syntax(formatted, "json", theme="monokai", line_numbers=False))
    console.print()
    console.print(f"[dim]Config file: {config_manager.get_config_path()}[/dim]")
    console.print()
