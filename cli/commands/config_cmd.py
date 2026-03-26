"""molebie-ai config — view and manage configuration."""

from __future__ import annotations

import json

import typer
from rich.syntax import Syntax
from rich.table import Table

from cli.services import config_manager
from cli.services.config_manager import get_project_root
from cli.services.env_generator import (
    get_env_key,
    get_valid_keys,
    parse_env_file,
    update_env_key,
)
from cli.ui.console import console, print_fail, print_ok, print_warn

app = typer.Typer(no_args_is_help=True)

# Keys containing any of these substrings are treated as sensitive
SENSITIVE_PATTERNS = ("SECRET", "API_KEY", "PASSWORD")


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


def _mask_value(key: str, value: str, show_secrets: bool = False) -> str:
    """Mask sensitive env values unless --show-secrets is set."""
    if show_secrets:
        return value
    if any(pat in key for pat in SENSITIVE_PATTERNS):
        if len(value) > 8:
            return value[:4] + "****" + value[-4:]
        return "****"
    return value


# ── Existing command ──────────────────────────────────────────


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


# ── Environment variable commands ─────────────────────────────


@app.command(name="env")
def env_list(
    show_secrets: bool = typer.Option(False, "--show-secrets", help="Show sensitive values unmasked"),
) -> None:
    """List all environment variables from .env.local."""
    console.print()
    env_path = get_project_root() / ".env.local"

    if not env_path.exists():
        print_fail("No .env.local found. Run [bold]molebie-ai install[/bold] first.")
        console.print()
        raise typer.Exit(1)

    entries = parse_env_file(env_path)
    table = Table(show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Variable", style="cyan", no_wrap=True)
    table.add_column("Value")

    for kind, key_or_text, value in entries:
        if kind == "section":
            table.add_section()
            table.add_row(f"[bold]{key_or_text}[/bold]", "")
        elif kind == "var":
            table.add_row(key_or_text, _mask_value(key_or_text, value or "", show_secrets))

    console.print(table)
    console.print()
    console.print(f"[dim]{env_path}[/dim]")
    console.print()


@app.command(name="get")
def env_get(
    key: str = typer.Argument(help="Environment variable name (e.g. INFERENCE_THINKING_URL)"),
    show_secrets: bool = typer.Option(False, "--show-secrets", help="Show sensitive values unmasked"),
) -> None:
    """Show the current value of an environment variable."""
    key = key.upper()
    value = get_env_key(key)

    if value is None:
        print_fail(f"{key} not found in .env.local")
        raise typer.Exit(1)

    print_ok(f"{key}={_mask_value(key, value, show_secrets)}")


@app.command(name="set")
def env_set(
    assignment: str = typer.Argument(help="KEY=VALUE pair (e.g. INFERENCE_THINKING_URL=http://gpu:8080)"),
) -> None:
    """Update an environment variable in .env.local."""
    key, sep, value = assignment.partition("=")
    if not sep:
        print_fail("Expected KEY=VALUE format (e.g. INFERENCE_THINKING_URL=http://gpu:8080)")
        raise typer.Exit(1)

    key = key.upper()

    # Validate against .env.example to prevent typos
    valid_keys = get_valid_keys()
    if valid_keys and key not in valid_keys:
        print_fail(f'"{key}" is not a recognized variable. Check .env.example for valid keys.')
        raise typer.Exit(1)

    # For sensitive keys with empty value, prompt interactively (avoids shell history)
    if not value and any(pat in key for pat in SENSITIVE_PATTERNS):
        from cli.ui.prompts import ask_text
        value = ask_text(f"Enter value for {key}")

    if update_env_key(key, value):
        print_ok(f"{key} updated")
    else:
        print_fail(f"{key} not found in .env.local. Run [bold]molebie-ai install[/bold] to regenerate it.")
        raise typer.Exit(1)
