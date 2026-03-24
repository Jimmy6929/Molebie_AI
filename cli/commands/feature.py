"""molebie-ai feature — manage optional features."""

from __future__ import annotations

import typer
from rich.table import Table

from cli.models.config import FEATURE_DESCRIPTIONS, VALID_FEATURES
from cli.services import config_manager, env_generator
from cli.ui.console import console, print_fail, print_ok

app = typer.Typer(no_args_is_help=True)


def _feature_field(feature: str) -> str:
    """Map feature name to config field name."""
    return f"{feature}_enabled"


@app.command(name="list")
def list_features() -> None:
    """Show all features and their current status."""
    console.print()

    config = config_manager.load_config()

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Feature", style="cyan", min_width=10)
    table.add_column("Status", min_width=10)
    table.add_column("Description")

    for name in VALID_FEATURES:
        enabled = getattr(config, _feature_field(name), False)
        status = "[green]enabled[/green]" if enabled else "[dim]disabled[/dim]"
        desc = FEATURE_DESCRIPTIONS.get(name, "")
        table.add_row(name, status, desc)

    console.print(table)
    console.print()


@app.command(name="add")
def add_feature(feature: str = typer.Argument(help="Feature to enable (voice, search, rag)")) -> None:
    """Enable an optional feature."""
    feature = feature.lower()
    if feature not in VALID_FEATURES:
        print_fail(f"Unknown feature: {feature}")
        console.print(f"  Available: {', '.join(VALID_FEATURES)}")
        raise typer.Exit(1)

    config = config_manager.load_config()
    field_name = _feature_field(feature)

    if getattr(config, field_name):
        print_ok(f"{feature} is already enabled")
        return

    setattr(config, field_name, True)
    config_manager.save_config(config)

    # Update .env.local if it exists
    env_keys = {
        "search": ("WEB_SEARCH_ENABLED", "true"),
        "rag": ("RAG_ENABLED", "true"),
    }
    if feature in env_keys:
        key, val = env_keys[feature]
        env_generator.update_env_key(key, val)

    print_ok(f"{feature} enabled")

    if feature == "voice":
        console.print("  [dim]Start Kokoro TTS: docker compose up -d kokoro-tts[/dim]")
    elif feature == "search":
        console.print("  [dim]Start SearXNG: docker compose up -d searxng[/dim]")

    console.print()


@app.command(name="remove")
def remove_feature(feature: str = typer.Argument(help="Feature to disable (voice, search, rag)")) -> None:
    """Disable an optional feature."""
    feature = feature.lower()
    if feature not in VALID_FEATURES:
        print_fail(f"Unknown feature: {feature}")
        console.print(f"  Available: {', '.join(VALID_FEATURES)}")
        raise typer.Exit(1)

    config = config_manager.load_config()
    field_name = _feature_field(feature)

    if not getattr(config, field_name):
        print_ok(f"{feature} is already disabled")
        return

    setattr(config, field_name, False)
    config_manager.save_config(config)

    # Update .env.local if it exists
    env_keys = {
        "search": ("WEB_SEARCH_ENABLED", "false"),
        "rag": ("RAG_ENABLED", "false"),
    }
    if feature in env_keys:
        key, val = env_keys[feature]
        env_generator.update_env_key(key, val)

    print_ok(f"{feature} disabled")
    console.print()
