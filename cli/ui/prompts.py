"""Interactive prompt helpers using Rich."""

from __future__ import annotations

from rich.prompt import Confirm, Prompt

from cli.ui.console import console


def ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Ask user to pick from a numbered list. Returns the chosen string."""
    console.print(f"[heading]{prompt}[/heading]")
    for i, choice in enumerate(choices, 1):
        marker = " [dim](default)[/dim]" if choice == default else ""
        console.print(f"  [{i}] {choice}{marker}")
    console.print()

    valid = {str(i): c for i, c in enumerate(choices, 1)}
    while True:
        raw = Prompt.ask("Choose", default="1" if default else None, console=console)
        if raw in valid:
            return valid[raw]
        # Also accept the choice text directly
        if raw in choices:
            return raw
        console.print(f"  [warn]Please enter a number 1–{len(choices)}[/warn]")


def ask_text(prompt: str, default: str = "") -> str:
    return Prompt.ask(prompt, default=default, console=console)


def ask_confirm(prompt: str, default: bool = True) -> bool:
    return Confirm.ask(prompt, default=default, console=console)


def ask_model_profile(recommended: str, ram_gb: float, backend_name: str) -> str:
    """Ask user to choose a model profile with RAM-aware recommendation."""
    from cli.services.backend_setup import (
        MLX_MODELS, OLLAMA_MODELS, PROFILE_DESCRIPTIONS, PROFILE_MIN_RAM,
    )

    catalog = MLX_MODELS if "mlx" in backend_name.lower() else OLLAMA_MODELS
    profiles = list(catalog.keys())

    console.print(f"[heading]Choose a model profile for {backend_name}:[/heading]")
    for i, name in enumerate(profiles, 1):
        thinking, instant = catalog[name]
        min_ram = PROFILE_MIN_RAM.get(name, 8)
        desc = PROFILE_DESCRIPTIONS.get(name, "")
        rec = " [green](recommended)[/green]" if name == recommended else ""

        # Short model display names
        t_short = thinking.split("/")[-1] if "/" in thinking else thinking
        i_short = instant.split("/")[-1] if "/" in instant else instant

        console.print(f"  [{i}] [bold]{name.capitalize()}[/bold] — {desc}, {min_ram}+ GB RAM{rec}")
        console.print(f"      Thinking: {t_short}  |  Instant: {i_short}")
    console.print()

    if ram_gb > 0:
        console.print(f"  [dim]Your system: {ram_gb} GB RAM[/dim]")
        console.print()

    # Default to recommended
    default_idx = str(profiles.index(recommended) + 1) if recommended in profiles else "1"
    valid = {str(i): p for i, p in enumerate(profiles, 1)}

    while True:
        raw = Prompt.ask("Choose", default=default_idx, console=console)
        if raw in valid:
            return valid[raw]
        if raw.lower() in profiles:
            return raw.lower()
        console.print(f"  [warn]Please enter a number 1–{len(profiles)}[/warn]")


def ask_features(available: dict[str, bool]) -> dict[str, bool]:
    """Toggle optional features on/off. Returns updated dict."""
    console.print("[heading]Optional Features[/heading]")
    result = {}
    for name, current in available.items():
        from cli.models.config import FEATURE_DESCRIPTIONS
        desc = FEATURE_DESCRIPTIONS.get(name, name)
        label = f"Enable {name}? ({desc})"
        result[name] = Confirm.ask(f"  {label}", default=current, console=console)
    return result
