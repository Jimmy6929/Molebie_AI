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
