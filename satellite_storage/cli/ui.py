"""Minimal rich-console helpers for the satellite CLI.

Deliberate small-duplication of the primary's ``cli/ui/console.py`` and
``cli/ui/prompts.py`` — the satellite is shipped as an independent
``pipx``-installable package, so it can't import from the primary's repo.
We only re-implement the ~5 helpers the install/join/uninstall flows
actually use; not the whole UI library.
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def print_ok(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def print_fail(msg: str) -> None:
    console.print(f"  [red]✗[/red] {msg}")


def print_warn(msg: str) -> None:
    console.print(f"  [yellow]![/yellow] {msg}")


def print_info(msg: str) -> None:
    console.print(f"  [cyan]→[/cyan] {msg}")


def print_step_header(step: int, total: int, title: str) -> None:
    console.print()
    console.print(f"[bold]Step {step}/{total}[/bold]  {title}")


def ask_confirm(prompt: str, default: bool = True) -> bool:
    return typer.confirm(prompt, default=default)


def ask_text(prompt: str, default: str | None = None) -> str:
    return typer.prompt(prompt, default=default)
