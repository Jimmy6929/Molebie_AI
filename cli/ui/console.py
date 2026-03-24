"""Shared Rich console and output helpers."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

theme = Theme(
    {
        "info": "blue",
        "ok": "green",
        "warn": "yellow",
        "fail": "red bold",
        "heading": "bold",
    }
)

console = Console(theme=theme)


def print_banner() -> None:
    console.print()
    console.print(
        Panel(
            "[heading]Molebie AI[/heading]",
            subtitle="v0.1.0",
            style="blue",
            expand=False,
        )
    )
    console.print()


def print_ok(msg: str) -> None:
    console.print(f"  [ok]✓[/ok] {msg}")


def print_fail(msg: str) -> None:
    console.print(f"  [fail]✗[/fail] {msg}")


def print_warn(msg: str) -> None:
    console.print(f"  [warn]![/warn] {msg}")


def print_info(msg: str) -> None:
    console.print(f"  [info]→[/info] {msg}")


def print_step_header(step: int, total: int, title: str) -> None:
    """Print a visually distinct install phase header."""
    console.print()
    console.rule(f"[heading]Step {step}/{total}: {title}[/heading]", style="blue")
    console.print()


def make_status_table() -> Table:
    """Create a reusable service-status table."""
    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Service", style="cyan", min_width=16)
    table.add_column("URL", min_width=30)
    table.add_column("Status", min_width=10)
    return table
