"""molebie-ai status — show current config and service state."""

from __future__ import annotations

import httpx
from rich.panel import Panel
from rich.table import Table

from cli.models.config import InferenceBackend, MolebieConfig, SetupType
from cli.services import config_manager
from cli.ui.console import console


def _health(url: str, timeout: float = 2.0) -> str:
    """Return 'up', 'down', or 'timeout'."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        return "[green]up[/green]" if resp.status_code < 400 else f"[red]{resp.status_code}[/red]"
    except httpx.TimeoutException:
        return "[yellow]timeout[/yellow]"
    except httpx.TransportError:
        return "[red]down[/red]"


def status() -> None:
    """Show current configuration and service state."""
    console.print()

    # Load config
    if not config_manager.config_exists():
        console.print("[warn]No configuration found. Run [bold]molebie-ai install[/bold] first.[/warn]")
        console.print()
        return

    config = config_manager.load_config()

    # Config summary panel
    lines = [
        f"Setup type:  [cyan]{config.setup_type.value}[/cyan]",
        f"Backend:     [cyan]{config.inference_backend.value}[/cyan]",
    ]
    if config.setup_type == SetupType.TWO_MACHINE:
        lines.append(f"GPU IP:      [cyan]{config.gpu_ip}[/cyan]")
        lines.append(f"Server IP:   [cyan]{config.server_ip}[/cyan]")
    lines.append(f"Search:      {'[green]enabled[/green]' if config.search_enabled else '[dim]disabled[/dim]'}")
    lines.append(f"RAG:         {'[green]enabled[/green]' if config.rag_enabled else '[dim]disabled[/dim]'}")
    lines.append(f"Voice:       {'[green]enabled[/green]' if config.voice_enabled else '[dim]disabled[/dim]'}")
    if config.last_install_at:
        lines.append(f"Installed:   [dim]{config.last_install_at}[/dim]")

    console.print(Panel("\n".join(lines), title="Configuration", expand=False))
    console.print()

    # Service status table
    ip = config.server_ip if config.setup_type == SetupType.TWO_MACHINE else "localhost"
    gpu_ip = config.gpu_ip if config.setup_type == SetupType.TWO_MACHINE else "localhost"

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Service", style="cyan", min_width=16)
    table.add_column("URL", min_width=30)
    table.add_column("Status", min_width=10)

    services: list[tuple[str, str]] = [
        ("Gateway", f"http://{ip}:8000/health"),
        ("Webapp", f"http://{ip}:3000"),
    ]

    if config.inference_backend == InferenceBackend.MLX:
        services.append(("MLX Thinking", f"http://{gpu_ip}:8080/v1/models"))
        services.append(("MLX Instant", f"http://{gpu_ip}:8081/v1/models"))
    elif config.inference_backend == InferenceBackend.OLLAMA:
        services.append(("Ollama", f"http://{gpu_ip}:11434/v1/models"))
    elif config.inference_backend == InferenceBackend.OPENAI_COMPATIBLE:
        if config.inference_url:
            services.append(("Inference", f"{config.inference_url}/v1/models"))

    if config.search_enabled:
        services.append(("SearXNG", f"http://{ip}:8888/"))
    if config.voice_enabled:
        services.append(("Kokoro TTS", f"http://{ip}:8880/"))

    for name, url in services:
        table.add_row(name, url, _health(url))

    console.print(table)
    console.print()
