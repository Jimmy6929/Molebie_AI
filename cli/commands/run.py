"""molebie-ai run — start all configured services."""

from __future__ import annotations

from typing import Optional

import httpx
import typer

from cli.models.config import InferenceBackend, SetupType
from cli.services import config_manager
from cli.services.service_manager import ServiceRunner
from cli.ui.console import console, print_banner, print_fail, print_ok, print_warn


def _check_remote_inference(config) -> None:
    """For two-machine setups, verify the remote GPU is reachable before starting local services."""
    if config.setup_type != SetupType.TWO_MACHINE:
        return

    gpu_ip = config.gpu_ip

    if config.inference_backend == InferenceBackend.MLX:
        endpoints = [
            ("MLX Thinking", f"http://{gpu_ip}:8080/v1/models"),
            ("MLX Instant", f"http://{gpu_ip}:8081/v1/models"),
        ]
    elif config.inference_backend == InferenceBackend.OLLAMA:
        endpoints = [("Ollama", f"http://{gpu_ip}:11434/v1/models")]
    elif config.inference_backend == InferenceBackend.OPENAI_COMPATIBLE and config.inference_url:
        endpoints = [("Inference", f"{config.inference_url}/v1/models")]
    else:
        return

    console.print(f"[heading]Checking remote inference ({gpu_ip})...[/heading]")
    all_ok = True
    for name, url in endpoints:
        try:
            resp = httpx.get(url, timeout=5.0, follow_redirects=True)
            if resp.status_code < 400:
                print_ok(f"{name} reachable at {url}")
            else:
                print_warn(f"{name} returned {resp.status_code} at {url}")
                all_ok = False
        except httpx.ConnectError:
            print_fail(f"{name} not reachable at {url}")
            all_ok = False
        except httpx.TimeoutException:
            print_warn(f"{name} timed out at {url}")
            all_ok = False

    if not all_ok:
        console.print()
        console.print(f"[warn]Inference on GPU machine ({gpu_ip}) is not fully reachable.[/warn]")
        console.print(f"  Make sure the LLM servers are running on {gpu_ip}.")
        console.print()
    else:
        console.print()


def run(
    service: Optional[str] = typer.Option(
        None, "--service", "-s",
        help="Start only a specific service (e.g., gateway, webapp, supabase)",
    ),
    no_inference: bool = typer.Option(
        False, "--no-inference",
        help="Skip starting inference servers (MLX/Ollama)",
    ),
) -> None:
    """Start all configured services."""
    print_banner()

    if not config_manager.config_exists():
        console.print("[warn]No configuration found. Run [bold]molebie-ai install[/bold] first.[/warn]")
        console.print()
        raise typer.Exit(1)

    config = config_manager.load_config()

    if not config.installed:
        console.print("[warn]Installation not complete. Run [bold]molebie-ai install[/bold] first.[/warn]")
        console.print()
        raise typer.Exit(1)

    _check_remote_inference(config)

    runner = ServiceRunner()
    runner.start_services(config, service_filter=service, skip_inference=no_inference)
