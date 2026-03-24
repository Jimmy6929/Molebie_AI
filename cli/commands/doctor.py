"""molebie-ai doctor — diagnose environment and setup problems."""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from cli.models.config import MolebieConfig
from cli.services import config_manager, prerequisite_checker
from cli.ui.console import console, make_status_table, print_fail, print_ok, print_warn


def _check_file(path: Path, label: str) -> bool:
    if path.exists():
        print_ok(f"{label} exists")
        return True
    print_fail(f"{label} missing")
    return False


def _check_env_keys(env_path: Path) -> bool:
    """Check that .env.local doesn't still have placeholder keys."""
    if not env_path.exists():
        return False
    text = env_path.read_text()
    placeholders = ["YOUR_ANON_KEY_HERE", "YOUR_SERVICE_ROLE_KEY_HERE", "YOUR_JWT_SECRET_HERE"]
    found = [p for p in placeholders if p in text]
    if found:
        print_warn(f".env.local still has placeholder keys: {', '.join(found)}")
        console.print("    Fix: run [bold]molebie-ai install[/bold] or update .env.local manually")
        return False
    print_ok(".env.local keys are populated")
    return True


def _health_check(url: str, name: str, timeout: float = 3.0) -> bool:
    """Quick HTTP health check."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        if resp.status_code < 400:
            print_ok(f"{name} reachable at {url}")
            return True
        print_fail(f"{name} returned {resp.status_code} at {url}")
        return False
    except httpx.ConnectError:
        print_fail(f"{name} not reachable at {url}")
        return False
    except httpx.TimeoutException:
        print_warn(f"{name} timed out at {url}")
        return False


def doctor() -> None:
    """Diagnose environment and setup problems."""
    console.print()
    console.print("[heading]Molebie AI — Doctor[/heading]")
    console.print()
    all_ok = True

    # 1. Prerequisites
    console.print("[heading]Prerequisites[/heading]")
    results = prerequisite_checker.check_all()
    for r in results:
        if r.passed:
            print_ok(f"{r.name}: {r.message}")
        elif r.is_warning:
            print_warn(f"{r.name}: {r.message}")
            if r.fix_hint:
                console.print(f"    Fix: {r.fix_hint}")
        else:
            print_fail(f"{r.name}: {r.message}")
            if r.fix_hint:
                console.print(f"    Fix: {r.fix_hint}")
            all_ok = False
    console.print()

    # 2. Config file
    console.print("[heading]Configuration[/heading]")
    try:
        root = config_manager.get_project_root()
    except FileNotFoundError:
        print_fail("Not inside a Molebie AI project directory")
        raise typer.Exit(1)

    config_path = config_manager.get_config_path()
    if config_path.exists():
        print_ok("Config file exists")
        try:
            config = config_manager.load_config()
            print_ok(f"Config valid (setup={config.setup_type.value}, backend={config.inference_backend.value})")
        except Exception as e:
            print_fail(f"Config file invalid: {e}")
            all_ok = False
            config = MolebieConfig()
    else:
        print_warn("No config file — run [bold]molebie-ai install[/bold] first")
        config = MolebieConfig()
    console.print()

    # 3. Environment files
    console.print("[heading]Environment[/heading]")
    env_path = root / ".env.local"
    _check_file(env_path, ".env.local")
    if env_path.exists():
        if not _check_env_keys(env_path):
            all_ok = False
    else:
        all_ok = False
    console.print()

    # 4. Service health
    console.print("[heading]Services[/heading]")
    ip = config.server_ip if config.setup_type.value == "two-machine" else "localhost"
    gpu_ip = config.gpu_ip if config.setup_type.value == "two-machine" else "localhost"

    services = [
        (f"http://{ip}:54321/rest/v1/", "Supabase"),
        (f"http://{ip}:8000/health", "Gateway"),
        (f"http://{ip}:3000", "Webapp"),
    ]

    # Inference endpoints
    if config.inference_backend == config.inference_backend.MLX:
        services.append((f"http://{gpu_ip}:8080/v1/models", "MLX Thinking"))
        services.append((f"http://{gpu_ip}:8081/v1/models", "MLX Instant"))
    elif config.inference_backend == config.inference_backend.OLLAMA:
        services.append((f"http://{gpu_ip}:11434/v1/models", "Ollama"))

    # Optional services
    if config.search_enabled:
        services.append((f"http://{ip}:8888/", "SearXNG"))
    if config.voice_enabled:
        services.append((f"http://{ip}:8880/", "Kokoro TTS"))

    for url, name in services:
        _health_check(url, name)
    console.print()

    # Summary
    if all_ok:
        console.print("[ok]All checks passed.[/ok]")
    else:
        console.print("[warn]Some checks failed. See above for details.[/warn]")
    console.print()
