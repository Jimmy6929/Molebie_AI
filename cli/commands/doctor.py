"""molebie-ai doctor — diagnose environment and setup problems."""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from cli.models.config import InferenceBackend, MolebieConfig
from cli.services import backend_setup, config_manager, prerequisite_checker
from cli.services.env_generator import init_env_local
from cli.ui.console import console, make_status_table, print_fail, print_info, print_ok, print_warn


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
    placeholders = ["YOUR_ANON_KEY_HERE", "YOUR_SERVICE_ROLE_KEY_HERE", "YOUR_JWT_SECRET_HERE", "CHANGE_ME_TO_A_RANDOM_SECRET"]
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
    except httpx.TimeoutException:
        print_warn(f"{name} timed out at {url}")
        return False
    except httpx.TransportError:
        print_fail(f"{name} not reachable at {url}")
        return False


def doctor(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix missing .env.local and config"),
    deep: bool = typer.Option(False, "--deep", help="Run deep application-logic checks (DB, embeddings, vector search)"),
) -> None:
    """Diagnose environment and setup problems."""
    console.print()
    console.print("[heading]Molebie AI — Doctor[/heading]")
    console.print()
    all_ok = True

    # 1. Prerequisites — load config early so we can pass feature flags
    #    (Docker is a hard failure when search/voice is enabled, warning otherwise)
    _config_path = config_manager.get_config_path()
    _pre_config = config_manager.load_config() if _config_path.exists() else MolebieConfig()

    console.print("[heading]Prerequisites[/heading]")
    results = prerequisite_checker.check_all(
        voice_enabled=_pre_config.voice_enabled,
        search_enabled=_pre_config.search_enabled,
    )
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
    elif fix:
        config = MolebieConfig(installed=True)
        config_manager.ensure_config_dir()
        config_manager.save_config(config)
        print_ok("Config file created with defaults")
    else:
        print_warn("No config file — run [bold]molebie-ai doctor --fix[/bold] or [bold]molebie-ai install[/bold]")
        config = MolebieConfig()
    console.print()

    # 3. Environment files
    console.print("[heading]Environment[/heading]")
    env_path = root / ".env.local"
    if not env_path.exists() and fix:
        init_env_local()
        print_ok(".env.local generated from template")
    elif not env_path.exists():
        print_fail(".env.local missing")
        console.print("    Fix: run [bold]molebie-ai doctor --fix[/bold] or [bold]molebie-ai config init[/bold]")
        all_ok = False
    else:
        print_ok(".env.local exists")

    if env_path.exists():
        if not _check_env_keys(env_path):
            all_ok = False
    console.print()

    # 4. Backend dependencies — deps the chosen inference backend needs at runtime
    # but that aren't declared by its pip package (silent failures at request time).
    if config.run_inference and config.inference_backend == InferenceBackend.MLX:
        console.print("[heading]Backend dependencies[/heading]")
        if backend_setup.is_torchvision_installed():
            print_ok("torchvision installed")
        elif fix:
            console.print("    Installing torchvision (required for image uploads)…")
            if backend_setup.install_torchvision():
                print_ok("torchvision installed")
            else:
                print_fail("Failed to install torchvision")
                console.print("    Fix: [bold]python -m pip install -U torchvision[/bold]")
                all_ok = False
        else:
            print_fail("torchvision missing — image uploads will return HTTP 500 from the MLX server")
            console.print("    Fix: run [bold]molebie-ai doctor --fix[/bold]")
            all_ok = False
        console.print()

    # 5. Service health
    console.print("[heading]Services[/heading]")
    services: list[tuple[str, str]] = []

    # Local services — always check what runs on this machine
    if config.run_gateway:
        services.append(("http://localhost:8000/health", "Gateway"))
    if config.run_webapp:
        services.append(("http://localhost:3000", "Webapp"))
    if config.run_inference:
        if config.inference_backend == InferenceBackend.MLX:
            services.append(("http://localhost:8080/v1/models", "MLX Thinking"))
            services.append(("http://localhost:8081/v1/models", "MLX Instant"))
        elif config.inference_backend == InferenceBackend.OLLAMA:
            services.append(("http://localhost:11434/v1/models", "Ollama"))

    # Remote services — only those this machine actually connects to
    for name, url in config.required_remote_endpoints():
        services.append((url, f"{name} (remote)"))

    # Optional services (co-located with gateway)
    if config.search_enabled and config.run_gateway:
        services.append(("http://localhost:8888/", "SearXNG"))
    if config.voice_enabled and config.run_gateway:
        services.append(("http://localhost:8880/health", "Kokoro TTS"))

    for url, name in services:
        if not _health_check(url, name):
            all_ok = False

    if not services:
        print_info("No services to check on this machine")

    console.print()

    # 6. Deep application-logic checks (optional)
    if deep:
        console.print("[heading]Deep Application Checks[/heading]")
        from cli.services.deep_checker import run_deep_checks

        deep_results = run_deep_checks(root)
        for r in deep_results:
            if r.skipped:
                print_warn(f"{r.name}: SKIPPED — {r.message}")
            elif r.passed:
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

    # Summary
    if all_ok:
        console.print("[ok]All checks passed.[/ok]")
    else:
        console.print("[warn]Some checks failed. See above for details.[/warn]")
    console.print()
