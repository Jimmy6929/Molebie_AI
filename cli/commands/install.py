"""molebie-ai install — interactive setup wizard."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import typer
from rich.table import Table

from cli.models.config import (
    FEATURE_DESCRIPTIONS,
    InferenceBackend,
    MolebieConfig,
    SetupType,
)
from cli.services import config_manager, env_generator, prerequisite_checker, supabase_manager
from cli.ui.console import console, print_banner, print_fail, print_info, print_ok, print_warn
from cli.ui.prompts import ask_choice, ask_confirm, ask_features, ask_text


def _wizard() -> MolebieConfig:
    """Run the interactive setup wizard. Returns config from user choices."""
    config = MolebieConfig()

    # ── Step 1: Setup type ──────────────────────────────────────
    console.print("[heading]Step 1/4: Setup Type[/heading]")
    console.print()
    setup_choice = ask_choice(
        "How will you run Molebie AI?",
        [
            "Single machine — everything on this Mac",
            "Two machines — GPU on a separate Mac (Tailscale/LAN)",
        ],
        default="Single machine — everything on this Mac",
    )

    if "Two" in setup_choice:
        config.setup_type = SetupType.TWO_MACHINE
        console.print()
        console.print("  Enter the IP addresses of your two machines.")
        console.print("  [dim](Use Tailscale IPs for reliable connectivity)[/dim]")
        config.gpu_ip = ask_text("  IP of GPU machine (runs inference)", default="")
        config.server_ip = ask_text("  IP of THIS machine (runs gateway/webapp)", default="")
        if not config.gpu_ip or not config.server_ip:
            print_fail("Both IPs are required for two-machine setup.")
            raise typer.Exit(1)
        print_ok(f"Two-machine mode: GPU={config.gpu_ip}, Server={config.server_ip}")
    else:
        config.setup_type = SetupType.SINGLE
        print_ok("Single-machine mode: all services on localhost")

    console.print()

    # ── Step 2: Inference backend ────────────────────────────────
    console.print("[heading]Step 2/4: Inference Backend[/heading]")
    console.print()
    backend_choice = ask_choice(
        "Choose your inference backend:",
        [
            "MLX (Apple Silicon — recommended for Mac)",
            "Ollama (cross-platform, easy setup)",
            "OpenAI-compatible (vLLM, llama.cpp, OpenAI API, etc.)",
        ],
        default="MLX (Apple Silicon — recommended for Mac)",
    )

    if "MLX" in backend_choice:
        config.inference_backend = InferenceBackend.MLX
    elif "Ollama" in backend_choice:
        config.inference_backend = InferenceBackend.OLLAMA
    else:
        config.inference_backend = InferenceBackend.OPENAI_COMPATIBLE
        console.print()
        config.inference_url = ask_text(
            "  Base URL for inference server",
            default="http://localhost:8080",
        )
        api_key = ask_text("  API key (leave empty if none)", default="")
        config.inference_api_key = api_key if api_key else None

    print_ok(f"Backend: {config.inference_backend.value}")
    console.print()

    # ── Step 3: Features ─────────────────────────────────────────
    console.print("[heading]Step 3/4: Optional Features[/heading]")
    console.print()
    features = ask_features({
        "search": True,
        "rag": True,
        "voice": False,
    })
    config.search_enabled = features.get("search", True)
    config.rag_enabled = features.get("rag", True)
    config.voice_enabled = features.get("voice", False)
    console.print()

    # ── Step 4: Review ───────────────────────────────────────────
    console.print("[heading]Step 4/4: Review[/heading]")
    console.print()

    table = Table(show_header=False, expand=False, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Setup type", config.setup_type.value)
    if config.setup_type == SetupType.TWO_MACHINE:
        table.add_row("GPU IP", config.gpu_ip)
        table.add_row("Server IP", config.server_ip)
    table.add_row("Inference backend", config.inference_backend.value)
    if config.inference_url:
        table.add_row("Inference URL", config.inference_url)
    table.add_row("Web Search", "enabled" if config.search_enabled else "disabled")
    table.add_row("RAG", "enabled" if config.rag_enabled else "disabled")
    table.add_row("Voice", "enabled" if config.voice_enabled else "disabled")

    console.print(table)
    console.print()

    if not ask_confirm("Proceed with installation?", default=True):
        print_info("Installation cancelled.")
        raise typer.Exit(0)

    return config


def _run_install(config: MolebieConfig) -> None:
    """Execute the installation steps after wizard completes."""
    root = config_manager.get_project_root()
    console.print()

    # 1. Check prerequisites
    with console.status("[info]Checking prerequisites...[/info]"):
        results = prerequisite_checker.check_all(voice_enabled=config.voice_enabled)
    has_failures = any(not r.passed and not r.is_warning for r in results)
    for r in results:
        if r.passed:
            print_ok(f"{r.name}: {r.message}")
        elif r.is_warning:
            print_warn(f"{r.name}: {r.message}")
        else:
            print_fail(f"{r.name}: {r.message}")
            if r.fix_hint:
                console.print(f"    Fix: {r.fix_hint}")

    if has_failures:
        console.print()
        print_fail("Missing prerequisites. Install them and run [bold]molebie-ai install[/bold] again.")
        raise typer.Exit(1)
    console.print()

    # 2. Generate .env.local
    env_path = root / ".env.local"
    if env_path.exists():
        console.print(f"  [warn]![/warn] .env.local already exists.")
        choice = ask_choice(
            "What would you like to do?",
            [
                "Keep existing .env.local (skip generation)",
                "Regenerate .env.local (overwrites current)",
            ],
            default="Keep existing .env.local (skip generation)",
        )
        if "Regenerate" in choice:
            with console.status("[info]Generating .env.local...[/info]"):
                env_generator.generate_env_local(config, force=True)
            print_ok(".env.local regenerated")
        else:
            print_ok(".env.local kept as-is")
    else:
        with console.status("[info]Generating .env.local...[/info]"):
            env_generator.generate_env_local(config)
        print_ok(".env.local created")
    console.print()

    # 3. Install gateway dependencies
    with console.status("[info]Installing gateway dependencies...[/info]"):
        result = subprocess.run(
            ["pip3", "install", "-r", "requirements.txt", "--quiet"],
            cwd=str(root / "gateway"),
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode == 0:
        print_ok("Gateway dependencies installed")
    else:
        print_warn("Gateway dependency install had issues — check manually")
        if result.stderr:
            console.print(f"    [dim]{result.stderr[:200]}[/dim]")

    # 4. Install webapp dependencies
    with console.status("[info]Installing webapp dependencies...[/info]"):
        result = subprocess.run(
            ["npm", "install", "--silent"],
            cwd=str(root / "webapp"),
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode == 0:
        print_ok("Webapp dependencies installed")
    else:
        print_warn("Webapp dependency install had issues — check manually")
    console.print()

    # 5. Start Supabase and inject keys
    supabase_manager.setup_supabase()
    console.print()

    # 6. Start Docker services if needed
    docker_services = []
    if config.search_enabled:
        docker_services.append("searxng")
    if config.voice_enabled:
        docker_services.append("kokoro-tts")

    if docker_services:
        with console.status(f"[info]Starting Docker services: {', '.join(docker_services)}...[/info]"):
            result = subprocess.run(
                ["docker", "compose", "up", "-d"] + docker_services,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
            )
        if result.returncode == 0:
            print_ok(f"Docker services started: {', '.join(docker_services)}")
        else:
            print_warn("Docker services had issues — you can start them later with: docker compose up -d")

    # 7. Save config
    config.installed = True
    config.last_install_at = datetime.now(timezone.utc).isoformat()
    config_manager.save_config(config)
    print_ok("Configuration saved")

    # 8. Done
    console.print()
    console.print("[ok bold]Installation complete![/ok bold]")
    console.print()
    console.print("  Next steps:")
    console.print()

    is_two_machine = config.setup_type == SetupType.TWO_MACHINE
    step = 1

    if config.inference_backend == InferenceBackend.MLX:
        if is_two_machine:
            console.print(f"  {step}. On the [bold]GPU machine ({config.gpu_ip})[/bold]:")
            console.print("     [bold]make mlx-thinking[/bold]     # Required — Qwen 3.5 9B")
            console.print("     [bold]make mlx-instant[/bold]      # Optional — fast mode")
        else:
            console.print(f"  {step}. Start inference (in a new terminal):")
            console.print("     [bold]make mlx-thinking[/bold]     # Required — Qwen 3.5 9B")
            console.print("     [bold]make mlx-instant[/bold]      # Optional — fast mode")
        console.print()
        step += 1
        if is_two_machine:
            console.print(f"  {step}. On [bold]this machine ({config.server_ip})[/bold], start the app:")
        else:
            console.print(f"  {step}. Start the app:")
    elif config.inference_backend == InferenceBackend.OLLAMA:
        if is_two_machine:
            console.print(f"  {step}. On the [bold]GPU machine ({config.gpu_ip})[/bold]:")
            console.print("     [bold]ollama serve[/bold]")
        else:
            console.print(f"  {step}. Start Ollama:")
            console.print("     [bold]ollama serve[/bold]")
        console.print()
        step += 1
        console.print(f"  {step}. Start the app:")
    else:
        console.print(f"  {step}. Ensure your inference server is running")
        console.print()
        step += 1
        console.print(f"  {step}. Start the app:")

    console.print("     [bold]molebie-ai run[/bold]")
    console.print()
    step += 1

    if is_two_machine:
        console.print(f"  {step}. Open http://{config.server_ip}:3000 and create an account")
    else:
        console.print(f"  {step}. Open http://localhost:3000 and create an account")
    console.print()


def install() -> None:
    """Interactive setup wizard for Molebie AI."""
    print_banner()
    config = _wizard()
    _run_install(config)
