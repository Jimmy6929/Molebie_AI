"""molebie-ai install — interactive setup wizard with real backend/feature setup."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

import typer
from rich.panel import Panel
from rich.table import Table

from cli.models.config import (
    InferenceBackend,
    MolebieConfig,
    ModelProfile,
    SetupType,
)
from cli.services import (
    backend_setup,
    config_manager,
    env_generator,
    feature_setup,
    prerequisite_checker,
    supabase_manager,
)
from cli.services.system_info import SystemInfo
from cli.ui.console import (
    console,
    print_banner,
    print_fail,
    print_info,
    print_ok,
    print_step_header,
    print_warn,
)
from cli.ui.prompts import (
    ask_choice,
    ask_confirm,
    ask_features,
    ask_model_profile,
    ask_text,
)


# ──────────────────────────────────────────────────────────────
# Auto-configure defaults for --quick mode
# ──────────────────────────────────────────────────────────────

def _auto_configure(config: MolebieConfig, sys_info: SystemInfo) -> None:
    """Set sensible defaults without prompting."""
    config.setup_type = SetupType.SINGLE

    rec = backend_setup.detect_recommended_backend(sys_info, config.setup_type)
    config.inference_backend = rec.backend
    print_ok(f"Backend: {rec.backend.value} — {rec.reason}")

    if config.inference_backend != InferenceBackend.OPENAI_COMPATIBLE:
        profile = backend_setup.recommend_model_profile(sys_info)
        config.model_profile = ModelProfile(profile)
        thinking, instant = backend_setup.get_models_for_profile(
            profile, config.inference_backend,
        )
        config.thinking_model = thinking
        config.instant_model = instant
        print_ok(f"Model profile: {profile}")

    config.search_enabled = True
    config.rag_enabled = True
    config.voice_enabled = False
    print_ok("Features: search + RAG enabled, voice disabled")


# ──────────────────────────────────────────────────────────────
# Phase 0: Welcome
# ──────────────────────────────────────────────────────────────

def _show_welcome() -> None:
    print_banner()
    console.print(Panel(
        "[bold]This installer will:[/bold]\n\n"
        "  1. Check your system and install missing tools\n"
        "  2. Configure your setup mode\n"
        "  3. Set up your inference backend\n"
        "  4. Choose AI models\n"
        "  5. Configure optional features\n"
        "  6. Install dependencies and start services",
        title="Molebie AI Installer",
        expand=False,
        style="blue",
    ))
    if not ask_confirm("Ready to begin?", default=True):
        raise typer.Exit(0)


# ──────────────────────────────────────────────────────────────
# Phase 1: System check + auto-install
# ──────────────────────────────────────────────────────────────

def _check_system(quick: bool = False) -> SystemInfo:
    print_step_header(1, 8, "System Check")
    results, sys_info = prerequisite_checker.check_system_early()

    for r in results:
        if r.passed:
            print_ok(f"{r.name}: {r.message}")
        elif r.is_warning:
            print_warn(f"{r.name}: {r.message}")
            if r.fix_hint:
                console.print(f"    {r.fix_hint}")
        else:
            print_fail(f"{r.name}: {r.message}")
            if r.fix_hint:
                console.print(f"    {r.fix_hint}")

    # Hard blocker: dangerously low resources
    if sys_info.total_memory_gb > 0 and sys_info.total_memory_gb < 4:
        console.print()
        print_fail("Less than 4 GB RAM detected — cannot run AI models.")
        raise typer.Exit(1)
    if sys_info.available_disk_gb > 0 and sys_info.available_disk_gb < 5:
        console.print()
        print_fail("Less than 5 GB free disk space — not enough for models and dependencies.")
        raise typer.Exit(1)

    return sys_info


# ──────────────────────────────────────────────────────────────
# Phase 2: Setup type
# ──────────────────────────────────────────────────────────────

def _ask_setup_type(config: MolebieConfig) -> None:
    print_step_header(2, 8, "Setup Type")
    choice = ask_choice(
        "How will you run Molebie AI?",
        [
            "Single machine — everything on this Mac",
            "Two machines — GPU on a separate Mac (Tailscale/LAN)",
        ],
        default="Single machine — everything on this Mac",
    )

    if "Two" in choice:
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


# ──────────────────────────────────────────────────────────────
# Phase 3: Backend selection
# ──────────────────────────────────────────────────────────────

def _ask_backend(config: MolebieConfig, sys_info: SystemInfo) -> None:
    print_step_header(3, 8, "Inference Backend")

    rec = backend_setup.detect_recommended_backend(sys_info, config.setup_type)
    print_info(f"Recommendation: {rec.backend.value} — {rec.reason}")
    console.print()

    if ask_confirm(f"Use {rec.backend.value}?", default=True):
        config.inference_backend = rec.backend
    else:
        choice = ask_choice(
            "Choose your inference backend:",
            [
                "MLX (Apple Silicon)",
                "Ollama (cross-platform)",
                "OpenAI-compatible (vLLM, llama.cpp, OpenAI API, etc.)",
            ],
        )
        if "MLX" in choice:
            config.inference_backend = InferenceBackend.MLX
        elif "Ollama" in choice:
            config.inference_backend = InferenceBackend.OLLAMA
        else:
            config.inference_backend = InferenceBackend.OPENAI_COMPATIBLE

    # OpenAI-compatible: ask for endpoint details
    if config.inference_backend == InferenceBackend.OPENAI_COMPATIBLE:
        console.print()
        config.inference_url = ask_text("  Endpoint URL", default="http://localhost:8080")
        api_key = ask_text("  API key (empty if none)", default="")
        config.inference_api_key = api_key if api_key else None
        config.thinking_model = ask_text("  Thinking model name (optional)", default="")
        config.instant_model = ask_text("  Instant model name (optional)", default="")
        config.thinking_model = config.thinking_model or None
        config.instant_model = config.instant_model or None
        config.model_profile = ModelProfile.CUSTOM

    print_ok(f"Backend: {config.inference_backend.value}")


# ──────────────────────────────────────────────────────────────
# Phase 4: Model profile
# ──────────────────────────────────────────────────────────────

def _ask_model_profile(config: MolebieConfig, sys_info: SystemInfo) -> None:
    if config.inference_backend == InferenceBackend.OPENAI_COMPATIBLE:
        return  # Models already specified in phase 3

    print_step_header(4, 8, "Model Profile")

    recommended = backend_setup.recommend_model_profile(sys_info)
    profile = ask_model_profile(
        recommended, sys_info.total_memory_gb, config.inference_backend.value,
    )

    config.model_profile = ModelProfile(profile)
    thinking, instant = backend_setup.get_models_for_profile(
        profile, config.inference_backend,
    )
    config.thinking_model = thinking
    config.instant_model = instant
    print_ok(f"Profile: {profile}")


# ──────────────────────────────────────────────────────────────
# Phase 5: Features
# ──────────────────────────────────────────────────────────────

def _ask_features(config: MolebieConfig) -> None:
    print_step_header(5, 8, "Optional Features")
    features = ask_features({
        "search": config.search_enabled,
        "rag": config.rag_enabled,
        "voice": config.voice_enabled,
    })
    config.search_enabled = features.get("search", True)
    config.rag_enabled = features.get("rag", True)
    config.voice_enabled = features.get("voice", False)


# ──────────────────────────────────────────────────────────────
# Phase 6: Review
# ──────────────────────────────────────────────────────────────

def _show_review(config: MolebieConfig) -> None:
    print_step_header(6, 8, "Review")

    table = Table(show_header=False, expand=False, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Setup type", config.setup_type.value)
    if config.setup_type == SetupType.TWO_MACHINE:
        table.add_row("GPU IP", config.gpu_ip)
        table.add_row("Server IP", config.server_ip)
    table.add_row("Backend", config.inference_backend.value)
    table.add_row("Model profile", config.model_profile.value)
    if config.thinking_model:
        table.add_row("Thinking model", config.thinking_model)
    if config.instant_model:
        table.add_row("Instant model", config.instant_model)
    if config.inference_url:
        table.add_row("Endpoint URL", config.inference_url)
    table.add_row("Search", "enabled" if config.search_enabled else "disabled")
    table.add_row("RAG", "enabled" if config.rag_enabled else "disabled")
    table.add_row("Voice", "enabled" if config.voice_enabled else "disabled")

    console.print(table)
    console.print()

    if not ask_confirm("Proceed with installation?", default=True):
        print_info("Installation cancelled.")
        raise typer.Exit(0)


# ──────────────────────────────────────────────────────────────
# Phase 7: Execute install
# ──────────────────────────────────────────────────────────────

def _display_check_results(results: list) -> bool:
    """Display check results. Returns True if any hard failure."""
    has_failure = False
    for r in results:
        if r.passed:
            print_ok(f"{r.name}: {r.message}")
        elif r.is_warning:
            print_warn(f"{r.name}: {r.message}")
        else:
            print_fail(f"{r.name}: {r.message}")
            if r.fix_hint:
                console.print(f"    Fix: {r.fix_hint}")
            has_failure = True
    return has_failure


def _install_gateway_deps(root) -> None:
    """Install gateway Python dependencies."""
    with console.status("[info]Installing gateway dependencies...[/info]"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
            cwd=str(root / "gateway"),
            capture_output=True, text=True, timeout=300,
        )
    if result.returncode == 0:
        print_ok("Gateway dependencies installed")
    else:
        print_warn("Gateway dependency install had issues — check manually")
        if result.stderr:
            console.print(f"    [dim]{result.stderr[:200]}[/dim]")


def _install_webapp_deps(root) -> None:
    """Install webapp Node.js dependencies."""
    with console.status("[info]Installing webapp dependencies...[/info]"):
        result = subprocess.run(
            ["npm", "install", "--silent"],
            cwd=str(root / "webapp"),
            capture_output=True, text=True, timeout=300,
        )
    if result.returncode == 0:
        print_ok("Webapp dependencies installed")
    else:
        print_warn("Webapp dependency install had issues — check manually")


def _setup_backend(config: MolebieConfig, sys_info: SystemInfo) -> None:
    """Run real backend setup: install packages, download models."""
    is_two_machine = config.setup_type == SetupType.TWO_MACHINE

    if is_two_machine:
        print_info(f"Two-machine mode: inference runs on {config.gpu_ip}")
        print_info("Backend setup is skipped — you need to set up the GPU machine yourself.")
        console.print()
        if config.inference_backend == InferenceBackend.MLX:
            console.print(f"    On the GPU machine ({config.gpu_ip}), run:")
            console.print("      [bold]python -m pip install -U mlx-vlm[/bold]")
            console.print("      [bold]make mlx-thinking[/bold]    # port 8080")
            console.print("      [bold]make mlx-instant[/bold]     # port 8081 (optional)")
        elif config.inference_backend == InferenceBackend.OLLAMA:
            console.print(f"    On the GPU machine ({config.gpu_ip}), run:")
            console.print("      [bold]ollama serve[/bold]")
            if config.thinking_model:
                console.print(f"      [bold]ollama pull {config.thinking_model}[/bold]")
            if config.instant_model and config.instant_model != config.thinking_model:
                console.print(f"      [bold]ollama pull {config.instant_model}[/bold]")
        else:
            console.print(f"    Ensure your inference server is running on {config.gpu_ip}")
        console.print()
        return

    # Single-machine: do real setup
    if config.inference_backend == InferenceBackend.MLX:
        print_info("Setting up MLX backend...")
        result = backend_setup.setup_mlx(config.model_profile.value, sys_info)
    elif config.inference_backend == InferenceBackend.OLLAMA:
        print_info("Setting up Ollama backend...")
        result = backend_setup.setup_ollama(config.model_profile.value)
    else:
        print_info("Validating endpoint...")
        result = backend_setup.validate_openai_endpoint(
            config.inference_url or "", config.inference_api_key,
        )

    if result.success:
        print_ok(result.message)
    else:
        print_warn(result.message)
    for w in result.warnings:
        print_warn(w)


def _setup_features(config: MolebieConfig, root) -> None:
    """Run real feature setup: Docker services, health checks, model downloads."""
    if config.search_enabled:
        print_info("Setting up web search...")
        r = feature_setup.setup_search(root)
        print_ok(r.message) if r.success else print_warn(r.message)
        for w in r.warnings:
            print_warn(w)

    if config.voice_enabled:
        print_info("Setting up voice...")
        r = feature_setup.setup_voice(root)
        print_ok(r.message) if r.success else print_warn(r.message)
        for w in r.warnings:
            print_warn(w)

    if config.rag_enabled:
        print_info("Preparing RAG prerequisites...")
        r = feature_setup.setup_rag()
        print_ok(r.message) if r.success else print_warn(r.message)
        for w in r.warnings:
            print_warn(w)


def _resolve_prerequisites(config: MolebieConfig, quick: bool = False) -> None:
    """Check prerequisites. If any are missing, offer to install or let user do it."""
    console.print("[heading]Checking prerequisites...[/heading]")
    results = prerequisite_checker.check_all(
        voice_enabled=config.voice_enabled,
        search_enabled=config.search_enabled,
    )
    _display_check_results(results)

    failures = [r for r in results if not r.passed and not r.is_warning]
    if not failures:
        return

    console.print()
    pkg_mgr = prerequisite_checker.detect_package_manager()

    if quick:
        user_choice = "Install missing tools for me"
    else:
        user_choice = ask_choice(
            "Missing prerequisites detected. What would you like to do?",
            [
                "Install missing tools for me",
                "I'll install them myself",
            ],
            default="Install missing tools for me",
        )

    if "myself" in user_choice:
        # Show the commands the user needs to run
        console.print()
        console.print("[heading]Run these commands to fix the issues:[/heading]")
        for r in failures:
            name_lower = r.name.lower()
            if "docker" in name_lower and "daemon" in name_lower:
                if prerequisite_checker.detect_os() == "darwin":
                    console.print("  Docker daemon: [bold]open -a Docker[/bold]")
                else:
                    console.print("  Docker daemon: [bold]sudo systemctl start docker[/bold]")
            elif "docker" in name_lower:
                if pkg_mgr:
                    cmd = prerequisite_checker.get_install_command_display(
                        prerequisite_checker.DOCKER_PREREQ, pkg_mgr,
                    )
                    console.print(f"  Docker: [bold]{cmd}[/bold]")
                else:
                    console.print("  Docker: Install from https://docker.com")
            elif "node" in name_lower:
                if pkg_mgr:
                    cmd = prerequisite_checker.get_install_command_display(
                        prerequisite_checker.NODE_PREREQ, pkg_mgr,
                    )
                    console.print(f"  {r.name}: [bold]{cmd}[/bold]")
                else:
                    console.print(f"  {r.name}: {r.fix_hint}")
            elif "supabase" in name_lower:
                if pkg_mgr:
                    cmd = prerequisite_checker.get_install_command_display(
                        prerequisite_checker.SUPABASE_PREREQ, pkg_mgr,
                    )
                    console.print(f"  {r.name}: [bold]{cmd}[/bold]")
                else:
                    console.print(f"  {r.name}: {r.fix_hint}")
            elif "ffmpeg" in name_lower:
                if pkg_mgr:
                    cmd = prerequisite_checker.get_install_command_display(
                        prerequisite_checker.FFMPEG_PREREQ, pkg_mgr,
                    )
                    console.print(f"  {r.name}: [bold]{cmd}[/bold]")
                else:
                    console.print(f"  {r.name}: {r.fix_hint}")
            elif "python" in name_lower:
                console.print(f"  {r.name}: {r.fix_hint}")
        console.print()
        print_info("After installing, run [bold]molebie-ai install[/bold] again.")
        raise typer.Exit(1)

    # User chose "Install for me"
    if not pkg_mgr:
        print_fail("No package manager found (brew, apt, dnf, pacman). Please install tools manually.")
        raise typer.Exit(1)

    console.print()
    for r in failures:
        name_lower = r.name.lower()
        if "docker" in name_lower and ("daemon" in name_lower or "running" in r.message.lower()):
            print_info("Starting Docker...")
            if prerequisite_checker.start_docker_daemon():
                print_ok("Docker daemon started")
            else:
                print_warn("Could not start Docker daemon")
        elif "docker" in name_lower:
            print_info("Installing Docker...")
            res = prerequisite_checker.install_prereq(prerequisite_checker.DOCKER_PREREQ, pkg_mgr)
            if res.success:
                print_ok("Docker installed")
                print_info("Starting Docker...")
                if prerequisite_checker.start_docker_daemon():
                    print_ok("Docker daemon started")
                else:
                    print_warn("Could not start Docker daemon")
            else:
                print_warn(f"Docker: {res.message}")
        elif "node" in name_lower:
            print_info("Installing Node.js...")
            res = prerequisite_checker.install_prereq(prerequisite_checker.NODE_PREREQ, pkg_mgr)
            print_ok("Node.js installed") if res.success else print_warn(f"Node.js: {res.message}")
        elif "supabase" in name_lower:
            print_info("Installing Supabase CLI...")
            res = prerequisite_checker.install_prereq(prerequisite_checker.SUPABASE_PREREQ, pkg_mgr)
            print_ok("Supabase CLI installed") if res.success else print_warn(f"Supabase CLI: {res.message}")
        elif "ffmpeg" in name_lower:
            print_info("Installing ffmpeg...")
            res = prerequisite_checker.install_prereq(prerequisite_checker.FFMPEG_PREREQ, pkg_mgr)
            print_ok("ffmpeg installed") if res.success else print_warn(f"ffmpeg: {res.message}")
        elif "python" in name_lower:
            print_warn(f"Python: {r.message} — please fix manually")

    # Re-check after auto-fix
    console.print()
    print_info("Re-checking prerequisites...")
    results = prerequisite_checker.check_all(
        voice_enabled=config.voice_enabled,
        search_enabled=config.search_enabled,
    )
    if _display_check_results(results):
        console.print()
        print_fail("Some prerequisites still missing. Fix them and run [bold]molebie-ai install[/bold] again.")
        raise typer.Exit(1)


def _execute_install(
    config: MolebieConfig, sys_info: SystemInfo, quick: bool = False,
) -> None:
    """Execute all installation steps."""
    root = config_manager.get_project_root()
    print_step_header(7, 8, "Installing")

    # 7a. Check prerequisites — offer to install if missing
    _resolve_prerequisites(config, quick=quick)
    console.print()

    # 7b. Generate .env.local
    console.print("[heading]Generating environment...[/heading]")
    env_path = root / ".env.local"
    if env_path.exists():
        if quick:
            print_ok(".env.local already exists — keeping as-is")
        else:
            console.print("  .env.local already exists.")
            choice = ask_choice(
                "What would you like to do?",
                [
                    "Keep existing .env.local (skip generation)",
                    "Regenerate .env.local (overwrites current)",
                ],
                default="Keep existing .env.local (skip generation)",
            )
            if "Regenerate" in choice:
                env_generator.generate_env_local(config, force=True)
                print_ok(".env.local regenerated")
            else:
                print_ok(".env.local kept as-is")
    else:
        env_generator.generate_env_local(config)
        print_ok(".env.local created")
    console.print()

    # 7c. Install gateway dependencies
    console.print("[heading]Installing dependencies...[/heading]")
    _install_gateway_deps(root)

    # 7d. Install webapp dependencies
    _install_webapp_deps(root)
    console.print()

    # 7e. Ensure Docker is running before Supabase
    docker_check = prerequisite_checker.check_docker()
    if not docker_check.passed:
        console.print("[heading]Starting Docker...[/heading]")
        if prerequisite_checker.start_docker_daemon(timeout_seconds=120):
            print_ok("Docker daemon started")
        else:
            print_warn("Could not start Docker — Supabase may fail")
        console.print()

    # 7f. Start Supabase + inject keys
    console.print("[heading]Starting Supabase...[/heading]")
    supabase_manager.setup_supabase()
    console.print()

    # 7f. Backend auto-setup
    console.print("[heading]Setting up inference backend...[/heading]")
    _setup_backend(config, sys_info)
    console.print()

    # 7g. Feature auto-setup
    if config.search_enabled or config.voice_enabled or config.rag_enabled:
        console.print("[heading]Setting up features...[/heading]")
        _setup_features(config, root)
        console.print()

    # 7h. Final save
    config.installed = True
    config.last_install_at = datetime.now(timezone.utc).isoformat()
    config_manager.save_config(config)
    print_ok("Configuration saved")


# ──────────────────────────────────────────────────────────────
# Phase 8: Summary + auto-start
# ──────────────────────────────────────────────────────────────

def _show_summary(config: MolebieConfig) -> None:
    print_step_header(8, 8, "Complete")

    table = Table(title="Installation Summary", show_header=False, expand=False, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Setup", config.setup_type.value)
    if config.setup_type == SetupType.TWO_MACHINE:
        table.add_row("GPU machine", config.gpu_ip)
        table.add_row("Server machine", config.server_ip)
    table.add_row("Backend", config.inference_backend.value)
    table.add_row("Model profile", config.model_profile.value)
    if config.thinking_model:
        table.add_row("Thinking model", config.thinking_model)
    if config.instant_model:
        table.add_row("Instant model", config.instant_model)
    table.add_row("Search", "enabled" if config.search_enabled else "disabled")
    table.add_row("RAG", "enabled" if config.rag_enabled else "disabled")
    table.add_row("Voice", "enabled" if config.voice_enabled else "disabled")
    table.add_row("Config", str(config_manager.get_config_path()))

    console.print(table)
    console.print()

    if ask_confirm("Start Molebie AI now?", default=True):
        console.print()
        from cli.services.service_manager import ServiceRunner
        runner = ServiceRunner()
        runner.start_services(config)
    else:
        console.print()
        console.print("  Run [bold]molebie-ai run[/bold] when you're ready to start.")
        console.print()


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def install(
    quick: bool = typer.Option(
        False, "--quick", "-q",
        help="Auto-select defaults without prompting (non-interactive mode).",
    ),
) -> None:
    """Interactive setup wizard for Molebie AI."""
    if not quick:
        _show_welcome()

    sys_info = _check_system(quick=quick)

    config = MolebieConfig()

    if quick:
        _auto_configure(config, sys_info)
    else:
        _ask_setup_type(config)
        _ask_backend(config, sys_info)
        _ask_model_profile(config, sys_info)
        _ask_features(config)
        _show_review(config)

    # Preliminary save — preserves wizard choices if install fails partway
    config_manager.save_config(config)

    _execute_install(config, sys_info, quick=quick)

    if quick:
        console.print()
        print_ok("Installation complete. Run: molebie-ai run")
        console.print("  Then open [bold]http://localhost:3000[/bold]")
    else:
        _show_summary(config)
