"""molebie-ai feature — manage optional features."""

from __future__ import annotations

import subprocess

import typer
from rich.table import Table

from cli.models.config import FEATURE_DESCRIPTIONS, VALID_FEATURES
from cli.services import config_manager, env_generator, prerequisite_checker
from cli.services.config_manager import get_project_root
from cli.ui.console import console, print_fail, print_info, print_ok, print_warn

app = typer.Typer(no_args_is_help=True)


def _feature_field(feature: str) -> str:
    """Map feature name to config field name."""
    return f"{feature}_enabled"


# Mapping: feature name → (env_key, enabled_value, disabled_value)
_ENV_KEYS = {
    "search": ("WEB_SEARCH_ENABLED", "true", "false"),
    "rag": ("RAG_ENABLED", "true", "false"),
}

# Mapping: feature name → docker compose service name
_DOCKER_SERVICES = {
    "voice": "kokoro-tts",
    "search": "searxng",
}


def _docker_up(service: str) -> tuple[bool, str]:
    """Start a Docker Compose service. Returns (success, error_detail)."""
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d", service],
            cwd=str(get_project_root()),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return True, ""
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return False, detail
    except subprocess.TimeoutExpired:
        return False, "timed out after 120s"
    except FileNotFoundError:
        return False, "docker command not found"


def _docker_down(service: str) -> bool:
    """Stop a Docker Compose service. Returns True on success."""
    try:
        result = subprocess.run(
            ["docker", "compose", "stop", service],
            cwd=str(get_project_root()),
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@app.command(name="list")
def list_features() -> None:
    """Show all features and their current status."""
    console.print()

    if not config_manager.config_exists():
        print_warn("No config found. Run [bold]molebie-ai run[/bold] or [bold]molebie-ai install[/bold] first.")
        console.print()
        raise typer.Exit(0)

    config = config_manager.load_config()

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Feature", style="cyan", min_width=10)
    table.add_column("Status", min_width=10)
    table.add_column("Description")

    for name in VALID_FEATURES:
        enabled = getattr(config, _feature_field(name), False)
        status = "[green]enabled[/green]" if enabled else "[dim]disabled[/dim]"
        desc = FEATURE_DESCRIPTIONS.get(name, "")
        table.add_row(name, status, desc)

    console.print(table)
    console.print()


@app.command(name="add")
def add_feature(feature: str = typer.Argument(help="Feature to enable (voice, search, rag)")) -> None:
    """Enable an optional feature."""
    feature = feature.lower()
    if feature not in VALID_FEATURES:
        print_fail(f"Unknown feature: {feature}")
        console.print(f"  Available: {', '.join(VALID_FEATURES)}")
        raise typer.Exit(1)

    if not config_manager.config_exists():
        print_fail("No config found. Run [bold]molebie-ai run[/bold] first to auto-configure.")
        raise typer.Exit(1)

    config = config_manager.load_config()
    field_name = _feature_field(feature)

    if getattr(config, field_name):
        print_ok(f"{feature} is already enabled")
        return

    # Resolve prerequisites before enabling Docker-dependent features
    if feature in _DOCKER_SERVICES:
        print_info("Checking Docker...")
        ok, msg = prerequisite_checker.resolve_docker(log=lambda m: print_info(m))
        if not ok:
            print_fail(f"Docker: {msg}")
            print_info(f"Install Docker first, then run: molebie-ai feature add {feature}")
            raise typer.Exit(1)

    # Voice-specific: check ffmpeg (needed for speaker verification)
    if feature == "voice":
        ffmpeg_check = prerequisite_checker.check_ffmpeg()
        if not ffmpeg_check.passed:
            pkg_mgr = prerequisite_checker.detect_package_manager()
            if pkg_mgr:
                res = prerequisite_checker.install_prereq(
                    prerequisite_checker.FFMPEG_PREREQ, pkg_mgr,
                )
                if res.success:
                    print_ok("ffmpeg installed")
                else:
                    print_warn(f"ffmpeg: could not auto-install — {ffmpeg_check.fix_hint}")
            else:
                print_warn(f"ffmpeg: {ffmpeg_check.fix_hint}")

    setattr(config, field_name, True)
    config_manager.save_config(config)

    # Update .env.local
    if feature in _ENV_KEYS:
        key, enabled_val, _ = _ENV_KEYS[feature]
        if not env_generator.update_env_key(key, enabled_val):
            print_warn(f"Could not update .env.local — run [bold]molebie-ai config init[/bold] to regenerate")

    print_ok(f"{feature} enabled")

    # Auto-start Docker service
    if feature in _DOCKER_SERVICES:
        svc_name = _DOCKER_SERVICES[feature]
        print_info(f"Starting {svc_name}...")
        ok, err = _docker_up(svc_name)
        if ok:
            print_ok(f"{svc_name} started")
            if feature == "voice":
                print_info("Kokoro TTS model (~1GB) may be downloading — first request may take a few minutes")
                print_info("Whisper STT model (~75MB) will download on first voice request")
        else:
            print_fail(f"{svc_name} failed to start: {err}")
            print_info(f"Try manually: docker compose up -d {svc_name}")
            # Rollback: don't leave broken config
            setattr(config, field_name, False)
            config_manager.save_config(config)
            print_info(f"{feature} disabled due to startup failure")
            raise typer.Exit(1)
    elif feature == "rag":
        print_info("Embedding model will download automatically on first use")

    console.print()


@app.command(name="remove")
def remove_feature(feature: str = typer.Argument(help="Feature to disable (voice, search, rag)")) -> None:
    """Disable an optional feature."""
    feature = feature.lower()
    if feature not in VALID_FEATURES:
        print_fail(f"Unknown feature: {feature}")
        console.print(f"  Available: {', '.join(VALID_FEATURES)}")
        raise typer.Exit(1)

    if not config_manager.config_exists():
        print_fail("No config found. Run [bold]molebie-ai run[/bold] first to auto-configure.")
        raise typer.Exit(1)

    config = config_manager.load_config()
    field_name = _feature_field(feature)

    if not getattr(config, field_name):
        print_ok(f"{feature} is already disabled")
        return

    setattr(config, field_name, False)
    config_manager.save_config(config)

    # Update .env.local
    if feature in _ENV_KEYS:
        key, _, disabled_val = _ENV_KEYS[feature]
        if not env_generator.update_env_key(key, disabled_val):
            print_warn(f"Could not update .env.local — run [bold]molebie-ai config init[/bold] to regenerate")

    print_ok(f"{feature} disabled")

    # Stop Docker service
    if feature in _DOCKER_SERVICES:
        svc_name = _DOCKER_SERVICES[feature]
        print_info(f"Stopping {svc_name}...")
        if _docker_down(svc_name):
            print_ok(f"{svc_name} stopped")
        else:
            print_warn(f"{svc_name} failed to stop — run manually: docker compose stop {svc_name}")

    console.print()
