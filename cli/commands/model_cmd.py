"""molebie-ai model — manage LLM models (download, remove, start/stop)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import typer
from rich.table import Table

from cli.models.config import InferenceBackend
from cli.services import config_manager
from cli.services.backend_setup import (
    MLX_MODELS,
    OLLAMA_MODELS,
    download_mlx_model,
    is_ollama_running,
    pull_ollama_model,
    start_ollama_daemon,
)
from cli.services.config_manager import get_project_root
from cli.services.env_generator import update_env_key
from cli.services.service_manager import ServiceRunner, _check_health, _wait_healthy
from cli.ui.console import console, print_fail, print_info, print_ok, print_warn

app = typer.Typer(no_args_is_help=True)

# ──────────────────────────────────────────────────────────────
# Alias mapping: friendly name → (mlx_model, ollama_model)
# ──────────────────────────────────────────────────────────────

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "4b": ("mlx-community/Qwen3.5-4B-4bit", "qwen3:4b"),
    "9b": ("mlx-community/Qwen3.5-9B-MLX-4bit", "qwen3:8b"),
}

_TIER_PORTS: dict[str, int] = {"thinking": 8080, "instant": 8081}

_VALID_TIERS = ("thinking", "instant", "all")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _resolve_model(alias_or_name: str, backend: InferenceBackend) -> str:
    """Resolve a friendly alias (4b, 9b) to a backend-specific model name."""
    key = alias_or_name.lower()
    if key in MODEL_ALIASES:
        mlx_name, ollama_name = MODEL_ALIASES[key]
        if backend == InferenceBackend.MLX:
            return mlx_name
        return ollama_name
    return alias_or_name


def _get_alias(model_name: str) -> str:
    """Return the alias for a model name, or empty string."""
    for alias, (mlx, ollama) in MODEL_ALIASES.items():
        if model_name in (mlx, ollama):
            return alias
    return ""


def _collect_catalog_models(backend: InferenceBackend) -> list[str]:
    """Return all unique model names from the catalog for a backend."""
    catalog = MLX_MODELS if backend == InferenceBackend.MLX else OLLAMA_MODELS
    seen: set[str] = set()
    models: list[str] = []
    for thinking, instant in catalog.values():
        for m in (thinking, instant):
            if m not in seen:
                seen.add(m)
                models.append(m)
    return models


def _get_downloaded_mlx_models() -> set[str]:
    """Check which MLX (HuggingFace) models are cached locally."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "from huggingface_hub import scan_cache_dir; "
             "info = scan_cache_dir(); "
             "print('\\n'.join(r.repo_id for r in info.repos))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return set(result.stdout.strip().splitlines())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return set()


def _get_downloaded_ollama_models() -> set[str]:
    """Check which Ollama models are available locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            models: set[str] = set()
            for line in result.stdout.strip().splitlines()[1:]:  # skip header
                parts = line.split()
                if parts:
                    models.add(parts[0])
            return models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return set()


def _remove_mlx_model(repo_id: str) -> bool:
    """Remove an MLX model from the HuggingFace cache."""
    result = subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import scan_cache_dir; "
         f"info = scan_cache_dir(); "
         f"repos = [r for r in info.repos if r.repo_id == '{repo_id}']; "
         f"assert repos, 'not found'; "
         f"hashes = [rev.commit_hash for rev in repos[0].revisions]; "
         f"strategy = info.delete_revisions(*hashes); "
         f"strategy.execute()"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def _remove_ollama_model(model: str) -> bool:
    """Remove an Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "rm", model],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _kill_on_port(port: int) -> bool:
    """Kill any process on the given port. Returns True if a process was killed."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        killed = False
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGTERM)
                killed = True
        return killed
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False


def _is_port_in_use(port: int) -> bool:
    """Check if a process is listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ──────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────

@app.command(name="list")
def list_models() -> None:
    """Show available models, download status, and active tier."""
    console.print()

    if not config_manager.config_exists():
        print_warn("No config found. Run [bold]molebie-ai run[/bold] or [bold]molebie-ai install[/bold] first.")
        console.print()
        raise typer.Exit(0)

    config = config_manager.load_config()
    backend = config.inference_backend

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Model", style="cyan", min_width=20)
    table.add_column("Alias", min_width=6)
    table.add_column("Downloaded", min_width=12)
    table.add_column("Active Tier", min_width=12)

    if backend == InferenceBackend.OPENAI_COMPATIBLE:
        # Show configured models as remote
        for label, model in [("thinking", config.thinking_model), ("instant", config.instant_model)]:
            if model:
                table.add_row(model, _get_alias(model), "[dim]remote[/dim]", label)
        console.print(table)
        console.print()
        return

    # Get downloaded models
    if backend == InferenceBackend.MLX:
        downloaded = _get_downloaded_mlx_models()
    else:
        downloaded = _get_downloaded_ollama_models()

    catalog_models = _collect_catalog_models(backend)

    for model in catalog_models:
        alias = _get_alias(model)
        is_downloaded = model in downloaded
        dl_status = "[green]yes[/green]" if is_downloaded else "[dim]no[/dim]"

        tiers: list[str] = []
        if config.thinking_model == model:
            tiers.append("thinking")
        if config.instant_model == model:
            tiers.append("instant")
        tier_label = ", ".join(tiers) if tiers else "[dim]—[/dim]"

        table.add_row(model, alias, dl_status, tier_label)

    console.print(table)
    console.print()
    console.print(f"[dim]Backend: {backend.value}[/dim]")
    console.print()


@app.command(name="add")
def add_model(
    model: str = typer.Argument(help="Model alias (4b, 9b) or full model name"),
) -> None:
    """Download / pull a model so it's available locally."""
    console.print()

    if not config_manager.config_exists():
        print_fail("No config found. Run [bold]molebie-ai run[/bold] first to auto-configure.")
        raise typer.Exit(1)

    config = config_manager.load_config()
    backend = config.inference_backend

    if backend == InferenceBackend.OPENAI_COMPATIBLE:
        print_info("Remote models are managed by the external provider — no download needed.")
        console.print()
        return

    resolved = _resolve_model(model, backend)

    if backend == InferenceBackend.MLX:
        short = resolved.split("/")[-1] if "/" in resolved else resolved
        print_info(f"Downloading {short}...")
        if download_mlx_model(resolved):
            print_ok(f"{short} downloaded successfully")
        else:
            print_fail(f"Failed to download {short}")
            console.print(f"  Try manually: python -c \"from huggingface_hub import snapshot_download; snapshot_download('{resolved}')\"")
            raise typer.Exit(1)

    elif backend == InferenceBackend.OLLAMA:
        if not shutil.which("ollama"):
            print_fail("Ollama is not installed")
            console.print("  Install: brew install ollama  (or https://ollama.com)")
            raise typer.Exit(1)
        if not is_ollama_running():
            print_info("Starting Ollama daemon...")
            if not start_ollama_daemon():
                print_fail("Could not start Ollama — start manually: ollama serve")
                raise typer.Exit(1)
        print_info(f"Pulling {resolved}...")
        if pull_ollama_model(resolved):
            print_ok(f"{resolved} pulled successfully")
        else:
            print_fail(f"Failed to pull {resolved}")
            console.print(f"  Try manually: ollama pull {resolved}")
            raise typer.Exit(1)

    console.print()


@app.command(name="remove")
def remove_model(
    model: str = typer.Argument(help="Model alias (4b, 9b) or full model name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
) -> None:
    """Remove a downloaded model from disk."""
    console.print()

    if not config_manager.config_exists():
        print_fail("No config found. Run [bold]molebie-ai run[/bold] first to auto-configure.")
        raise typer.Exit(1)

    config = config_manager.load_config()
    backend = config.inference_backend

    if backend == InferenceBackend.OPENAI_COMPATIBLE:
        print_info("Remote models are managed by the external provider.")
        console.print()
        return

    resolved = _resolve_model(model, backend)

    # Warn if this is the active model
    is_active_thinking = config.thinking_model == resolved
    is_active_instant = config.instant_model == resolved
    if is_active_thinking or is_active_instant:
        active_tiers = []
        if is_active_thinking:
            active_tiers.append("thinking")
        if is_active_instant:
            active_tiers.append("instant")
        print_warn(f"{resolved} is the active {' + '.join(active_tiers)} model")
        if not force:
            from cli.ui.prompts import ask_confirm
            if not ask_confirm("Remove the active model?"):
                print_info("Cancelled")
                console.print()
                return

    if backend == InferenceBackend.MLX:
        short = resolved.split("/")[-1] if "/" in resolved else resolved
        print_info(f"Removing {short} from cache...")
        if _remove_mlx_model(resolved):
            print_ok(f"{short} removed")
        else:
            print_fail(f"Could not remove {short} — it may not be cached")
            raise typer.Exit(1)

    elif backend == InferenceBackend.OLLAMA:
        if not shutil.which("ollama"):
            print_fail("Ollama is not installed")
            raise typer.Exit(1)
        print_info(f"Removing {resolved}...")
        if _remove_ollama_model(resolved):
            print_ok(f"{resolved} removed")
        else:
            print_fail(f"Could not remove {resolved}")
            console.print(f"  Try manually: ollama rm {resolved}")
            raise typer.Exit(1)

    # Clear from config if it was active
    changed = False
    if is_active_thinking:
        config.thinking_model = None
        update_env_key("INFERENCE_THINKING_MODEL", "")
        changed = True
    if is_active_instant:
        config.instant_model = None
        update_env_key("INFERENCE_INSTANT_MODEL", "")
        changed = True
    if changed:
        config_manager.save_config(config)
        print_warn("Active model removed. Set a new model with: molebie-ai config profile <profile>")

    console.print()


@app.command(name="start")
def start_model(
    tier: str = typer.Option("all", "--tier", "-t", help="Tier to start: thinking, instant, or all"),
) -> None:
    """Start inference server(s) in the background."""
    console.print()

    tier = tier.lower()
    if tier not in _VALID_TIERS:
        print_fail(f"Invalid tier: {tier}")
        console.print(f"  Available: {', '.join(_VALID_TIERS)}")
        raise typer.Exit(1)

    if not config_manager.config_exists():
        print_fail("No config found. Run [bold]molebie-ai run[/bold] first to auto-configure.")
        raise typer.Exit(1)

    config = config_manager.load_config()
    backend = config.inference_backend

    if backend == InferenceBackend.OPENAI_COMPATIBLE:
        print_info("Remote inference endpoints are managed externally.")
        console.print()
        return

    if backend == InferenceBackend.OLLAMA:
        if not shutil.which("ollama"):
            print_fail("Ollama is not installed")
            raise typer.Exit(1)
        if is_ollama_running():
            print_ok("Ollama daemon is already running")
        else:
            print_info("Starting Ollama daemon...")
            if start_ollama_daemon():
                print_ok("Ollama daemon started — models load on first request")
            else:
                print_fail("Could not start Ollama — start manually: ollama serve")
                raise typer.Exit(1)
        console.print()
        return

    # MLX: start individual servers
    root = get_project_root()
    tiers_to_start = list(_TIER_PORTS.keys()) if tier == "all" else [tier]

    for t in tiers_to_start:
        port = _TIER_PORTS[t]
        health_url = f"http://localhost:{port}/v1/models"

        # Already running?
        if _check_health(health_url, timeout=2.0):
            print_ok(f"MLX {t.capitalize()} already running on :{port}")
            continue

        # Free stale process
        ServiceRunner._free_port(port, f"MLX {t.capitalize()}")

        # Start server
        start_cmd = [
            "python3", str(root / "scripts" / "mlx_server.py"),
            "--host", "0.0.0.0", "--port", str(port),
        ]
        log_dir = root / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"mlx-{t}.log"

        try:
            log_file = open(log_path, "w")
            subprocess.Popen(
                start_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            print_fail(f"Could not start MLX {t} server — python3 not found")
            continue

        with console.status(f"[info]Waiting for MLX {t.capitalize()} to become healthy...[/info]"):
            healthy = _wait_healthy(health_url, f"MLX {t.capitalize()}", max_wait=30)

        if healthy:
            print_ok(f"MLX {t.capitalize()} started on :{port}")
        else:
            print_warn(f"MLX {t.capitalize()} started but not yet healthy — model loads on first request")
            console.print(f"  [dim]Log: {log_path}[/dim]")

    console.print()


@app.command(name="stop")
def stop_model(
    tier: str = typer.Option("all", "--tier", "-t", help="Tier to stop: thinking, instant, or all"),
) -> None:
    """Stop running inference server(s)."""
    console.print()

    tier = tier.lower()
    if tier not in _VALID_TIERS:
        print_fail(f"Invalid tier: {tier}")
        console.print(f"  Available: {', '.join(_VALID_TIERS)}")
        raise typer.Exit(1)

    if not config_manager.config_exists():
        print_fail("No config found. Run [bold]molebie-ai run[/bold] first to auto-configure.")
        raise typer.Exit(1)

    config = config_manager.load_config()
    backend = config.inference_backend

    if backend == InferenceBackend.OPENAI_COMPATIBLE:
        print_info("Remote inference endpoints are managed externally.")
        console.print()
        return

    if backend == InferenceBackend.OLLAMA:
        print_info("Ollama is a shared daemon serving all models.")
        console.print("  To stop: [bold]pkill ollama[/bold]")
        console.print()
        return

    # MLX: stop by killing process on port
    tiers_to_stop = list(_TIER_PORTS.keys()) if tier == "all" else [tier]

    for t in tiers_to_stop:
        port = _TIER_PORTS[t]
        if not _is_port_in_use(port):
            print_info(f"No MLX {t.capitalize()} server running on :{port}")
            continue

        if _kill_on_port(port):
            print_ok(f"MLX {t.capitalize()} stopped (:{port})")
        else:
            print_fail(f"Could not stop process on :{port}")
            console.print(f"  Try manually: lsof -ti :{port} | xargs kill")

    console.print()
