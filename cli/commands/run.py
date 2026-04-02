"""molebie-ai run — start all configured services."""

from __future__ import annotations

from typing import Optional

import httpx
import typer

from cli.models.config import InferenceBackend, MolebieConfig, ModelProfile, SetupType
from cli.services import config_manager
from cli.services.service_manager import ServiceRunner
from cli.ui.console import console, print_banner, print_fail, print_info, print_ok, print_warn


def _ensure_ready() -> MolebieConfig:
    """Ensure config, .env.local, and data/ exist. Auto-create if missing.

    Returns a valid MolebieConfig ready for service startup.
    """
    from cli.services import backend_setup, env_generator
    from cli.services.system_info import get_system_info

    root = config_manager.get_project_root()
    needs_save = False

    # --- 1. Config ---
    if config_manager.config_exists():
        config = config_manager.load_config()
    else:
        console.print("[heading]First run — auto-configuring...[/heading]")
        console.print()
        config = MolebieConfig()
        sys_info = get_system_info(root)

        # Detect best backend
        rec = backend_setup.detect_recommended_backend(sys_info, config.run_inference)
        config.inference_backend = rec.backend
        print_ok(f"Backend: {rec.backend.value} ({rec.reason})")

        # Pick model profile
        if config.inference_backend != InferenceBackend.OPENAI_COMPATIBLE:
            profile = backend_setup.recommend_model_profile(sys_info)
            config.model_profile = ModelProfile(profile)
            thinking, instant = backend_setup.get_models_for_profile(
                profile, config.inference_backend,
            )
            config.thinking_model = thinking
            config.instant_model = instant
            print_ok(f"Models: {profile} profile")

        config.search_enabled = True
        config.rag_enabled = True
        config.voice_enabled = False
        config.installed = True
        needs_save = True

    # Mark installed if not already (handles partial-install case)
    if not config.installed:
        config.installed = True
        needs_save = True

    # Save config if we created or modified it
    if needs_save:
        config_manager.ensure_config_dir()
        config_manager.save_config(config)
        print_ok(f"Config saved")

    # --- 2. .env.local ---
    env_path = root / ".env.local"
    if not env_path.exists():
        print_info("Generating .env.local...")
        try:
            env_generator.generate_env_local(config)
        except Exception:
            env_generator.init_env_local()
        print_ok(".env.local created")

    # --- 3. data/ directory ---
    data_dir = root / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print_ok("Data directory created")

    # --- 4. Embedding model (shared by memory + RAG) ---
    # Memory is always on, so the embedding model is always needed.
    # Provision it here so `molebie-ai run` works even without prior install.
    if needs_save:
        from cli.services import feature_setup
        model = config_manager.read_embedding_model()
        r = feature_setup.ensure_embedding_model(model=model)
        if r.success:
            print_ok(r.message)
        else:
            print_warn(r.message)
            print_info("Core chat still works — memory and RAG need this model.")
        for w in r.warnings:
            print_warn(w)

    if needs_save:
        console.print()

    return config


def _check_remote_services(config) -> None:
    """For distributed setups, verify remote services are reachable before starting."""
    if config.setup_type != SetupType.DISTRIBUTED:
        return

    endpoints: list[tuple[str, str]] = []

    # Check remote inference
    if not config.run_inference:
        host = config.inference_host
        if config.inference_backend == InferenceBackend.MLX:
            endpoints.append(("MLX Thinking", f"http://{host}:8080/v1/models"))
            endpoints.append(("MLX Instant", f"http://{host}:8081/v1/models"))
        elif config.inference_backend == InferenceBackend.OLLAMA:
            endpoints.append(("Ollama", f"http://{host}:11434/v1/models"))
        elif config.inference_backend == InferenceBackend.OPENAI_COMPATIBLE and config.inference_url:
            endpoints.append(("Inference", f"{config.inference_url}/v1/models"))

    # Check remote gateway
    if not config.run_gateway:
        endpoints.append(("Gateway", f"http://{config.gateway_host}:8000/health"))

    # Check remote webapp
    if not config.run_webapp:
        endpoints.append(("Webapp", f"http://{config.webapp_host}:3000"))

    if not endpoints:
        return

    console.print("[heading]Checking remote services...[/heading]")
    all_ok = True
    for name, url in endpoints:
        try:
            resp = httpx.get(url, timeout=5.0, follow_redirects=True)
            if resp.status_code < 400:
                print_ok(f"{name} reachable at {url}")
            else:
                print_warn(f"{name} returned {resp.status_code} at {url}")
                all_ok = False
        except httpx.TimeoutException:
            print_warn(f"{name} timed out at {url}")
            all_ok = False
        except httpx.TransportError:
            print_fail(f"{name} not reachable at {url}")
            all_ok = False

    if not all_ok:
        console.print()
        console.print("[warn]Some remote services are not reachable.[/warn]")
        console.print("  Make sure the services are running on their respective machines.")
        console.print()
    else:
        console.print()


def run(
    service: Optional[str] = typer.Option(
        None, "--service", "-s",
        help="Start only a specific service (e.g., gateway, webapp)",
    ),
    no_inference: bool = typer.Option(
        False, "--no-inference",
        help="Skip starting inference servers (MLX/Ollama)",
    ),
) -> None:
    """Start all configured services."""
    print_banner()

    config = _ensure_ready()

    _check_remote_services(config)

    runner = ServiceRunner()
    runner.start_services(config, service_filter=service, skip_inference=no_inference)
