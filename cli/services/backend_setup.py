"""Backend auto-detection, recommendation, and setup for MLX/Ollama/OpenAI-compatible."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import httpx

from cli.models.config import InferenceBackend, SetupType
from cli.services.system_info import SystemInfo

# ──────────────────────────────────────────────────────────────
# Model catalog — single source of truth
# ──────────────────────────────────────────────────────────────

MLX_MODELS: dict[str, tuple[str, str]] = {
    "light": (
        "mlx-community/Qwen3.5-4B-4bit",
        "mlx-community/Qwen3.5-4B-4bit",
    ),
    "balanced": (
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        "mlx-community/Qwen3.5-4B-4bit",
    ),
}

OLLAMA_MODELS: dict[str, tuple[str, str]] = {
    "light": ("qwen3:4b", "qwen3:4b"),
    "balanced": ("qwen3:8b", "qwen3:4b"),
}

PROFILE_MIN_RAM: dict[str, int] = {
    "light": 8,
    "balanced": 16,
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "light": "Single small model, fast responses",
    "balanced": "Reasoning model + fast model",
}


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class BackendRecommendation:
    backend: InferenceBackend
    reason: str


@dataclass
class SetupResult:
    success: bool
    thinking_model: str = ""
    instant_model: str = ""
    message: str = ""
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Detection and recommendation
# ──────────────────────────────────────────────────────────────

def detect_recommended_backend(
    sys_info: SystemInfo, setup_type: SetupType
) -> BackendRecommendation:
    """Auto-detect the best backend for this system."""
    if setup_type == SetupType.TWO_MACHINE and not sys_info.is_apple_silicon:
        return BackendRecommendation(
            InferenceBackend.OPENAI_COMPATIBLE,
            "Two-machine setup — configure the remote endpoint URL",
        )
    if sys_info.is_apple_silicon:
        return BackendRecommendation(
            InferenceBackend.MLX,
            f"Apple Silicon detected ({sys_info.chip_name}) — native GPU acceleration",
        )
    if shutil.which("ollama"):
        return BackendRecommendation(
            InferenceBackend.OLLAMA,
            "Ollama is installed — good cross-platform choice",
        )
    return BackendRecommendation(
        InferenceBackend.OLLAMA,
        "Ollama recommended — easiest cross-platform setup",
    )


def recommend_model_profile(sys_info: SystemInfo) -> str:
    """Recommend a model profile based on available RAM."""
    if sys_info.total_memory_gb >= 12:
        return "balanced"
    return "light"


def get_models_for_profile(
    profile: str, backend: InferenceBackend
) -> tuple[str, str]:
    """Return (thinking_model, instant_model) for a profile and backend."""
    if backend == InferenceBackend.MLX:
        return MLX_MODELS.get(profile, MLX_MODELS["balanced"])
    elif backend == InferenceBackend.OLLAMA:
        return OLLAMA_MODELS.get(profile, OLLAMA_MODELS["balanced"])
    return ("", "")


# ──────────────────────────────────────────────────────────────
# MLX setup
# ──────────────────────────────────────────────────────────────

def is_mlx_vlm_installed() -> bool:
    """Check if mlx-vlm is installed (lightweight metadata check, no heavy import)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from importlib.metadata import version; version('mlx-vlm')"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def install_mlx_vlm() -> bool:
    """Install mlx-vlm using sys.executable -m pip. Matches Makefile mlx-vlm-install."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "mlx-vlm"],
        capture_output=False, timeout=600,
    )
    return result.returncode == 0


def is_mlx_model_cached(repo_id: str) -> bool:
    """Check if an MLX model is already in the huggingface cache."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"from huggingface_hub import scan_cache_dir; "
             f"repos = [r.repo_id for r in scan_cache_dir().repos]; "
             f"exit(0 if '{repo_id}' in repos else 1)"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def download_mlx_model(repo_id: str) -> bool:
    """Pre-download an MLX model via huggingface_hub.snapshot_download."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"from huggingface_hub import snapshot_download; snapshot_download('{repo_id}')"],
        capture_output=False, timeout=1800,  # 30 min max for large models
    )
    return result.returncode == 0


def setup_mlx(profile: str, sys_info: SystemInfo) -> SetupResult:
    """Set up MLX backend: install mlx-vlm, download models."""
    warnings: list[str] = []

    # 1. Verify Apple Silicon
    if not sys_info.is_apple_silicon:
        return SetupResult(
            success=False,
            message="MLX requires Apple Silicon (M1/M2/M3/M4). This machine is not compatible.",
        )

    # 2. Install mlx-vlm if needed
    if not is_mlx_vlm_installed():
        if not install_mlx_vlm():
            return SetupResult(
                success=False,
                message="Failed to install mlx-vlm. Try manually: python -m pip install -U mlx-vlm",
            )

    # 3. Get models for profile
    thinking, instant = get_models_for_profile(profile, InferenceBackend.MLX)

    # 4. Pre-download models (skip if already cached)
    models_to_download = [thinking]
    if instant != thinking:
        models_to_download.append(instant)

    for model in models_to_download:
        short_name = model.split("/")[-1] if "/" in model else model
        if is_mlx_model_cached(model):
            continue
        if not download_mlx_model(model):
            warnings.append(f"Could not pre-download {short_name} — it will download on first server start")

    return SetupResult(
        success=True,
        thinking_model=thinking,
        instant_model=instant,
        message=f"MLX ready (profile: {profile})",
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────
# Ollama setup
# ──────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Check if Ollama API is reachable."""
    try:
        resp = httpx.get("http://localhost:11434/api/version", timeout=3)
        return resp.status_code < 400
    except httpx.TransportError:
        return False


def start_ollama_daemon() -> bool:
    """Attempt to start ollama serve in the background."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False

    # Wait up to 15 seconds for it to come up
    for _ in range(15):
        time.sleep(1)
        if is_ollama_running():
            return True
    return False


def is_ollama_model_local(model: str) -> bool:
    """Check if an Ollama model is already pulled locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        # ollama list output: NAME ID SIZE MODIFIED
        # Match base name (e.g. "qwen3" matches "qwen3:4b")
        base = model.split(":")[0]
        return any(base in line for line in result.stdout.splitlines())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def pull_ollama_model(model: str) -> bool:
    """Pull an Ollama model. Lets native progress bars show through."""
    result = subprocess.run(
        ["ollama", "pull", model],
        capture_output=False, timeout=1800,
    )
    return result.returncode == 0


def setup_ollama(profile: str) -> SetupResult:
    """Set up Ollama backend: ensure running, pull models."""
    warnings: list[str] = []

    # 1. Check Ollama binary
    if not shutil.which("ollama"):
        return SetupResult(
            success=False,
            message="Ollama is not installed",
            warnings=["Install Ollama: brew install ollama  (or download from https://ollama.com)"],
        )

    # 2. Ensure Ollama is running
    if not is_ollama_running():
        if not start_ollama_daemon():
            warnings.append(
                "Could not start Ollama automatically — start it manually: ollama serve"
            )

    # 3. Get models for profile
    thinking, instant = get_models_for_profile(profile, InferenceBackend.OLLAMA)

    # 4. Pull models (only if Ollama is running, skip if already local)
    if is_ollama_running():
        models_to_pull = [thinking]
        if instant != thinking:
            models_to_pull.append(instant)

        for model in models_to_pull:
            if is_ollama_model_local(model):
                continue
            if not pull_ollama_model(model):
                warnings.append(f"Could not pull {model} — pull it manually: ollama pull {model}")
    else:
        warnings.append(
            f"Ollama not running — pull models later: ollama pull {thinking}"
        )

    # 5. Verify
    if is_ollama_running():
        try:
            httpx.get("http://localhost:11434/v1/models", timeout=3)
        except httpx.TransportError:
            warnings.append("Ollama is running but /v1/models endpoint not responding")

    return SetupResult(
        success=True,
        thinking_model=thinking,
        instant_model=instant,
        message=f"Ollama ready (profile: {profile})",
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────
# OpenAI-compatible validation
# ──────────────────────────────────────────────────────────────

def validate_openai_endpoint(
    url: str, api_key: str | None = None
) -> SetupResult:
    """Validate an OpenAI-compatible endpoint."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    models_url = f"{url.rstrip('/')}/v1/models"
    try:
        resp = httpx.get(models_url, headers=headers, timeout=5)
        if resp.status_code < 400:
            return SetupResult(
                success=True,
                message=f"Endpoint reachable at {url}",
            )
        return SetupResult(
            success=False,
            message=f"Endpoint returned {resp.status_code} at {models_url}",
            warnings=["Check that the server is running and the URL is correct"],
        )
    except httpx.TimeoutException:
        return SetupResult(
            success=False,
            message=f"Connection to {url} timed out",
            warnings=["The server may be starting up — try again shortly"],
        )
    except httpx.TransportError:
        return SetupResult(
            success=False,
            message=f"Could not connect to {url}",
            warnings=[
                "The endpoint may not be running yet — you can start it later",
                "Configuration has been saved and will work once the endpoint is available",
            ],
        )
