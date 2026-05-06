"""Real feature provisioning: start Docker services, verify health, pre-download models."""

from __future__ import annotations

import secrets
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from cli.services import prerequisite_checker

_SEARXNG_SECRET_PLACEHOLDER = "CHANGE_ME_GENERATED_ON_FIRST_RUN"


def _ensure_searxng_secret_key(project_root: Path) -> None:
    """Replace the committed placeholder in searxng/settings.yml with a per-install token."""
    settings = project_root / "searxng" / "settings.yml"
    if not settings.exists():
        return
    text = settings.read_text(encoding="utf-8")
    if _SEARXNG_SECRET_PLACEHOLDER not in text:
        return
    settings.write_text(
        text.replace(_SEARXNG_SECRET_PLACEHOLDER, secrets.token_hex(32)),
        encoding="utf-8",
    )


@dataclass
class FeatureSetupResult:
    feature: str
    success: bool
    message: str
    warnings: list[str] = field(default_factory=list)


def _wait_healthy(url: str, timeout: int = 30) -> bool:
    """Poll a health endpoint until reachable or timeout."""
    for _ in range(timeout):
        try:
            resp = httpx.get(url, timeout=3, follow_redirects=True)
            if resp.status_code < 400:
                return True
        except httpx.TransportError:
            pass
        time.sleep(1)
    return False


def setup_search(project_root: Path) -> FeatureSetupResult:
    """Start SearXNG Docker container and verify health."""
    warnings: list[str] = []

    # Resolve Docker — actively fixes intermediate states instead of just checking
    ok, msg = prerequisite_checker.resolve_docker()
    if not ok:
        return FeatureSetupResult(
            feature="search",
            success=False,
            message=f"Docker is required for web search (SearXNG): {msg}",
        )

    # Generate a unique secret_key on first install so two self-hosters don't
    # share session-signing material from the committed placeholder.
    _ensure_searxng_secret_key(project_root)

    # Remove stale container if it exists (prevents name-conflict errors on re-install)
    subprocess.run(
        ["docker", "rm", "-f", "searxng"],
        capture_output=True, text=True, timeout=30,
    )

    # Start SearXNG — same command as existing install.py and Makefile
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "searxng"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return FeatureSetupResult(
            feature="search",
            success=False,
            message=f"Failed to start SearXNG container: {detail}",
            warnings=["Try manually: docker compose up -d searxng"],
        )

    # Wait for health
    if _wait_healthy("http://localhost:8888/", timeout=30):
        return FeatureSetupResult(
            feature="search", success=True,
            message="SearXNG running on :8888",
        )

    warnings.append("SearXNG started but not yet healthy — may need more time")
    return FeatureSetupResult(
        feature="search", success=True,
        message="SearXNG container started",
        warnings=warnings,
    )


def setup_voice(project_root: Path) -> FeatureSetupResult:
    """Start Kokoro TTS Docker container, check ffmpeg, verify health."""
    warnings: list[str] = []

    # ffmpeg: try to auto-install, warn if still missing
    ffmpeg_check = prerequisite_checker.check_ffmpeg()
    if not ffmpeg_check.passed:
        pkg_mgr = prerequisite_checker.detect_package_manager()
        if pkg_mgr:
            res = prerequisite_checker.install_prereq(
                prerequisite_checker.FFMPEG_PREREQ, pkg_mgr,
            )
            if not res.success:
                warnings.append(f"ffmpeg: could not auto-install — {ffmpeg_check.fix_hint}")
        else:
            warnings.append(f"ffmpeg: {ffmpeg_check.message} — {ffmpeg_check.fix_hint}")

    # Resolve Docker — actively fixes intermediate states
    ok, msg = prerequisite_checker.resolve_docker()
    if not ok:
        return FeatureSetupResult(
            feature="voice",
            success=False,
            message=f"Docker is required for voice (Kokoro TTS): {msg}",
        )

    # Remove stale container if it exists (prevents name-conflict errors on re-install)
    subprocess.run(
        ["docker", "rm", "-f", "kokoro-tts"],
        capture_output=True, text=True, timeout=30,
    )

    # Start Kokoro TTS — same command as existing install.py and Makefile
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "kokoro-tts"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return FeatureSetupResult(
            feature="voice",
            success=False,
            message=f"Failed to start Kokoro TTS container: {detail}",
            warnings=warnings + ["Try manually: docker compose up -d kokoro-tts"],
        )

    # Wait for health — first run downloads ~1GB model, so long timeout
    if _wait_healthy("http://localhost:8880/health", timeout=120):
        msg = "Kokoro TTS running on :8880"
    else:
        msg = "Kokoro TTS container started (may still be downloading model)"
        warnings.append("First run downloads ~1GB model — give it a few minutes")

    # Note about Whisper
    warnings.append("Whisper STT model (~75MB) will download on first voice request")

    return FeatureSetupResult(
        feature="voice", success=True, message=msg, warnings=warnings,
    )


def _is_model_cached(model: str) -> bool:
    """Check if a HuggingFace model is already in the local cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HuggingFace cache uses models--{org}--{name} directory format
    model_dir_name = f"models--{model.replace('/', '--')}"
    model_path = cache_dir / model_dir_name
    if not model_path.is_dir():
        return False
    # Verify it has actual snapshot data (not just a partial download)
    snapshots = model_path / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def ensure_embedding_model(model: str) -> FeatureSetupResult:
    """Ensure the shared embedding model is downloaded and cached.

    This is a shared dependency used by both the memory service and RAG.
    It should run during install whenever *either* feature is enabled.

    Strategy:
      1. Check local cache first (instant, no network).
      2. If cached → done. Offline install works fine.
      3. If not cached → try to download. If offline, fail clearly.
    """
    # Step 1: Check local cache (no network, no ML imports needed)
    if _is_model_cached(model):
        return FeatureSetupResult(
            feature="embedding", success=True,
            message=f"Embedding model already cached ({model})",
        )

    # Not cached — need ML deps + internet to download
    check = subprocess.run(
        [sys.executable, "-c", "import torch; import sentence_transformers"],
        capture_output=True, text=True, timeout=30,
    )
    if check.returncode != 0:
        return FeatureSetupResult(
            feature="embedding",
            success=False,
            message="ML dependencies (torch, sentence-transformers) not installed",
            warnings=[
                "Memory and RAG features need these to work.",
                "Install with: pip install -r requirements-ml.txt",
            ],
        )

    # Try to download (timeout=120s is plenty for the initial connection;
    # if we're offline, SentenceTransformer fails within ~10s on connect)
    result = subprocess.run(
        [sys.executable, "-c",
         "from sentence_transformers import SentenceTransformer; "
         f"SentenceTransformer('{model}')"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        return FeatureSetupResult(
            feature="embedding", success=True,
            message=f"Embedding model downloaded and cached ({model})",
        )

    # Download failed — likely offline
    return FeatureSetupResult(
        feature="embedding",
        success=False,
        message="Embedding model not cached and download failed (no internet?)",
        warnings=[
            "Memory and RAG will be unavailable until the model is cached.",
            "Connect to internet and run: molebie-ai install --quick",
        ],
    )


def setup_rag() -> FeatureSetupResult:
    """Prepare RAG prerequisites (embedding model is handled separately)."""
    warnings: list[str] = []

    # Quick check: are ML dependencies (torch + sentence-transformers) available?
    check = subprocess.run(
        [sys.executable, "-c", "import torch; import sentence_transformers"],
        capture_output=True, text=True, timeout=30,
    )
    if check.returncode != 0:
        return FeatureSetupResult(
            feature="rag",
            success=False,
            message="RAG unavailable — ML dependencies (torch, sentence-transformers) not installed",
            warnings=[
                "This typically means PyTorch has no wheels for your platform/Python version.",
                "Core chat, web search, and auth work normally without it.",
            ],
        )

    # Note: full RAG readiness depends on gateway running
    warnings.append(
        "Document upload and retrieval are gateway services — start with: molebie-ai run"
    )

    return FeatureSetupResult(
        feature="rag", success=True, message="RAG prerequisites ready",
        warnings=warnings,
    )
