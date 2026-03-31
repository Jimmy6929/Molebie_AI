"""Real feature provisioning: start Docker services, verify health, pre-download models."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from cli.services import prerequisite_checker


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
        except (httpx.ConnectError, httpx.TimeoutException):
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


def setup_rag() -> FeatureSetupResult:
    """Prepare RAG prerequisites and optionally pre-download the embedding model."""
    warnings: list[str] = []

    # Attempt to pre-download embedding model
    result = subprocess.run(
        [sys.executable, "-c",
         "from sentence_transformers import SentenceTransformer; "
         "SentenceTransformer('Orange/orange-nomic-v1.5-1536')"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        msg = "RAG prerequisites ready (embedding model cached)"
    else:
        msg = "RAG configuration saved"
        warnings.append(
            "Embedding model (~500MB) will download on first use"
        )

    # Note: full RAG readiness depends on gateway running
    warnings.append(
        "Document upload and retrieval are gateway services — start with: molebie-ai run"
    )

    return FeatureSetupResult(
        feature="rag", success=True, message=msg, warnings=warnings,
    )
