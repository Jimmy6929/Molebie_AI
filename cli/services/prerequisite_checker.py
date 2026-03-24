"""Check system prerequisites for Molebie AI."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fix_hint: str = ""
    is_warning: bool = False


def _run_quiet(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a command and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def check_docker() -> CheckResult:
    if not shutil.which("docker"):
        return CheckResult(
            "Docker", False, "Not installed",
            fix_hint="Install from https://docker.com or: brew install --cask docker",
        )
    rc, _ = _run_quiet(["docker", "info"])
    if rc != 0:
        return CheckResult(
            "Docker daemon", False, "Not running",
            fix_hint="Open Docker Desktop first",
        )
    return CheckResult("Docker", True, "Installed and running")


def check_node() -> CheckResult:
    if not shutil.which("node"):
        return CheckResult(
            "Node.js", False, "Not installed",
            fix_hint="brew install node",
        )
    rc, out = _run_quiet(["node", "-v"])
    if rc != 0:
        return CheckResult("Node.js", False, "Could not determine version")
    try:
        major = int(out.lstrip("v").split(".")[0])
    except (ValueError, IndexError):
        return CheckResult("Node.js", False, f"Unexpected version: {out}")
    if major < 18:
        return CheckResult(
            "Node.js", False, f"v{major} found, need 18+",
            fix_hint="brew install node",
        )
    return CheckResult("Node.js", True, out)


def check_python() -> CheckResult:
    if not shutil.which("python3"):
        return CheckResult(
            "Python 3", False, "Not installed",
            fix_hint="brew install python",
        )
    rc, out = _run_quiet(["python3", "-c", "import sys; print(sys.version_info.minor)"])
    if rc != 0:
        return CheckResult("Python 3", False, "Could not determine version")
    try:
        minor = int(out.strip())
    except ValueError:
        return CheckResult("Python 3", False, f"Unexpected version output: {out}")
    if minor < 10:
        return CheckResult(
            "Python 3", False, f"3.{minor} found, need 3.10+",
            fix_hint="brew install python",
        )
    return CheckResult("Python 3", True, f"3.{minor}")


def check_supabase_cli() -> CheckResult:
    if not shutil.which("supabase"):
        return CheckResult(
            "Supabase CLI", False, "Not installed",
            fix_hint="brew install supabase/tap/supabase",
        )
    rc, out = _run_quiet(["supabase", "--version"])
    version = out if rc == 0 else "unknown"
    return CheckResult("Supabase CLI", True, version)


def check_ffmpeg() -> CheckResult:
    if not shutil.which("ffmpeg"):
        return CheckResult(
            "ffmpeg", False, "Not installed (voice features won't work)",
            fix_hint="brew install ffmpeg",
            is_warning=True,
        )
    return CheckResult("ffmpeg", True, "Installed")


def check_all(voice_enabled: bool = False) -> list[CheckResult]:
    """Run all prerequisite checks."""
    results = [
        check_docker(),
        check_node(),
        check_python(),
        check_supabase_cli(),
    ]
    if voice_enabled:
        results.append(check_ffmpeg())
    else:
        ffmpeg_result = check_ffmpeg()
        ffmpeg_result.is_warning = True
        results.append(ffmpeg_result)
    return results
