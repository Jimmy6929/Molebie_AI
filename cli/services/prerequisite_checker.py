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


def check_memory(total_gb: float, min_gb: float = 8.0) -> CheckResult:
    if total_gb <= 0:
        return CheckResult("Memory", True, "Could not detect", is_warning=True)
    if total_gb < min_gb:
        return CheckResult(
            "Memory", False, f"{total_gb} GB (need {min_gb}+ GB)",
            fix_hint="Close other applications or consider a lighter model profile",
            is_warning=True,
        )
    return CheckResult("Memory", True, f"{total_gb} GB")


def check_disk_space(available_gb: float, min_gb: float = 10.0) -> CheckResult:
    if available_gb <= 0:
        return CheckResult("Disk space", True, "Could not detect", is_warning=True)
    if available_gb < min_gb:
        return CheckResult(
            "Disk space", False, f"{available_gb} GB free (need {min_gb}+ GB)",
            fix_hint="Free up disk space before installing models",
            is_warning=True,
        )
    return CheckResult("Disk space", True, f"{available_gb} GB free")


def check_system_early() -> tuple[list[CheckResult], "SystemInfo"]:
    """Run lightweight system checks BEFORE the wizard. Returns (results, system_info)."""
    from cli.services.system_info import SystemInfo, get_system_info

    sys_info = get_system_info()
    results: list[CheckResult] = []

    # OS
    os_display = {"darwin": "macOS", "linux": "Linux", "windows": "Windows"}.get(
        sys_info.os, sys_info.os
    )
    results.append(CheckResult("OS", True, f"{os_display} ({sys_info.os})"))

    # Chip / architecture
    results.append(CheckResult("Chip", True, f"{sys_info.chip_name} ({sys_info.arch})"))

    # Memory
    results.append(check_memory(sys_info.total_memory_gb))

    # Disk space
    results.append(check_disk_space(sys_info.available_disk_gb))

    return results, sys_info


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
