"""Check and install system prerequisites for Molebie AI."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fix_hint: str = ""
    is_warning: bool = False


@dataclass
class InstallablePrereq:
    """A prerequisite that can be auto-installed."""
    name: str
    install_cmds: dict[str, list[str]]  # package_manager -> command parts
    post_install_msg: str = ""
    is_required: bool = True


@dataclass
class InstallResult:
    name: str
    success: bool
    message: str


def _run_quiet(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a command and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


# ──────────────────────────────────────────────────────────────
# OS / package manager detection
# ──────────────────────────────────────────────────────────────

def detect_os() -> str:
    """Return 'darwin' or 'linux'."""
    return platform.system().lower()


def detect_package_manager() -> str | None:
    """Detect the available package manager. Returns name or None."""
    if shutil.which("brew"):
        return "brew"
    if shutil.which("apt-get"):
        return "apt"
    if shutil.which("dnf"):
        return "dnf"
    if shutil.which("pacman"):
        return "pacman"
    return None


# ──────────────────────────────────────────────────────────────
# Install command registry
# ──────────────────────────────────────────────────────────────

DOCKER_PREREQ = InstallablePrereq(
    name="Docker",
    install_cmds={
        "brew": ["brew", "install", "--cask", "docker"],
        "apt": ["sudo", "sh", "-c",
                "curl -fsSL https://get.docker.com | sh"],
        "dnf": ["sudo", "dnf", "install", "-y", "docker-ce"],
        "pacman": ["sudo", "pacman", "-S", "--noconfirm", "docker"],
    },
    post_install_msg="Please open Docker Desktop to finish setup, then press Enter.",
)

NODE_PREREQ = InstallablePrereq(
    name="Node.js",
    install_cmds={
        "brew": ["brew", "install", "node"],
        "apt": ["sudo", "sh", "-c",
                "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - "
                "&& apt-get install -y nodejs"],
        "dnf": ["sudo", "dnf", "install", "-y", "nodejs"],
        "pacman": ["sudo", "pacman", "-S", "--noconfirm", "nodejs", "npm"],
    },
)

SUPABASE_PREREQ = InstallablePrereq(
    name="Supabase CLI",
    install_cmds={
        "brew": ["brew", "install", "supabase/tap/supabase"],
        "apt": ["npm", "i", "-g", "supabase"],
        "dnf": ["npm", "i", "-g", "supabase"],
        "pacman": ["npm", "i", "-g", "supabase"],
    },
)

FFMPEG_PREREQ = InstallablePrereq(
    name="ffmpeg",
    install_cmds={
        "brew": ["brew", "install", "ffmpeg"],
        "apt": ["sudo", "apt-get", "install", "-y", "ffmpeg"],
        "dnf": ["sudo", "dnf", "install", "-y", "ffmpeg"],
        "pacman": ["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"],
    },
    is_required=False,
)


# ──────────────────────────────────────────────────────────────
# Individual check functions
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Composite checks
# ──────────────────────────────────────────────────────────────

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


def check_all(
    voice_enabled: bool = False,
    search_enabled: bool = False,
) -> list[CheckResult]:
    """Run all prerequisite checks."""
    docker_required = voice_enabled or search_enabled
    docker_result = check_docker()
    if not docker_required and not docker_result.passed:
        docker_result.is_warning = True
    results = [
        docker_result,
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


# ──────────────────────────────────────────────────────────────
# Auto-installation of missing prerequisites
# ──────────────────────────────────────────────────────────────

def find_missing_prereqs() -> list[InstallablePrereq]:
    """Return the list of installable prerequisites that are missing."""
    missing: list[InstallablePrereq] = []

    # Docker (binary or daemon)
    if not shutil.which("docker"):
        missing.append(DOCKER_PREREQ)
    else:
        rc, _ = _run_quiet(["docker", "info"])
        if rc != 0:
            # Docker installed but daemon not running — handle separately
            pass

    # Node.js 18+
    node_ok = False
    if shutil.which("node"):
        rc, out = _run_quiet(["node", "-v"])
        if rc == 0:
            try:
                major = int(out.lstrip("v").split(".")[0])
                node_ok = major >= 18
            except (ValueError, IndexError):
                pass
    if not node_ok:
        missing.append(NODE_PREREQ)

    # Supabase CLI
    if not shutil.which("supabase"):
        missing.append(SUPABASE_PREREQ)

    # ffmpeg (optional)
    if not shutil.which("ffmpeg"):
        missing.append(FFMPEG_PREREQ)

    return missing


def get_install_command_display(prereq: InstallablePrereq, pkg_mgr: str) -> str:
    """Return a human-readable install command string for display."""
    cmds = prereq.install_cmds.get(pkg_mgr)
    if not cmds:
        return "(no auto-install available)"
    return " ".join(cmds)


def install_prereq(prereq: InstallablePrereq, pkg_mgr: str) -> InstallResult:
    """Install a single prerequisite. Returns result."""
    cmds = prereq.install_cmds.get(pkg_mgr)
    if not cmds:
        return InstallResult(
            prereq.name, False,
            f"No install command for package manager '{pkg_mgr}'",
        )

    try:
        result = subprocess.run(
            cmds,
            timeout=600,
            capture_output=False,
        )
        if result.returncode == 0:
            return InstallResult(prereq.name, True, "Installed successfully")
        return InstallResult(
            prereq.name, False,
            f"Install command exited with code {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return InstallResult(prereq.name, False, "Installation timed out")
    except FileNotFoundError:
        return InstallResult(
            prereq.name, False,
            f"Command not found: {cmds[0]}",
        )


def wait_for_docker_daemon(timeout_seconds: int = 120) -> bool:
    """Wait for Docker daemon to become responsive. Returns True if it responds."""
    for _ in range(timeout_seconds // 3):
        rc, _ = _run_quiet(["docker", "info"], timeout=5)
        if rc == 0:
            return True
        time.sleep(3)
    return False


def start_docker_daemon(timeout_seconds: int = 120) -> bool:
    """Attempt to start Docker daemon automatically. Returns True if daemon becomes responsive."""
    # Check if already running
    rc, _ = _run_quiet(["docker", "info"], timeout=5)
    if rc == 0:
        return True

    os_name = detect_os()
    if os_name == "darwin":
        # Launch Docker Desktop on macOS
        subprocess.Popen(
            ["open", "-a", "Docker"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif os_name == "linux":
        # Try systemctl to start the docker service
        subprocess.run(
            ["sudo", "systemctl", "start", "docker"],
            capture_output=True,
            timeout=30,
        )

    return wait_for_docker_daemon(timeout_seconds=timeout_seconds)
