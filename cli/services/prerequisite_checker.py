"""Check and install system prerequisites for Molebie AI."""

from __future__ import annotations

import enum
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


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


class DockerState(enum.Enum):
    """Possible Docker installation states."""
    READY = "ready"                       # CLI in PATH + daemon running
    DAEMON_STOPPED = "daemon_stopped"     # CLI in PATH + daemon not running
    APP_EXISTS_NO_CLI = "app_no_cli"      # Docker.app exists but CLI not in PATH (macOS)
    NOT_INSTALLED = "not_installed"        # Nothing found


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
# Docker state detection & resolution
# ──────────────────────────────────────────────────────────────

_DOCKER_APP_PATHS = [
    Path("/Applications/Docker.app"),
    Path.home() / "Applications" / "Docker.app",
]


def detect_docker_state() -> DockerState:
    """Detect the current Docker installation state."""
    if shutil.which("docker"):
        rc, _ = _run_quiet(["docker", "info"])
        return DockerState.READY if rc == 0 else DockerState.DAEMON_STOPPED
    # macOS: Docker Desktop may be installed but CLI not yet symlinked
    if detect_os() == "darwin":
        for app_path in _DOCKER_APP_PATHS:
            if app_path.exists():
                return DockerState.APP_EXISTS_NO_CLI
    return DockerState.NOT_INSTALLED


def _wait_for_docker_cli(timeout_seconds: int = 30) -> bool:
    """Wait for the ``docker`` CLI to appear in PATH (e.g. after Docker Desktop launch)."""
    for _ in range(timeout_seconds // 2):
        if shutil.which("docker"):
            return True
        time.sleep(2)
    return False


def _check_compose_plugin() -> tuple[bool, str]:
    """Verify the Docker Compose plugin is available."""
    rc, _ = _run_quiet(["docker", "compose", "version"])
    if rc != 0:
        return False, (
            "Docker Compose plugin not found — "
            "install from https://docs.docker.com/compose/install/"
        )
    return True, ""


def resolve_docker(log: "Callable[[str], None] | None" = None) -> tuple[bool, str]:
    """Attempt to bring Docker to the READY state.

    Unlike ``check_docker()`` (read-only diagnostic), this function actively
    fixes each intermediate state: launching the app, starting the daemon, or
    installing Docker from scratch.  Also verifies the Docker Compose plugin
    is available (required for feature services).

    *log* is an optional callback for progress messages (e.g. ``console.print``).

    Returns ``(success, message)``.
    """
    state = detect_docker_state()

    if state == DockerState.READY:
        compose_ok, compose_msg = _check_compose_plugin()
        if not compose_ok:
            return False, compose_msg
        return True, "Docker is ready"

    if state == DockerState.APP_EXISTS_NO_CLI:
        if log:
            log("Docker Desktop found but CLI not in PATH — launching app...")
        subprocess.Popen(
            ["open", "-a", "Docker"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if _wait_for_docker_cli() and wait_for_docker_daemon():
            compose_ok, compose_msg = _check_compose_plugin()
            if not compose_ok:
                return False, compose_msg
            return True, "Docker Desktop launched and ready"
        return (
            False,
            "Docker Desktop launched but CLI not available — try opening Docker Desktop manually",
        )

    if state == DockerState.DAEMON_STOPPED:
        if log:
            log("Docker CLI found — starting daemon...")
        if start_docker_daemon():
            compose_ok, compose_msg = _check_compose_plugin()
            if not compose_ok:
                return False, compose_msg
            return True, "Docker daemon started"
        return False, "Could not start Docker daemon — open Docker Desktop manually"

    # NOT_INSTALLED — attempt auto-install
    pkg_mgr = detect_package_manager()
    if not pkg_mgr:
        return False, "Docker not installed. Install from https://docker.com"

    if log:
        log("Installing Docker...")

    # On macOS brew, use --adopt to handle a leftover Docker.app from a
    # direct download that wasn't fully cleaned up.
    if pkg_mgr == "brew":
        rc, _ = _run_quiet(
            ["brew", "install", "--cask", "--adopt", "docker"], timeout=600,
        )
        success = rc == 0
    else:
        res = install_prereq(DOCKER_PREREQ, pkg_mgr)
        success = res.success

    if not success:
        return False, "Docker installation failed — install manually from https://docker.com"

    if start_docker_daemon():
        compose_ok, compose_msg = _check_compose_plugin()
        if not compose_ok:
            return False, compose_msg
        return True, "Docker installed and running"
    return False, "Docker installed but daemon not starting — open Docker Desktop manually"


def check_docker() -> CheckResult:
    """Read-only Docker status check (used by ``doctor`` and display functions)."""
    state = detect_docker_state()
    if state == DockerState.READY:
        return CheckResult("Docker", True, "Installed and running")
    if state == DockerState.DAEMON_STOPPED:
        return CheckResult(
            "Docker daemon", False, "Not running",
            fix_hint="Open Docker Desktop or run: open -a Docker",
        )
    if state == DockerState.APP_EXISTS_NO_CLI:
        return CheckResult(
            "Docker", False, "App installed but CLI not in PATH",
            fix_hint="Open Docker Desktop once to create CLI symlinks, or run: open -a Docker",
        )
    return CheckResult(
        "Docker", False, "Not installed",
        fix_hint="Install from https://docker.com or: brew install --cask docker",
    )


# ──────────────────────────────────────────────────────────────
# Individual check functions
# ──────────────────────────────────────────────────────────────


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
    core_only: bool = False,
) -> list[CheckResult]:
    """Run prerequisite checks.

    When *core_only* is True only Python and Node are checked — Docker and
    ffmpeg are skipped entirely (they are resolved at point-of-need by feature
    setup functions).
    """
    results: list[CheckResult] = [
        check_node(),
        check_python(),
    ]

    if core_only:
        return results

    # Docker: hard requirement only when features need it
    docker_required = voice_enabled or search_enabled
    docker_result = check_docker()
    if not docker_required and not docker_result.passed:
        docker_result.is_warning = True
    results.insert(0, docker_result)

    # ffmpeg
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
