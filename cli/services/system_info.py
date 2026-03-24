"""Detect system information: OS, architecture, memory, disk space."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemInfo:
    os: str  # "darwin", "linux", "windows"
    arch: str  # "arm64", "x86_64"
    is_apple_silicon: bool
    chip_name: str  # "Apple M2 Pro", "Intel Core i7", etc.
    total_memory_gb: float
    available_disk_gb: float


def _get_chip_name() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return platform.processor() or "Unknown"


def _get_total_memory_gb() -> float:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
    else:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        except (ValueError, AttributeError):
            pass
    return 0.0


def _get_available_disk_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024**3)
    except OSError:
        return 0.0


def get_system_info(project_root: Path | None = None) -> SystemInfo:
    """Gather system information for the current machine."""
    os_name = platform.system().lower()
    arch = platform.machine()
    is_apple_silicon = os_name == "darwin" and arch == "arm64"

    disk_path = project_root or Path.cwd()

    return SystemInfo(
        os=os_name,
        arch=arch,
        is_apple_silicon=is_apple_silicon,
        chip_name=_get_chip_name(),
        total_memory_gb=round(_get_total_memory_gb(), 1),
        available_disk_gb=round(_get_available_disk_gb(disk_path), 1),
    )
