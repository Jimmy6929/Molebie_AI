"""Cross-platform system probe for the live monitor.

Polls host CPU / RAM / GPU once per second in a background task and caches
the most recent snapshot. The `/metrics/live` route reads the cache — it
never triggers a probe itself, so request latency is unaffected.

Three implementations, chosen by `probe_for_host()` at startup:

  * MacOSAppleSiliconProbe — psutil always, `macmon pipe` subprocess for
    GPU/power/temp if `macmon` is on PATH. Graceful-degrade: GPU fields
    return None when macmon isn't installed.
  * LinuxNvidiaProbe — psutil + pynvml (lazy-imported). If pynvml fails,
    degrades to CPU-only.
  * GenericCpuProbe — psutil only. Used on Windows, or Linux without GPU.

All probes return a `SystemSnapshot` dataclass; callers treat `None`
fields as "unavailable, show em-dash".
"""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(slots=True)
class SystemSnapshot:
    ts: float
    cpu_percent: float | None = None
    cpu_cores: int | None = None         # logical cores (psutil.cpu_count)
    cpu_cores_physical: int | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    ram_percent: float | None = None
    gpu_percent: float | None = None
    gpu_temp_c: float | None = None
    power_w: float | None = None
    # Human-readable note for the CLI footer (e.g. "macmon not installed").
    note: str | None = None


class SystemProbe:
    """Base class — subclass implements `poll_once()`."""

    def __init__(self) -> None:
        self._latest = SystemSnapshot(ts=time.time())
        self._task: asyncio.Task | None = None
        self._stopping = False

    def latest(self) -> SystemSnapshot:
        return self._latest

    async def start(self, interval: float = 1.0) -> None:
        """Kick off the background poll loop. Idempotent."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(interval))

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self, interval: float) -> None:
        while not self._stopping:
            try:
                snap = await asyncio.to_thread(self.poll_once)
                self._latest = snap
            except Exception as exc:
                self._latest = SystemSnapshot(ts=time.time(), note=f"probe error: {exc}")
            await asyncio.sleep(interval)

    def poll_once(self) -> SystemSnapshot:        # pragma: no cover - overridden
        raise NotImplementedError


# ── concrete impls ────────────────────────────────────────────────────


class GenericCpuProbe(SystemProbe):
    """psutil-only fallback. Works on any OS with psutil installed."""

    def poll_once(self) -> SystemSnapshot:
        import psutil

        mem = psutil.virtual_memory()
        return SystemSnapshot(
            ts=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_cores=psutil.cpu_count(logical=True),
            cpu_cores_physical=psutil.cpu_count(logical=False),
            ram_used_gb=mem.used / (1024 ** 3),
            ram_total_gb=mem.total / (1024 ** 3),
            ram_percent=(mem.used / mem.total * 100.0) if mem.total else 0.0,
        )


class MacOSAppleSiliconProbe(SystemProbe):
    """psutil for CPU/RAM; macmon subprocess for GPU/power/temp."""

    def __init__(self) -> None:
        super().__init__()
        self._macmon_path = shutil.which("macmon")
        self._macmon_proc: subprocess.Popen | None = None
        self._macmon_note_emitted = False
        self._last_gpu: float | None = None
        self._last_temp: float | None = None
        self._last_power: float | None = None

    def _start_macmon(self) -> None:
        """Launch `macmon pipe -s 1000 -i 500` — emits NDJSON on stdout."""
        if self._macmon_path is None or self._macmon_proc is not None:
            return
        try:
            self._macmon_proc = subprocess.Popen(
                [self._macmon_path, "pipe", "-s", "1000", "-i", "500"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (OSError, FileNotFoundError):
            self._macmon_proc = None

    def _read_macmon(self) -> None:
        """Drain any pending lines from macmon; update last-seen fields."""
        if self._macmon_proc is None or self._macmon_proc.stdout is None:
            return
        import select

        fd = self._macmon_proc.stdout.fileno()
        while True:
            rlist, _, _ = select.select([fd], [], [], 0)
            if not rlist:
                break
            line = self._macmon_proc.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            # macmon schema: gpu_usage = [freq, usage], temp.gpu_temp_avg, gpu_power, ...
            gu = data.get("gpu_usage")
            if isinstance(gu, list) and len(gu) >= 2:
                self._last_gpu = float(gu[1]) * 100.0 if gu[1] <= 1.0 else float(gu[1])
            temp = data.get("temp", {})
            if isinstance(temp, dict) and "gpu_temp_avg" in temp:
                self._last_temp = float(temp["gpu_temp_avg"])
            if "gpu_power" in data and "cpu_power" in data:
                self._last_power = float(data["gpu_power"]) + float(data["cpu_power"])

    def poll_once(self) -> SystemSnapshot:
        import psutil

        mem = psutil.virtual_memory()
        snap = SystemSnapshot(
            ts=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_cores=psutil.cpu_count(logical=True),
            cpu_cores_physical=psutil.cpu_count(logical=False),
            ram_used_gb=mem.used / (1024 ** 3),
            ram_total_gb=mem.total / (1024 ** 3),
            ram_percent=(mem.used / mem.total * 100.0) if mem.total else 0.0,
        )

        if self._macmon_path is None:
            if not self._macmon_note_emitted:
                snap.note = "Install `brew install macmon` for GPU metrics"
                self._macmon_note_emitted = True
            return snap

        self._start_macmon()
        self._read_macmon()
        snap.gpu_percent = self._last_gpu
        snap.gpu_temp_c = self._last_temp
        snap.power_w = self._last_power
        return snap

    async def stop(self) -> None:
        if self._macmon_proc is not None:
            try:
                self._macmon_proc.terminate()
            except ProcessLookupError:
                pass
            self._macmon_proc = None
        await super().stop()


class LinuxNvidiaProbe(SystemProbe):
    """psutil for CPU/RAM; pynvml for GPU. Degrades to CPU if NVML missing."""

    def __init__(self) -> None:
        super().__init__()
        self._nvml_ok: bool | None = None
        self._handle = None

    def _try_init_nvml(self) -> None:
        if self._nvml_ok is not None:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
        except Exception:
            self._nvml_ok = False

    def poll_once(self) -> SystemSnapshot:
        import psutil

        mem = psutil.virtual_memory()
        snap = SystemSnapshot(
            ts=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_cores=psutil.cpu_count(logical=True),
            cpu_cores_physical=psutil.cpu_count(logical=False),
            ram_used_gb=mem.used / (1024 ** 3),
            ram_total_gb=mem.total / (1024 ** 3),
            ram_percent=(mem.used / mem.total * 100.0) if mem.total else 0.0,
        )

        self._try_init_nvml()
        if self._nvml_ok and self._handle is not None:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                except Exception:
                    power = None
                snap.gpu_percent = float(util.gpu)
                snap.gpu_temp_c = float(temp)
                snap.power_w = power
            except Exception as exc:
                snap.note = f"pynvml error: {exc}"
        else:
            snap.note = "Install `pip install nvidia-ml-py` for GPU metrics"
        return snap


# ── factory ──────────────────────────────────────────────────────────


def probe_for_host() -> SystemProbe:
    """Pick the right probe implementation for the current machine."""
    sysname = platform.system()
    machine = platform.machine()

    if sysname == "Darwin" and machine in ("arm64", "aarch64"):
        return MacOSAppleSiliconProbe()
    if sysname == "Linux":
        # Cheap heuristic — try NVML; the probe itself degrades if it fails.
        return LinuxNvidiaProbe()
    return GenericCpuProbe()


_probe: SystemProbe | None = None


def get_system_probe() -> SystemProbe:
    global _probe
    if _probe is None:
        _probe = probe_for_host()
    return _probe


def reset_system_probe() -> None:
    """Test-only helper."""
    global _probe
    _probe = None


# Silence unused-import for platforms where sys isn't otherwise used
_ = sys
