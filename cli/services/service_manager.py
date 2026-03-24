"""Start, stop, and health-check services."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from cli.models.config import InferenceBackend, MolebieConfig, SetupType
from cli.services.config_manager import get_project_root
from cli.ui.console import console, print_fail, print_info, print_ok, print_warn


@dataclass
class ServiceDef:
    name: str
    port: int
    health_url: str
    start_cmd: list[str]
    cwd: Optional[str] = None
    env_extra: dict[str, str] = field(default_factory=dict)
    is_docker: bool = False
    is_optional: bool = False
    start_order: int = 0


def _check_health(url: str, timeout: float = 3.0) -> bool:
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        return resp.status_code < 400
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _wait_healthy(url: str, name: str, max_wait: int = 60) -> bool:
    """Poll health endpoint until healthy or timeout."""
    for i in range(max_wait):
        if _check_health(url):
            return True
        time.sleep(1)
    return False


def get_service_definitions(config: MolebieConfig) -> list[ServiceDef]:
    """Build list of services to run based on config."""
    root = get_project_root()
    ip = config.server_ip if config.setup_type == SetupType.TWO_MACHINE else "localhost"
    gpu_ip = config.gpu_ip if config.setup_type == SetupType.TWO_MACHINE else "localhost"

    services: list[ServiceDef] = []

    # 1. Supabase
    services.append(ServiceDef(
        name="Supabase",
        port=54321,
        health_url=f"http://{ip}:54321/rest/v1/",
        start_cmd=["supabase", "start"],
        cwd=str(root / "supabase"),
        start_order=0,
    ))

    # 2. Docker services
    if config.search_enabled:
        services.append(ServiceDef(
            name="SearXNG",
            port=8888,
            health_url=f"http://{ip}:8888/",
            start_cmd=["docker", "compose", "up", "-d", "searxng"],
            cwd=str(root),
            is_docker=True,
            is_optional=True,
            start_order=1,
        ))

    if config.voice_enabled:
        services.append(ServiceDef(
            name="Kokoro TTS",
            port=8880,
            health_url=f"http://{ip}:8880/",
            start_cmd=["docker", "compose", "up", "-d", "kokoro-tts"],
            cwd=str(root),
            is_docker=True,
            is_optional=True,
            start_order=1,
        ))

    # 3. Inference (only on single-machine or if we're the GPU machine)
    if config.setup_type == SetupType.SINGLE:
        if config.inference_backend == InferenceBackend.MLX:
            services.append(ServiceDef(
                name="MLX Thinking",
                port=8080,
                health_url=f"http://localhost:8080/v1/models",
                start_cmd=["python3", str(root / "scripts" / "mlx_server.py"),
                            "--host", "0.0.0.0", "--port", "8080"],
                start_order=2,
            ))
            services.append(ServiceDef(
                name="MLX Instant",
                port=8081,
                health_url=f"http://localhost:8081/v1/models",
                start_cmd=["python3", str(root / "scripts" / "mlx_server.py"),
                            "--host", "0.0.0.0", "--port", "8081"],
                is_optional=True,
                start_order=2,
            ))

    # 4. Gateway
    services.append(ServiceDef(
        name="Gateway",
        port=8000,
        health_url=f"http://{ip}:8000/health",
        start_cmd=[
            "python3", "-m", "uvicorn", "app.main:app",
            "--reload", "--host", "0.0.0.0", "--port", "8000",
        ],
        cwd=str(root / "gateway"),
        env_extra={"KMP_DUPLICATE_LIB_OK": "TRUE"},
        start_order=3,
    ))

    # 5. Webapp
    services.append(ServiceDef(
        name="Webapp",
        port=3000,
        health_url=f"http://{ip}:3000",
        start_cmd=["npm", "run", "dev"],
        cwd=str(root / "webapp"),
        start_order=4,
    ))

    services.sort(key=lambda s: s.start_order)
    return services


class ServiceRunner:
    """Manages starting services and clean shutdown."""

    def __init__(self) -> None:
        self._processes: list[tuple[str, subprocess.Popen]] = []
        self._shutdown = False

    def _start_background(self, svc: ServiceDef) -> subprocess.Popen | None:
        """Start a long-running service as a background process."""
        env = os.environ.copy()
        env.update(svc.env_extra)
        try:
            proc = subprocess.Popen(
                svc.start_cmd,
                cwd=svc.cwd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return proc
        except FileNotFoundError:
            print_fail(f"Command not found: {svc.start_cmd[0]}")
            return None

    def _start_and_wait(self, svc: ServiceDef) -> bool:
        """Start a service that runs to completion (supabase start, docker compose)."""
        try:
            result = subprocess.run(
                svc.start_cmd,
                cwd=svc.cwd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def start_services(
        self,
        config: MolebieConfig,
        service_filter: str | None = None,
        skip_inference: bool = False,
    ) -> None:
        """Start all configured services."""
        definitions = get_service_definitions(config)

        if service_filter:
            definitions = [s for s in definitions if s.name.lower() == service_filter.lower()]
            if not definitions:
                print_fail(f"Unknown service: {service_filter}")
                return

        if skip_inference:
            definitions = [s for s in definitions if "MLX" not in s.name and "Ollama" not in s.name]

        # Set up signal handler
        def _signal_handler(sig: int, frame: object) -> None:
            self._shutdown = True
            console.print()
            print_info("Shutting down services...")
            self.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        for svc in definitions:
            if self._shutdown:
                break

            # Skip if already running
            if _check_health(svc.health_url, timeout=2.0):
                print_ok(f"{svc.name} already running on :{svc.port}")
                continue

            print_info(f"Starting {svc.name}...")

            # Supabase and Docker services run-to-completion
            if svc.name == "Supabase" or svc.is_docker:
                if self._start_and_wait(svc):
                    if _wait_healthy(svc.health_url, svc.name, max_wait=30):
                        print_ok(f"{svc.name} started on :{svc.port}")
                    else:
                        print_warn(f"{svc.name} started but not yet healthy — may need more time")
                else:
                    if svc.is_optional:
                        print_warn(f"{svc.name} failed to start (optional, continuing)")
                    else:
                        print_fail(f"{svc.name} failed to start")
                continue

            # Long-running services
            proc = self._start_background(svc)
            if proc is None:
                if svc.is_optional:
                    print_warn(f"{svc.name} failed to start (optional, continuing)")
                else:
                    print_fail(f"{svc.name} failed to start")
                continue

            self._processes.append((svc.name, proc))

            # Wait for health
            if _wait_healthy(svc.health_url, svc.name, max_wait=30):
                print_ok(f"{svc.name} started on :{svc.port}")
            else:
                # Check if process died
                if proc.poll() is not None:
                    if svc.is_optional:
                        print_warn(f"{svc.name} exited unexpectedly (optional, continuing)")
                    else:
                        print_fail(f"{svc.name} exited unexpectedly")
                else:
                    print_warn(f"{svc.name} running but not healthy yet — may need more time")

        if not self._shutdown and self._processes:
            console.print()
            print_ok("All services started. Press Ctrl+C to stop.")
            console.print()
            # Block until signal
            try:
                while not self._shutdown:
                    # Check if any process has died
                    for name, proc in self._processes:
                        if proc.poll() is not None:
                            print_warn(f"{name} exited with code {proc.returncode}")
                    time.sleep(2)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop_all()

    def stop_all(self) -> None:
        """Stop all managed background processes."""
        for name, proc in self._processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print_ok(f"{name} stopped")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print_warn(f"{name} killed (did not stop gracefully)")
        self._processes.clear()
