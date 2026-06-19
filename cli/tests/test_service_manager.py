"""Tests for service launch definitions.

Regression guard for the fresh-install bug where MLX servers were launched with
a bare ``"python3"`` (PATH-resolved → system Python without uvicorn/mlx-vlm)
while the Gateway correctly used the project venv interpreter. Every Python
service must launch from the same ``.venv`` interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cli.models.config import InferenceBackend, MolebieConfig
from cli.services.config_manager import get_project_root
from cli.services.service_manager import get_service_definitions


def _expected_venv_python() -> str:
    """The interpreter path get_service_definitions should use, derived the same way."""
    root = get_project_root()
    if sys.platform == "win32":
        return str(root / ".venv" / "Scripts" / "python.exe")
    return str(root / ".venv" / "bin" / "python")


def test_mlx_uses_venv_python() -> None:
    """MLX Thinking/Instant must launch from the project venv, not a bare 'python3'."""
    config = MolebieConfig(
        run_inference=True,
        run_gateway=False,
        run_webapp=False,
        inference_backend=InferenceBackend.MLX,
    )
    defs = {s.name: s for s in get_service_definitions(config)}

    expected = _expected_venv_python()
    for name in ("MLX Thinking", "MLX Instant"):
        assert name in defs, f"{name} should be defined for MLX inference"
        launcher = defs[name].start_cmd[0]
        assert launcher == expected, f"{name} launched with {launcher!r}, expected {expected!r}"
        assert launcher != "python3", f"{name} must not use a PATH-resolved python3"
        # Sanity: it still runs the MLX server script.
        assert defs[name].start_cmd[1].endswith("mlx_server.py")


def test_mlx_and_gateway_share_one_interpreter() -> None:
    """MLX and Gateway must agree on the launch interpreter (single source of truth)."""
    config = MolebieConfig(
        run_inference=True,
        run_gateway=True,
        run_webapp=False,
        inference_backend=InferenceBackend.MLX,
    )
    defs = {s.name: s for s in get_service_definitions(config)}

    gateway_python = defs["Gateway"].start_cmd[0]
    assert defs["MLX Thinking"].start_cmd[0] == gateway_python
    assert defs["MLX Instant"].start_cmd[0] == gateway_python
    assert ".venv" in Path(gateway_python).parts
