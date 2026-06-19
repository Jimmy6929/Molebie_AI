"""Regression tests for install.sh's Python capability gate.

sqlite-vec loads as a runtime extension and requires a Python whose sqlite3 was
built with loadable-extension support (``conn.enable_load_extension``). Version
and wheel availability are NOT sufficient — a python.org macOS build passes both
yet crashes the gateway at startup. install.sh must reject such interpreters.

These tests guard both the canonical probe and the bash wiring against drift.
The pure-bash control flow is hard to unit-test, so we exercise the real
extracted helper and assert the structural wiring is present.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"

CAPABILITY_PROBE = (
    "import sqlite3,sys; "
    "sys.exit(0 if hasattr(sqlite3.connect(':memory:'),'enable_load_extension') else 1)"
)


def _extract_bash_func(name: str) -> str:
    """Pull a top-level ``name() { ... }`` function out of install.sh verbatim."""
    text = INSTALL_SH.read_text()
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", text, re.MULTILINE | re.DOTALL)
    assert m, f"{name}() not found in install.sh"
    return m.group(0)


def test_capability_probe_truthy_on_current_interpreter() -> None:
    """The canonical one-liner detects extension support on the test interpreter.

    The test suite runs under the project venv, which must be capable — otherwise
    the gateway could not load sqlite-vec.
    """
    rc = subprocess.run([sys.executable, "-c", CAPABILITY_PROBE]).returncode
    assert rc == 0, "test interpreter lacks sqlite extension support"


def test_install_sh_bash_helper_matches_probe() -> None:
    """The real py_supports_sqlite_ext bash function agrees on the test interpreter."""
    func = _extract_bash_func("py_supports_sqlite_ext")
    script = f'{func}\npy_supports_sqlite_ext "{sys.executable}"\n'
    rc = subprocess.run(["bash", "-c", script]).returncode
    assert rc == 0, "extracted bash helper disagrees with the canonical probe"


def test_discovery_gate_is_capability_aware() -> None:
    """install.sh selects Python on fitness (version + extensions), not version alone."""
    text = INSTALL_SH.read_text()
    for fn in ("py_supports_sqlite_ext()", "py_is_fit()", "rebuild_and_verify_venv()"):
        assert fn in text, f"{fn} missing from install.sh"
    # Discovery loop must screen candidates through the fitness predicate.
    assert 'if py_is_fit "$candidate"' in text
    # The old version-only acceptance (PY_MINOR / FB_MINOR) must be gone.
    assert "PY_MINOR" not in text and "FB_MINOR" not in text
    # Step 2b must verify extension support, not just wheels.
    assert "EXT_OK" in text and "WHEELS_OK" in text


def test_macos_remediation_points_to_homebrew_not_pythonorg() -> None:
    """On macOS the documented fix must be Homebrew; python.org won't satisfy sqlite-vec."""
    text = INSTALL_SH.read_text()
    assert "python.org builds will NOT work" in text
    assert "brew install python@3.12" in text
