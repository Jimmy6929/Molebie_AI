"""
Gateway test configuration.

Pytest fixtures and shared test utilities.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the gateway app to the path for imports
gateway_root = Path(__file__).parent.parent
sys.path.insert(0, str(gateway_root))


@pytest.fixture
def test_env(tmp_path):
    """Set up test environment variables with a temporary data directory."""
    original_env = os.environ.copy()

    os.environ.update({
        "DEBUG": "true",
        "JWT_SECRET": "test_jwt_secret_for_testing_only",
        "DATA_DIR": str(tmp_path / "data"),
        "AUTH_MODE": "single",
    })

    yield os.environ

    os.environ.clear()
    os.environ.update(original_env)
