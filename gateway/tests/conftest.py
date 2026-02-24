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
def test_env():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Override with test values
    os.environ.update({
        "DEBUG": "true",
        "SUPABASE_URL": "http://127.0.0.1:54321",
        "SUPABASE_ANON_KEY": "test_anon_key",
        "SUPABASE_SERVICE_ROLE_KEY": "test_service_role_key",
        "JWT_SECRET": "test_jwt_secret",
    })
    
    yield os.environ
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_supabase_client(mocker):
    """Mock Supabase client for unit tests."""
    return mocker.patch("app.services.database.get_supabase_client")
