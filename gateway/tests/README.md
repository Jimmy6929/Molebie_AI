# gateway/tests

This directory contains all tests for the Gateway API.

## Running Tests

```bash
# From the gateway directory
cd gateway

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_inference.py -v
```

## Test Files

- `conftest.py` - Shared fixtures and test configuration
- `test_chat_full.py` - Full integration tests for chat endpoints
- `test_inference.py` - Tests for inference service
- `test_sessions.py` - Tests for session management
- `test_chat.sh` - Shell script for manual API testing
