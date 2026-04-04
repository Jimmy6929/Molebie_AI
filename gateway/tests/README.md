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
pytest tests/test_web_search_routing.py -v
```

## Test Files

- `conftest.py` - Shared fixtures and test configuration
- `test_web_search_routing.py` - Tests for web search intent routing
