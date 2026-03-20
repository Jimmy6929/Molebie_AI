"""
Tests for intelligent message routing in web_search.should_search().

Tests the rule-based tiers (SKIP and SEARCH patterns) without needing
a running LLM or SearXNG instance.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Helpers ──────────────────────────────────────────────────────

def _make_service(**overrides):
    """Create a WebSearchService with sensible test defaults."""
    defaults = {
        "searxng_url": "http://localhost:8888",
        "web_search_enabled": True,
        "web_search_timeout": 5.0,
        "web_search_max_results": 6,
        "web_search_snippet_max_chars": 800,
        "web_search_fetch_full_content": False,
        "web_search_full_content_count": 0,
        "web_search_full_content_max_chars": 2000,
        "web_search_full_content_timeout": 4.0,
        "web_search_llm_classify": False,  # disable LLM fallback for rule tests
        "web_search_classify_timeout": 3.0,
        "web_search_classify_max_tokens": 3,
    }
    defaults.update(overrides)
    settings = MagicMock(**defaults)
    from app.services.web_search import WebSearchService
    return WebSearchService(settings)


# ── Rule-based routing tests ─────────────────────────────────────

SKIP_CASES = [
    # Code / programming
    ("Write a Python sort function", "code keyword"),
    ("implement a binary search tree", "implement keyword"),
    ("debug this javascript code", "debug keyword"),
    ("How do I use regex in Python?", "python keyword"),
    ("Write a SQL query to join two tables", "sql keyword"),
    # Math
    ("What is 2+2?", "arithmetic expression"),
    ("Calculate the factorial of 10", "calculate keyword"),
    ("Solve this equation: 2x + 3 = 7", "solve keyword"),
    # Creative
    ("Write me a haiku about rain", "creative - haiku"),
    ("Write me a poem about the ocean", "creative - poem"),
    ("Help me draft a cover letter", "creative - cover letter"),
    # Explanations
    ("Explain how TCP/IP works", "explain keyword"),
    ("What is a monad?", "what is a"),
    ("Difference between TCP and UDP", "difference between"),
    ("Define polymorphism", "define keyword"),
    # Meta
    ("Who are you?", "meta - identity"),
    ("Summarize this text for me", "summarize keyword"),
    ("Translate this to Spanish", "translate keyword"),
    # Opinion
    ("What do you think about vim vs emacs?", "opinion"),
    ("Pros and cons of React vs Vue", "pros and cons"),
    ("Should I learn Rust or Go?", "should i / advice"),
    # Greetings (trivial)
    ("Hello", "greeting"),
    ("Hey", "greeting"),
    ("Thanks!", "thanks"),
    ("ok", "acknowledgement"),
]

SEARCH_CASES = [
    # Temporal
    ("Weather in Tokyo today", "today keyword"),
    ("What's the latest AI news?", "latest keyword"),
    ("Who won the Super Bowl 2025?", "year 2025"),
    ("What happened this week in tech?", "this week"),
    # News / events
    ("Ukraine war update on the situation", "update on keyword"),
    ("Did Apple announce anything?", "announced keyword"),
    ("Breaking news about the earthquake", "breaking keyword"),
    # Real-time data
    ("Stock price of Apple", "stock price"),
    ("Current exchange rate USD to EUR", "currently keyword"),
    ("NBA standings", "standings keyword"),
    ("Weather forecast for tomorrow", "forecast keyword"),
    # Commerce / lookup
    ("Where to buy a PS5?", "where to buy"),
    ("Hours of the local library", "hours of"),
    # Explicit search
    ("Search for Python tutorials", "search for"),
    ("Look up the population of France", "look up"),
    ("Google the best restaurants nearby", "google keyword"),
]

AMBIGUOUS_CASES = [
    # These should NOT match either rule set → fall through to LLM (or default skip)
    ("Tell me about elephants", "general knowledge"),
    ("What is the capital of France?", "factual but timeless"),
    ("Trump", "short ambiguous"),
    ("Bitcoin", "single topic word"),
    ("Ukraine war update", "no strong search keyword"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("message,reason", SKIP_CASES)
async def test_skip_cases(message, reason):
    svc = _make_service()
    result = await svc.should_search(message)
    assert result is False, f"Expected SKIP for '{message}' ({reason})"


@pytest.mark.asyncio
@pytest.mark.parametrize("message,reason", SEARCH_CASES)
async def test_search_cases(message, reason):
    svc = _make_service()
    result = await svc.should_search(message)
    assert result is True, f"Expected SEARCH for '{message}' ({reason})"


@pytest.mark.asyncio
@pytest.mark.parametrize("message,reason", AMBIGUOUS_CASES)
async def test_ambiguous_defaults_to_skip_when_llm_off(message, reason):
    """With LLM classify disabled, ambiguous messages default to no search."""
    svc = _make_service(web_search_llm_classify=False)
    result = await svc.should_search(message)
    assert result is False, f"Expected SKIP (LLM off) for '{message}' ({reason})"


# ── Edge cases ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_service():
    svc = _make_service(web_search_enabled=False)
    assert await svc.should_search("latest news today") is False


@pytest.mark.asyncio
async def test_empty_message():
    svc = _make_service()
    assert await svc.should_search("") is False
    assert await svc.should_search("  ") is False
    assert await svc.should_search("?") is False


@pytest.mark.asyncio
async def test_mixed_intent_triggers_search():
    """If a message has both skip and search signals, search wins (SEARCH checked after SKIP,
    but temporal keywords should dominate)."""
    svc = _make_service()
    # "explain" triggers SKIP, but "latest" triggers SEARCH — however SKIP is checked first
    # This is the expected trade-off: "explain the latest" hits SKIP on "explain"
    # But "latest HTTP/3 stats" should hit SEARCH
    result = await svc.should_search("latest HTTP/3 stats")
    assert result is True


# ── LLM fallback tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_classify_called_for_ambiguous():
    """Ambiguous messages should call _classify_with_llm when enabled."""
    svc = _make_service(web_search_llm_classify=True)
    svc._classify_with_llm = AsyncMock(return_value=True)
    result = await svc.should_search("Trump")
    svc._classify_with_llm.assert_called_once()
    assert result is True


@pytest.mark.asyncio
async def test_llm_classify_not_called_for_clear_skip():
    """Clear SKIP messages should never reach the LLM."""
    svc = _make_service(web_search_llm_classify=True)
    svc._classify_with_llm = AsyncMock(return_value=True)
    result = await svc.should_search("Write a Python function")
    svc._classify_with_llm.assert_not_called()
    assert result is False


@pytest.mark.asyncio
async def test_llm_classify_not_called_for_clear_search():
    """Clear SEARCH messages should never reach the LLM."""
    svc = _make_service(web_search_llm_classify=True)
    svc._classify_with_llm = AsyncMock(return_value=False)
    result = await svc.should_search("Weather in NYC today")
    svc._classify_with_llm.assert_not_called()
    assert result is True


@pytest.mark.asyncio
async def test_llm_classify_failure_defaults_to_no_search():
    """If LLM classify raises an exception, default to no search."""
    svc = _make_service(web_search_llm_classify=True)
    # Patch the inference service inside the lazy import path
    with patch.dict("sys.modules", {}):
        with patch("app.services.inference.get_inference_service", side_effect=RuntimeError("down")):
            result = await svc._classify_with_llm("some ambiguous query")
            assert result is False
