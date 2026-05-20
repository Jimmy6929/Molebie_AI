"""Integration tests for cross-tier fallback in InferenceService.

The bug these tests guard against: when the selector cross-tier-falls-back
(thinking pool has no eligible backend → selector returns an instant backend),
the code used to resolve all per-mode parameters (max_tokens, temperature,
enable_thinking, thinking_budget, extra_payload, etc.) against the *requested*
mode (`"thinking"`), not the backend's actual tier. Result: an instant model
would receive thinking-tier params and even thinking_budget — silently wrong.

The fix: when ``backend.tier != requested_tier``, all per-mode resolutions
use ``backend.tier`` instead. These tests pin that behavior so a future
refactor can't regress it without going red.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from app.config import get_settings
from app.services.inference import InferenceService


@pytest.fixture(autouse=True)
def _restore_settings_cache():
    """Other tests in this repo share a cached Settings singleton; we mutate it
    in these tests to simulate different deployment configs. Clear the lru_cache
    around each test so the cached instance from a previous test (or this one)
    never leaks into a sibling test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _service_with_only_instant() -> InferenceService:
    """Build a service where only the instant tier is configured.

    Forces the selector into cross-tier fallback territory: a request for
    mode='thinking' must route to the instant backend.
    """
    settings = get_settings()
    settings.inference_instant_url = "http://test-instant:8081"
    settings.inference_thinking_url = None
    settings.routing_thinking_fallback_to_instant = True
    return InferenceService(settings)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _StreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_non_stream_cross_tier_fallback_uses_instant_params():
    """generate_response(mode='thinking') with only-instant pool must POST
    a payload whose max_tokens, temperature, enable_thinking come from the
    instant tier — not the thinking tier."""
    svc = _service_with_only_instant()

    captured_payload: dict = {}

    async def fake_post(self, url, json=None, headers=None, **kw):  # noqa: A002
        captured_payload["url"] = url
        captured_payload["body"] = json
        return _FakeResponse({
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 1, "prompt_tokens": 1, "completion_tokens": 0},
            "model": "instant-model",
        })

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        result = await svc.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            mode="thinking",
        )

    # The request actually went to the instant URL (cross-tier fallback).
    assert "test-instant:8081" in captured_payload["url"]

    # Critical assertion: the params must match instant-tier defaults, not
    # thinking-tier. Thinking-tier max_tokens defaults are large (8k+); instant
    # is small (1-4k). The exact value depends on settings — we assert by
    # comparing the captured value against what the service would resolve for
    # the instant tier directly.
    assert captured_payload["body"]["max_tokens"] == svc._get_max_tokens("instant")
    assert captured_payload["body"]["temperature"] == svc._get_temperature("instant")
    # enable_thinking must NOT be the thinking-tier default if it differs from instant.
    assert captured_payload["body"]["enable_thinking"] == svc._get_enable_thinking("instant")
    # thinking_budget must NOT be set when calling an instant backend.
    assert "thinking_budget" not in captured_payload["body"]
    # Result surfaces the cross-tier fallback to the caller.
    assert result["fallback_used"] is True
    assert result["original_mode"] == "thinking"


@pytest.mark.asyncio
async def test_stream_cross_tier_fallback_uses_instant_params_and_meta():
    """generate_response_stream(mode='thinking') with only-instant pool must
    stream with instant-tier params AND emit the metadata event with the
    *actual* tier, not the requested one."""
    svc = _service_with_only_instant()

    captured_payload: dict = {}

    class _StreamCtx:
        def __init__(self, url, json=None, headers=None, **kw):  # noqa: A002
            captured_payload["url"] = url
            captured_payload["body"] = json

        async def __aenter__(self):
            return _StreamResponse([
                'data: {"choices":[{"delta":{"content":"hi"}}]}',
                "data: [DONE]",
            ])

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_stream(self, method, url, **kw):
        return _StreamCtx(url, **kw)

    chunks: list[str] = []
    with patch.object(httpx.AsyncClient, "stream", new=fake_stream):
        async for chunk in svc.generate_response_stream(
            messages=[{"role": "user", "content": "hi"}],
            mode="thinking",
        ):
            chunks.append(chunk)

    # The stream actually went to the instant URL.
    assert "test-instant:8081" in captured_payload["url"]

    # Same param assertions as the non-stream case.
    body = captured_payload["body"]
    assert body["max_tokens"] == svc._get_max_tokens("instant")
    assert body["temperature"] == svc._get_temperature("instant")
    assert body["enable_thinking"] == svc._get_enable_thinking("instant")
    assert "thinking_budget" not in body

    # The metadata event (the very first SSE the gateway emits) must report
    # the actual tier serving the request, not the originally-requested mode.
    first = chunks[0]
    assert first.startswith("data: ")
    meta_event = json.loads(first[6:].strip())
    assert meta_event["metadata"]["mode"] == "instant"
    assert meta_event["metadata"]["fallback_used"] is True


@pytest.mark.asyncio
async def test_in_tier_call_uses_requested_mode_params():
    """Sanity check: when the selector returns a backend on the requested
    tier (no cross-tier fallback), per-mode params resolve to that tier's
    defaults — not the fallback's."""
    settings = get_settings()
    settings.inference_instant_url = "http://test-instant:8081"
    settings.inference_thinking_url = "http://test-thinking:8080"
    settings.routing_thinking_fallback_to_instant = True
    svc = InferenceService(settings)

    captured_payload: dict = {}

    async def fake_post(self, url, json=None, headers=None, **kw):  # noqa: A002
        captured_payload["url"] = url
        captured_payload["body"] = json
        return _FakeResponse({
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 1, "prompt_tokens": 1, "completion_tokens": 0},
            "model": "thinking-model",
        })

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        await svc.generate_response(
            messages=[{"role": "user", "content": "solve"}],
            mode="thinking",
        )

    assert "test-thinking:8080" in captured_payload["url"]
    assert captured_payload["body"]["max_tokens"] == svc._get_max_tokens("thinking")
    assert captured_payload["body"]["temperature"] == svc._get_temperature("thinking")
    # No fallback flag because we stayed on-tier.
    # (Note: when enable_thinking and budget are set, they should be the thinking-tier values.)
    if svc._get_enable_thinking("thinking"):
        assert captured_payload["body"]["enable_thinking"] is True
        if svc._get_thinking_budget("thinking") is not None:
            assert captured_payload["body"]["thinking_budget"] == svc._get_thinking_budget("thinking")
