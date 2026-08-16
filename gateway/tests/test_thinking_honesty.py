"""Tests for the T3 thinking-honesty rules.

The 2026-08-14 eval found thinking mode barely thinks: the CoT
auto-disable fired on ANY retrieval (even LOW/generative routing), and
the intent classifier's load-induced timeouts failed closed to "lookup",
silently stripping reasoning. These tests pin the new policy:
explicit user mode choice wins; only lookup-routed retrievals qualify;
classifier failure fails OPEN.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.models.chat import ChatRequest
from app.routes.chat import _decide_thinking_override
from app.services.intent import _classify_inner


class _Settings:
    inference_thinking_auto_disable_for_rag = True


def _decide(**overrides):
    kwargs = {
        "mode_source": "default",
        "inference_mode": "thinking",
        "rag_chunks": [{"chunk_id": "c1"}],
        "routing_mode": "lookup",
        "explain_intent": False,
        "explain_classify_failed": False,
        "settings": _Settings(),
    }
    kwargs.update(overrides)
    return _decide_thinking_override(**kwargs)


# ── _decide_thinking_override policy matrix ────────────────────────────────


def test_defaulted_lookup_turn_disables_cot_with_reason():
    assert _decide() == (False, "rag_lookup_auto_disable")


def test_explicit_user_mode_choice_always_wins():
    assert _decide(mode_source="user") == (None, None)


def test_generative_routing_keeps_cot():
    # A LOW/NONE-confidence turn that happened to retrieve a weak chunk
    # must not lose CoT — only lookup-routed retrievals qualify.
    assert _decide(routing_mode="generative") == (None, None)


def test_classifier_failure_fails_open():
    assert _decide(explain_classify_failed=True) == (None, None)


def test_explain_intent_keeps_cot():
    assert _decide(explain_intent=True) == (None, None)


def test_instant_mode_untouched():
    assert _decide(inference_mode="instant") == (None, None)


def test_no_chunks_untouched():
    assert _decide(rag_chunks=[]) == (None, None)


def test_kill_switch_config_off_untouched():
    class _Off:
        inference_thinking_auto_disable_for_rag = False

    assert _decide(settings=_Off()) == (None, None)


def test_thinking_harder_also_gated():
    assert _decide(inference_mode="thinking_harder") == (
        False, "rag_lookup_auto_disable",
    )


# ── ChatRequest.mode_source ────────────────────────────────────────────────


def test_mode_source_defaults_to_default():
    req = ChatRequest(message="hi")
    assert req.mode_source == "default"


def test_mode_source_accepts_user():
    req = ChatRequest(message="hi", mode_source="user")
    assert req.mode_source == "user"


def test_mode_source_rejects_unknown_values():
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", mode_source="sometimes")


# ── classify_explain_intent tri-state ──────────────────────────────────────


class _StubInference:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    async def generate_response(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return {"content": self._content}


@pytest.mark.asyncio
async def test_classify_returns_true_on_explain():
    with patch("app.services.inference.get_inference_service",
               return_value=_StubInference(content="EXPLAIN")):
        assert await _classify_inner("why does the sky look blue?") is True


@pytest.mark.asyncio
async def test_classify_returns_false_on_lookup():
    with patch("app.services.inference.get_inference_service",
               return_value=_StubInference(content="LOOKUP")):
        assert await _classify_inner("what port does the gateway use?") is False


@pytest.mark.asyncio
async def test_classify_returns_none_on_failure():
    # TimeoutError's str() is empty — the exact failure signature that
    # used to fail closed and strip CoT under load.
    with patch("app.services.inference.get_inference_service",
               return_value=_StubInference(exc=TimeoutError())):
        assert await _classify_inner("anything") is None


@pytest.mark.asyncio
async def test_classify_timeout_wrapper_returns_none():
    async def _hang(**kwargs):
        import asyncio
        await asyncio.sleep(10)

    stub = _StubInference()
    stub.generate_response = _hang
    with patch("app.services.inference.get_inference_service",
               return_value=stub), \
         patch("app.services.intent.get_settings") as gs:
        gs.return_value.intent_classify_timeout = 0.05
        gs.return_value.intent_classify_max_tokens = 4
        assert await _classify_inner("anything") is None
