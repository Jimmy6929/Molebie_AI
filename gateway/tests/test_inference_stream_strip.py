"""Tests for the inference-layer streaming think-block strip.

This filter operates on raw SSE ``data: {...}`` lines (different layer
from ``streaming_think_filter.ThinkBlockFilter`` which works on the
visible/reasoning split inside the gateway). It runs only when the
gateway requested ``enable_thinking=False`` but the backend leaked a
``<think>...</think>`` block anyway (mlx_vlm + Qwen 3.5 Format C).
"""

import json

import pytest

from app.services.inference import _StreamThinkStripper, _strip_think_async

# ── _StreamThinkStripper unit tests ──────────────────────────────────


def test_format_c_drops_pre_close_then_passes_through():
    """Content + </think> + answer — drop everything up to and including the close tag."""
    s = _StreamThinkStripper()
    assert s.filter_content("reasoning step 1\n") == ""
    assert s.filter_content("reasoning step 2\n") == ""
    out = s.filter_content("done</think>The answer is 42.")
    assert out == "The answer is 42."
    # Subsequent chunks pass through
    assert s.filter_content(" Continue.") == " Continue."


def test_close_tag_straddling_chunks():
    s = _StreamThinkStripper()
    assert s.filter_content("thinking</thi") == ""
    out = s.filter_content("nk>final answer")
    assert out == "final answer"


def test_no_close_tag_flushes_on_stream_end():
    """Backend honored the flag (no think block); flush() must release the buffer."""
    s = _StreamThinkStripper()
    assert s.filter_content("hello world") == ""
    assert s.filter_content(" more") == ""
    assert s.flush() == "hello world more"
    # flush is idempotent
    assert s.flush() == ""


def test_empty_stream():
    s = _StreamThinkStripper()
    assert s.flush() == ""


def test_close_tag_with_trailing_whitespace_consumed():
    """The </think>\\s* regex eats trailing whitespace so the answer doesn't start with a newline."""
    s = _StreamThinkStripper()
    out = s.filter_content("thoughts</think>\n\nanswer here")
    assert out == "answer here"


def test_multiple_blocks_only_first_stripped():
    """We only strip the leading block; a stray later <think> would be passthrough.

    Acceptable: with enable_thinking=False the model should not emit nested
    blocks, and the leading-block strip handles the Format C case which is
    what the backend actually leaks."""
    s = _StreamThinkStripper()
    assert s.filter_content("first</think>real answer ") == "real answer "
    # A later stray tag passes through verbatim — caller can post-process if needed
    assert s.filter_content("<think>noise</think>tail") == "<think>noise</think>tail"


# ── _strip_think_async SSE-line wrapper tests ────────────────────────


@pytest.mark.asyncio
async def test_sse_wrapper_drops_pre_close_chunks():
    async def feed():
        yield 'data: {"choices":[{"delta":{"content":"thinking..."}}]}'
        yield 'data: {"choices":[{"delta":{"content":"</think>real "}}]}'
        yield 'data: {"choices":[{"delta":{"content":"answer"}}]}'
        yield 'data: [DONE]'

    out = []
    async for line in _strip_think_async(feed()):
        out.append(line)

    contents = []
    saw_done = False
    for ln in out:
        if ln == "data: [DONE]":
            saw_done = True
            continue
        assert ln.startswith("data: ")
        contents.append(json.loads(ln[6:])["choices"][0]["delta"]["content"])
    assert contents == ["real ", "answer"]
    assert saw_done


@pytest.mark.asyncio
async def test_sse_wrapper_flushes_buffer_on_done_when_no_close_tag():
    """If the backend honored enable_thinking=False (no leaked block), the
    filter must still emit the buffered content as a synthetic delta before
    [DONE] so the user sees the answer."""
    async def feed():
        yield 'data: {"choices":[{"delta":{"content":"clean "}}]}'
        yield 'data: {"choices":[{"delta":{"content":"answer"}}]}'
        yield 'data: [DONE]'

    out = []
    async for line in _strip_think_async(feed()):
        out.append(line)

    # The buffered "clean answer" should appear as a single synthetic delta
    # right before [DONE].
    assert out[-1] == "data: [DONE]"
    synthetic = json.loads(out[-2][6:])
    assert synthetic["choices"][0]["delta"]["content"] == "clean answer"


@pytest.mark.asyncio
async def test_sse_wrapper_passes_through_non_data_lines():
    async def feed():
        yield ': heartbeat'
        yield 'event: ping'
        yield 'data: {"choices":[{"delta":{"content":"</think>hi"}}]}'
        yield 'data: [DONE]'

    out = []
    async for line in _strip_think_async(feed()):
        out.append(line)
    assert ": heartbeat" in out
    assert "event: ping" in out


@pytest.mark.asyncio
async def test_sse_wrapper_ignores_malformed_json():
    """Don't crash on a busted line — pass it through and keep going."""
    async def feed():
        yield 'data: not-json{'
        yield 'data: {"choices":[{"delta":{"content":"</think>ok"}}]}'
        yield 'data: [DONE]'

    out = []
    async for line in _strip_think_async(feed()):
        out.append(line)
    assert "data: not-json{" in out
    parsed = [
        json.loads(ln[6:])["choices"][0]["delta"]["content"]
        for ln in out
        if ln.startswith("data: ") and ln != "data: [DONE]" and ln[6:].startswith("{")
    ]
    assert parsed == ["ok"]


@pytest.mark.asyncio
async def test_sse_wrapper_preserves_non_content_deltas():
    """Tool calls and finish_reason chunks (no content) must pass through untouched."""
    async def feed():
        yield 'data: {"choices":[{"delta":{"content":"</think>"}}]}'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        yield 'data: [DONE]'

    out = []
    async for line in _strip_think_async(feed()):
        out.append(line)
    # The finish_reason chunk passes through verbatim
    assert any('"finish_reason"' in ln for ln in out)
