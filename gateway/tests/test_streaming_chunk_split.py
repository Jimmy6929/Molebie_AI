"""Tests for the gateway SSE chunk splitter (Layer A of the
backend-agnostic streaming plan).

The splitter takes one SSE ``data: {...}\\n\\n`` line from any
OpenAI-compatible backend and, if the embedded ``delta.content`` or
``delta.reasoning_content`` is oversized, fans it out into multiple
smaller lines. Everything else is passed through untouched. The function
is pure, has no awareness of which backend produced the line, and is
the single mechanism by which the frontend gets a fine-grained stream
regardless of how the upstream chunked.
"""

import json

from app.services.sse_split import split_oversized_sse_delta


def _build_data_line(delta: dict, finish_reason=None, extra=None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if extra:
        payload.update(extra)
    return f"data: {json.dumps(payload)}\n\n"


def _content_of(line: str) -> str:
    body = line[len("data: "):].strip()
    return json.loads(body)["choices"][0]["delta"].get("content", "")


def _reasoning_of(line: str) -> str:
    body = line[len("data: "):].strip()
    return json.loads(body)["choices"][0]["delta"].get("reasoning_content", "")


def _finish_of(line: str):
    body = line[len("data: "):].strip()
    return json.loads(body)["choices"][0]["finish_reason"]


# ── Passthrough cases ───────────────────────────────────────────────────────

def test_below_threshold_passthrough():
    line = _build_data_line({"content": "hi"})
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_exact_threshold_passthrough():
    line = _build_data_line({"content": "x" * 40})
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_done_marker_passthrough():
    line = "data: [DONE]\n\n"
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_metadata_envelope_passthrough():
    """Non-choice envelopes (gateway emits these for first-token mode info)
    must never be split — they have no content to split."""
    line = f'data: {json.dumps({"metadata": {"mode": "instant", "model": "x"}})}\n\n'
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_session_id_envelope_passthrough():
    line = f'data: {json.dumps({"session_id": "abc-123"})}\n\n'
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_search_start_envelope_passthrough():
    line = f'data: {json.dumps({"type": "search_start"})}\n\n'
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_malformed_json_passthrough_no_exception():
    line = "data: {not valid json\n\n"
    # Must not raise — preserve resilience to upstream weirdness.
    assert split_oversized_sse_delta(line, max_chars=40) == [line]


def test_non_data_line_passthrough():
    """Anything not prefixed with `data: ` is passed through unchanged
    (e.g. SSE comments, blank keep-alives)."""
    assert split_oversized_sse_delta(": keep-alive\n\n", max_chars=40) == [
        ": keep-alive\n\n"
    ]


def test_zero_max_chars_disables_splitting():
    """Defensive: max_chars=0 (or negative) is a no-op kill-switch path."""
    line = _build_data_line({"content": "x" * 200})
    assert split_oversized_sse_delta(line, max_chars=0) == [line]


# ── Splitting cases ─────────────────────────────────────────────────────────

def test_two_x_threshold_splits_to_two_pieces():
    line = _build_data_line({"content": "x" * 80})
    out = split_oversized_sse_delta(line, max_chars=40)
    assert len(out) == 2
    assert _content_of(out[0]) == "x" * 40
    assert _content_of(out[1]) == "x" * 40


def test_five_x_threshold_splits_to_five_pieces_concat_reproduces():
    text = "abcdefghij" * 20  # 200 chars
    line = _build_data_line({"content": text})
    out = split_oversized_sse_delta(line, max_chars=40)
    assert len(out) == 5
    rebuilt = "".join(_content_of(piece) for piece in out)
    assert rebuilt == text


def test_finish_reason_only_on_last_piece():
    """Intermediate pieces must NOT claim the stream is finished —
    a frontend that keys on finish_reason would close the connection
    halfway through."""
    line = _build_data_line({"content": "x" * 200}, finish_reason="stop")
    out = split_oversized_sse_delta(line, max_chars=40)
    assert len(out) == 5
    for piece in out[:-1]:
        assert _finish_of(piece) is None
    assert _finish_of(out[-1]) == "stop"


def test_role_only_on_first_piece():
    """Role marker (e.g. 'assistant') belongs on the leading frame only;
    parsers that re-set role on every chunk get confused otherwise."""
    line = _build_data_line({"role": "assistant", "content": "x" * 100})
    out = split_oversized_sse_delta(line, max_chars=40)
    body0 = json.loads(out[0][len("data: "):].strip())
    assert body0["choices"][0]["delta"].get("role") == "assistant"
    for piece in out[1:]:
        body = json.loads(piece[len("data: "):].strip())
        assert "role" not in body["choices"][0]["delta"]


def test_oversized_reasoning_content_splits():
    """Format-A streams (vLLM with --enable-reasoning) carry reasoning
    in delta.reasoning_content. It must be split too."""
    line = _build_data_line({"reasoning_content": "y" * 120})
    out = split_oversized_sse_delta(line, max_chars=40)
    assert len(out) == 3
    rebuilt = "".join(_reasoning_of(piece) for piece in out)
    assert rebuilt == "y" * 120


def test_both_content_and_reasoning_oversized():
    """When both fields are oversized, both get split independently and
    paired piece-by-piece."""
    line = _build_data_line({
        "content": "a" * 100,
        "reasoning_content": "b" * 100,
    })
    out = split_oversized_sse_delta(line, max_chars=40)
    assert len(out) == 3  # max(ceil(100/40), ceil(100/40)) = 3
    assert "".join(_content_of(p) for p in out) == "a" * 100
    assert "".join(_reasoning_of(p) for p in out) == "b" * 100


def test_tag_straddle_keeps_close_tag_intact():
    """If a slice would end mid-tag (e.g., '...foo</' followed by 'think>'),
    extend the slice to consume the rest of the tag. Otherwise the
    downstream parser sees a half-tag and may render literal '</thi'
    in the message bubble before the rest arrives."""
    text = "padpadpa</think>tail"  # 21 chars; '</' lands at char 8 with max_chars=8
    line = _build_data_line({"content": text})
    out = split_oversized_sse_delta(line, max_chars=8)
    # Reconstruct must be lossless.
    assert "".join(_content_of(p) for p in out) == text
    # No piece may contain a half-tag — every '</' that appears must be
    # followed (in the SAME piece) by the closing '>'.
    for piece in out:
        c = _content_of(piece)
        idx = 0
        while True:
            idx = c.find("</", idx)
            if idx == -1:
                break
            close = c.find(">", idx)
            assert close != -1, f"half-tag in piece: {c!r}"
            idx = close + 1


def test_unicode_multibyte_safe():
    """Slicing must not corrupt multi-codepoint characters. Python str
    indexes by codepoint, so this is structurally safe — but we lock in
    the property with a test."""
    text = "🎉" * 50  # 50 codepoints, but 4 bytes each in UTF-8
    line = _build_data_line({"content": text})
    out = split_oversized_sse_delta(line, max_chars=10)
    assert len(out) == 5
    rebuilt = "".join(_content_of(piece) for piece in out)
    assert rebuilt == text
    # Each piece must round-trip through JSON (proves no surrogate damage).
    for piece in out:
        json.loads(piece[len("data: "):].strip())


def test_envelope_id_preserved_on_all_pieces():
    """Every emitted piece must carry the original envelope id so a
    frontend that correlates by id keeps tracking the same stream."""
    line = _build_data_line({"content": "x" * 100})
    out = split_oversized_sse_delta(line, max_chars=40)
    ids = [json.loads(p[len("data: "):].strip()).get("id") for p in out]
    assert all(i == "chatcmpl-test" for i in ids)


def test_empty_delta_passthrough():
    """A delta with neither content nor reasoning_content (e.g. a
    keep-alive frame) is passed through unchanged."""
    line = _build_data_line({})
    assert split_oversized_sse_delta(line, max_chars=40) == [line]
