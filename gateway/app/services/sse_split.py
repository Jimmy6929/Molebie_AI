"""Backend-agnostic SSE chunk shape-normalizer.

Some inference backends (notably mlx_vlm under certain payload shapes)
buffer the entire response and emit it as one or two large SSE deltas at
end-of-generation. The frontend typewriter then has nothing to pace and
the user sees a "pop" instead of a character-by-character reveal.

Splitting oversized deltas at the gateway makes the frontend always see
fine-grained chunks regardless of how the upstream backend chunked. No
artificial delays — just shape normalization. Fully backend-agnostic:
this helper has zero knowledge of which backend produced the line.
"""

from __future__ import annotations

import json
from typing import Any


# When a slice ends with ``<`` or ``</`` we extend it by up to this many
# chars to consume the rest of the tag. Keeps ``</think>`` intact through
# splitting so the downstream parser doesn't see ``</thi`` then ``nk>``.
_TAG_STRADDLE_LOOKAHEAD = 8


def _slice_text(text: str, max_chars: int) -> list[str]:
    """Slice ``text`` into pieces of at most ``max_chars`` codepoints.

    If a piece's boundary lands inside a ``<...>`` tag (final char is
    ``<`` or ``</``), extend the piece by up to _TAG_STRADDLE_LOOKAHEAD
    chars so the tag is delivered atomically.
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        # Tag-straddle defence: if we'd cut the start of a tag, extend
        # until we close the tag or hit the lookahead budget.
        if end < n:
            tail = text[max(end - 2, i):end]
            if tail.endswith("<") or tail.endswith("</"):
                limit = min(end + _TAG_STRADDLE_LOOKAHEAD, n)
                close = text.find(">", end, limit)
                if close != -1:
                    end = close + 1
        pieces.append(text[i:end])
        i = end
    return pieces


def _envelope_for_piece(
    base: dict[str, Any],
    delta: dict[str, Any],
    *,
    is_first: bool,
    is_last: bool,
    finish_reason: Any,
) -> dict[str, Any]:
    """Build a per-piece SSE envelope from the base chunk dict.

    First piece keeps the original ``id`` / ``role`` / etc. unchanged
    (since they were already on ``base``). Subsequent pieces are clones
    with delta replaced. Only the last piece keeps any non-None
    ``finish_reason`` from the original — intermediate pieces must NOT
    claim the stream is finished.
    """
    new_choices = []
    for choice in base.get("choices", []):
        new_choice = {**choice, "delta": delta}
        # Strip role from non-first pieces — OpenAI streams put role only
        # on the first content delta. Sending it again can confuse some
        # client parsers that key on its presence.
        if not is_first and "role" in delta:
            new_delta = {k: v for k, v in delta.items() if k != "role"}
            new_choice["delta"] = new_delta
        new_choice["finish_reason"] = finish_reason if is_last else None
        new_choices.append(new_choice)
    return {**base, "choices": new_choices}


def split_oversized_sse_delta(line: str, max_chars: int) -> list[str]:
    """Split one ``data: {...}\\n\\n`` SSE line whose ``delta.content`` (or
    ``delta.reasoning_content``) exceeds ``max_chars`` into N smaller lines.

    Passthrough cases (return ``[line]`` unchanged):
      - Empty / non-``data:`` lines
      - ``data: [DONE]`` sentinel
      - Malformed JSON body
      - Non-choice envelopes (e.g. ``{"metadata": {...}}``,
        ``{"session_id": "..."}``, ``{"type": "search_start"}``)
      - Choices present but ``delta.content`` and ``delta.reasoning_content``
        both ≤ ``max_chars`` (or both absent)

    When splitting, both ``content`` and ``reasoning_content`` are sliced
    independently. If both are present, they are zipped piece-by-piece;
    the longer field's tail pieces carry the remainder solo.

    The function is pure and idempotent on already-small lines.
    """
    if max_chars <= 0:
        return [line]
    stripped = line.rstrip()
    if not stripped.startswith("data: "):
        return [line]
    body = stripped[len("data: "):].strip()
    if not body or body == "[DONE]":
        return [line]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return [line]
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices, list):
        return [line]

    # Read content + reasoning_content from the first choice's delta —
    # OpenAI streams emit one choice per delta in chat-completion mode.
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return [line]
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    content_oversized = isinstance(content, str) and len(content) > max_chars
    reasoning_oversized = isinstance(reasoning, str) and len(reasoning) > max_chars
    if not content_oversized and not reasoning_oversized:
        return [line]

    content_pieces = (
        _slice_text(content, max_chars) if content_oversized
        else ([content] if isinstance(content, str) else [])
    )
    reasoning_pieces = (
        _slice_text(reasoning, max_chars) if reasoning_oversized
        else ([reasoning] if isinstance(reasoning, str) else [])
    )
    n_pieces = max(len(content_pieces), len(reasoning_pieces))

    # Keep the original finish_reason for the final piece only.
    finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None

    out: list[str] = []
    for idx in range(n_pieces):
        piece_delta: dict[str, Any] = {}
        # Carry forward any non-content/non-reasoning fields on the first
        # piece (e.g. role marker) — they belong only to the leading frame.
        if idx == 0:
            for k, v in delta.items():
                if k not in ("content", "reasoning_content"):
                    piece_delta[k] = v
        if idx < len(content_pieces):
            piece_delta["content"] = content_pieces[idx]
        if idx < len(reasoning_pieces):
            piece_delta["reasoning_content"] = reasoning_pieces[idx]
        new_data = _envelope_for_piece(
            data, piece_delta,
            is_first=(idx == 0),
            is_last=(idx == n_pieces - 1),
            finish_reason=finish_reason,
        )
        out.append(f"data: {json.dumps(new_data)}\n\n")
    return out
