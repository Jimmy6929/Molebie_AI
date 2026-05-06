"""Streaming-aware <think>...</think> block filter.

The non-streaming `_strip_thinking()` regex in chat.py works on a complete
buffer; it can't be applied to SSE deltas because tags can split across
chunks (e.g. one delta ends with `<thi`, the next starts with `nk>`).

This module is the streaming counterpart: a small state machine that
consumes deltas one at a time and returns (visible_text, reasoning_text)
with all `<think>...</think>` blocks removed from visible_text. Partial
tags at chunk boundaries are buffered into `_pending` and resolved when
more text arrives. `flush()` drains the buffer at stream end.

The captured reasoning is returned to the caller so it can still be
persisted to the DB row, matching what `_finalize_assistant_text` does
on the non-streaming path.
"""

from __future__ import annotations

_OPEN = "<think>"
_CLOSE = "</think>"


class ThinkBlockFilter:
    """State machine for stripping <think>...</think> from a streamed text.

    Modes:
      NORMAL   — emit text as visible until an opening tag is seen.
      IN_THINK — capture text as reasoning until a closing tag is seen.

    Boundary handling: at most 7 characters (the length of `</think` minus 1)
    may be held in `_pending` waiting to disambiguate a possible partial tag.
    """

    __slots__ = ("_mode", "_pending")

    def __init__(self) -> None:
        self._mode = "NORMAL"
        self._pending = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Consume a chunk of streamed content. Returns (visible, reasoning).

        Either component may be empty. Partial-tag tails are stashed and
        not emitted until more text arrives or `flush()` is called.
        """
        if not text and not self._pending:
            return "", ""

        buf = self._pending + text
        self._pending = ""
        visible: list[str] = []
        reasoning: list[str] = []
        i = 0

        while i < len(buf):
            if self._mode == "NORMAL":
                j = buf.find(_OPEN, i)
                if j == -1:
                    # No full opener — check for a partial at the tail so we
                    # don't emit "<thi" as visible text only to discover next
                    # chunk it was the start of "<think>".
                    tail = buf[i:]
                    p = _partial_suffix_len(tail, _OPEN)
                    if p > 0:
                        visible.append(tail[: len(tail) - p])
                        self._pending = tail[len(tail) - p :]
                    else:
                        visible.append(tail)
                    break
                visible.append(buf[i:j])
                i = j + len(_OPEN)
                self._mode = "IN_THINK"
            else:  # IN_THINK
                j = buf.find(_CLOSE, i)
                if j == -1:
                    tail = buf[i:]
                    p = _partial_suffix_len(tail, _CLOSE)
                    if p > 0:
                        reasoning.append(tail[: len(tail) - p])
                        self._pending = tail[len(tail) - p :]
                    else:
                        reasoning.append(tail)
                    break
                reasoning.append(buf[i:j])
                i = j + len(_CLOSE)
                self._mode = "NORMAL"

        return "".join(visible), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        """Drain any buffered tail at stream end.

        If we are still IN_THINK (orphan opener — stream ended mid-think),
        the buffered tail is treated as reasoning and no visible text is
        emitted. This mirrors `_finalize_assistant_text`'s auto-close
        behaviour for the persistence path.
        """
        out = self._pending
        self._pending = ""
        if self._mode == "NORMAL":
            return out, ""
        return "", out

    @property
    def in_think(self) -> bool:
        return self._mode == "IN_THINK"


def _partial_suffix_len(tail: str, target: str) -> int:
    """Longest proper-prefix of `target` that is a suffix of `tail`.

    Used to detect `<thi`, `<think`, `</thi`, `</think` etc. at chunk
    boundaries. Returns 0 if none. Excludes the full target (those are
    handled by `find()` above) and the empty prefix.
    """
    upper = min(len(target) - 1, len(tail))
    for n in range(upper, 0, -1):
        if tail.endswith(target[:n]):
            return n
    return 0
