"""Tests for the streaming-aware <think>...</think> filter.

The filter strips reasoning blocks from streamed delta.content while
preserving partial-tag tails across chunk boundaries. The non-streaming
counterpart (`_strip_thinking()` in chat.py) only sees a complete buffer
and can't be applied chunk-by-chunk.
"""

from app.services.streaming_think_filter import ThinkBlockFilter


def test_single_block_in_one_chunk():
    f = ThinkBlockFilter()
    visible, reasoning = f.feed("hello <think>secret</think> world")
    assert visible == "hello  world"
    assert reasoning == "secret"


def test_tag_split_across_chunks():
    f = ThinkBlockFilter()
    v1, r1 = f.feed("hello <thi")
    v2, r2 = f.feed("nk>secret</thi")
    v3, r3 = f.feed("nk> world")
    assert (v1, r1) == ("hello ", "")
    assert (v2, r2) == ("", "secret")
    assert (v3, r3) == (" world", "")


def test_multiple_blocks_in_one_chunk():
    f = ThinkBlockFilter()
    visible, reasoning = f.feed("A<think>x</think>B<think>y</think>C")
    assert visible == "ABC"
    assert reasoning == "xy"


def test_orphan_opener_stream_truncated():
    """Stream ends mid-think (no closing tag) — buffered reasoning is
    captured for DB persistence, no visible text leaks."""
    f = ThinkBlockFilter()
    visible, reasoning = f.feed("hello <think>thinking but never closed")
    visible_tail, reasoning_tail = f.flush()
    assert visible == "hello "
    assert reasoning == "thinking but never closed"
    assert (visible_tail, reasoning_tail) == ("", "")


def test_plain_less_than_does_not_get_held_forever():
    f = ThinkBlockFilter()
    visible, reasoning = f.feed("x < y")
    assert visible == "x < y"
    assert reasoning == ""


def test_false_positive_partial_buffered_then_released():
    """A trailing '<' might be the start of '<think>' — buffer it. When
    the next chunk proves it isn't, flush and emit."""
    f = ThinkBlockFilter()
    v1, r1 = f.feed("a<")
    v2, r2 = f.feed("b")
    assert (v1, r1) == ("a", "")
    assert (v2, r2) == ("<b", "")


def test_empty_input():
    f = ThinkBlockFilter()
    visible, reasoning = f.feed("")
    assert visible == ""
    assert reasoning == ""


def test_whole_tag_chunks_qwen3_pattern():
    """Common Qwen3 streaming pattern: opener, reasoning body, closer,
    visible answer all arrive as separate deltas."""
    f = ThinkBlockFilter()
    v1, r1 = f.feed("<think>")
    v2, r2 = f.feed("reasoning here")
    v3, r3 = f.feed("</think>")
    v4, r4 = f.feed(" visible answer")
    assert (v1, r1) == ("", "")
    assert (v2, r2) == ("", "reasoning here")
    assert (v3, r3) == ("", "")
    assert (v4, r4) == (" visible answer", "")


def test_close_tag_split_at_angle_bracket():
    """The exact '<' or '>' character of a closing tag splits chunks."""
    f = ThinkBlockFilter()
    v1, r1 = f.feed("<think>inner<")
    v2, r2 = f.feed("/think>after")
    assert (v1, r1) == ("", "inner")
    assert (v2, r2) == ("after", "")


def test_consecutive_blocks_split_across_chunks():
    """Reasoning text from the first block and the partial body of the
    second block are both captured in chunk 1's feed; only the trailing
    text after the second close tag arrives in chunk 2."""
    f = ThinkBlockFilter()
    v1, r1 = f.feed("A<think>x</think>B<think>y")
    v2, r2 = f.feed("</think>C")
    assert (v1, r1) == ("AB", "xy")
    assert (v2, r2) == ("C", "")


def test_in_think_property():
    f = ThinkBlockFilter()
    assert f.in_think is False
    f.feed("<think>partial")
    assert f.in_think is True
    f.feed("</think>")
    assert f.in_think is False
