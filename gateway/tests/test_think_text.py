"""Tests for the linear <think>-block helpers that replaced the ReDoS regex.

`strip_think_blocks` / `first_think_inner` must be byte-for-byte equivalent to
the old `re.compile(r"<think>.*?</think>", re.DOTALL)` (.sub / .search) — which
was O(n^2) on unclosed tags (CodeQL py/polynomial-redos) — while running linear.
"""

import re
import time

from app.services.streaming_think_filter import first_think_inner, strip_think_blocks

# The exact pattern the helpers replace (kept here only to assert equivalence).
_LEGACY = re.compile(r"<think>.*?</think>", re.DOTALL)

CASES = [
    "<think>hello</think>world",
    "<think>a\nb\nc</think>X",                       # DOTALL across newlines
    "<think>one</think>mid<think>two</think>end",    # multiple blocks
    "<think></think>empty",                          # empty body
    "no tags here",
    "leading</think>tail",                           # </think>-only
    "<think>unclosed forever",
    "pre<think>x</think>post<think>y",               # trailing unclosed
    "<think>a<think>nested</think>b</think>c",        # lazy = first close
    "",
]


def test_strip_matches_legacy_regex_sub():
    for c in CASES:
        assert strip_think_blocks(c) == _LEGACY.sub("", c), repr(c)


def test_first_inner_matches_legacy_regex_search():
    for c in CASES:
        m = _LEGACY.search(c)
        expected = c[m.start() + len("<think>"):m.end() - len("</think>")] if m else None
        assert first_think_inner(c) == expected, repr(c)


def test_linear_on_pathological_unclosed_input():
    # The old regex took ~20s on this; the linear scan is instant. A generous
    # bound that the regex could never meet but linear always does — also a
    # regression guard if anyone reintroduces a backtracking regex here.
    payload = "<think>" * 20000
    t = time.perf_counter()
    out = strip_think_blocks(payload)
    assert out == payload  # no complete block -> unchanged
    assert (time.perf_counter() - t) < 1.0
