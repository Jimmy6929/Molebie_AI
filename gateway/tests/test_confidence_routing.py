"""Tests for confidence-driven prompt routing in chat.py.

Locks in the contract introduced to fix the four user-visible regressions on
``feature/hallucination-mitigation`` (see plan file
``so-after-the-last-jolly-thimble.md``):

  1. Tool-call denial — model said "I cannot directly search the live web"
     even though web_search was available. Fix: advertise TOOLS AVAILABLE
     in the system prompt when tool calling is on.
  2. ``<think>`` block leakage — fixed via ``_finalize_assistant_text``
     chokepoint that closes orphan openers + strips closed blocks.
  3. Mechanical citations on casual turns — fixed by making
     ``_build_evidence_summary`` return "" on no evidence (was returning
     a "No web results..." block that primed the model to use [S#]/[W#]
     tags even when nothing was retrieved).
  4. Over-refusal on general knowledge — fixed by dropping the REFUSE
     confidence tier entirely (low scores route to LOW/lookup or NONE/
     generative silently) and removing the literal "REFUSE" string from
     all directive text.
"""

import pytest

from app.routes.chat import (
    _build_evidence_summary,
    _build_system_message,
    _confidence_directive,
    _explicit_lookup_phrasing,
    _finalize_assistant_text,
    _routing_mode,
)
from app.services.rag import compute_retrieval_confidence

# ── _explicit_lookup_phrasing ───────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "what did I write about RAG?",
    "in my notes, where is the section on auth?",
    "did I mention the deadline anywhere?",
    "find the note about web search",
    "from my notes, summarize the API",
    "Did I write about this?",
    "find notes about onboarding",
])
def test_explicit_lookup_phrasing_matches(message):
    assert _explicit_lookup_phrasing(message) is True


@pytest.mark.parametrize("message", [
    "how can I add agentic function to molebie ai?",
    "what is molebie ai?",
    "hi",
    "brainstorm with me",
    "can you search online?",
    "explain how transformers work",
    "",
])
def test_explicit_lookup_phrasing_does_not_match_generative(message):
    assert _explicit_lookup_phrasing(message) is False


# ── _routing_mode ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("confidence,expected", [
    ("HIGH", "lookup"),
    ("MODERATE", "lookup"),
    # LOW now routes to generative — strict grounding on weak retrievals
    # produced over-refusal and "cite anything to satisfy the rule"
    # behaviour. The LOW+generative directive in _confidence_directive
    # tells the model to use the chunks as background instead.
    ("LOW", "generative"),
    ("NONE", "generative"),
])
def test_routing_mode_by_confidence(confidence, expected):
    assert _routing_mode(confidence, explicit_lookup=False) == expected


@pytest.mark.parametrize("confidence", ["HIGH", "MODERATE", "LOW", "NONE"])
def test_explicit_lookup_overrides_into_lookup(confidence):
    """Explicit 'in my notes' phrasing forces lookup mode regardless of
    retrieval confidence — so the strict-grounding template fires when
    the user genuinely wanted notes-lookup but the retrieval missed."""
    assert _routing_mode(confidence, explicit_lookup=True) == "lookup"


# ── _build_system_message ────────────────────────────────────────────────────

def test_build_system_message_picks_generative_on_none():
    msg = _build_system_message(
        confidence="NONE",
        explicit_lookup=False,
        evidence=None,
    )
    # Permissive marker
    assert "Default is to answer" in msg["content"]
    # No legacy abstain rule, no leftover REFUSE vocabulary leaking from the
    # old _confidence_directive. (The substring "I cannot" appears in the
    # prompt body inside a "do NOT refuse" instruction — that's intentional;
    # we check the actual abstain-string instead.)
    assert "I don't have that in your notes" not in msg["content"]
    assert "REFUSE" not in msg["content"]
    assert "RETRIEVAL CONFIDENCE" not in msg["content"]


def test_build_system_message_picks_rag_on_high():
    msg = _build_system_message(
        confidence="HIGH",
        explicit_lookup=False,
        evidence="[S1] (file: x.md)\nsome content",
    )
    # Strict-grounding marker (cite-when-used rule)
    assert "CITE WHEN USED" in msg["content"]
    assert "[S1]" in msg["content"]


def test_build_system_message_explicit_lookup_overrides_to_rag():
    msg = _build_system_message(
        confidence="NONE",
        explicit_lookup=True,
        evidence="[S1] (file: x.md)\nsome content",
    )
    assert "CITE WHEN USED" in msg["content"]


def test_build_system_message_voice_mode_unaffected():
    msg = _build_system_message(
        conversation_mode=True,
        confidence="NONE",
        explicit_lookup=False,
    )
    # Voice mode bypasses the routing — no strict-grounding rules.
    assert "CITE WHEN USED" not in msg["content"]
    # And no tools hint (voice mode has no tool calling).
    assert "TOOLS AVAILABLE" not in msg["content"]


# ── Tools-hint advertising (regression #1: "I cannot search the web") ───────

# A unique substring of the tools hint block — not present in the base
# prompt body — used to distinguish "hint present" from "hint absent" in
# system messages. (The literal "TOOLS AVAILABLE" appears in the generative
# prompt body inside a "see TOOLS AVAILABLE when present" instruction, so
# we can't use that as the marker.)
_TOOLS_HINT_MARKER = "web_search (live web)"


def test_tools_hint_present_when_tool_calling_enabled():
    msg = _build_system_message(
        confidence="NONE",
        explicit_lookup=False,
        use_tool_calling=True,
    )
    assert _TOOLS_HINT_MARKER in msg["content"]


def test_tools_hint_absent_when_tool_calling_disabled():
    msg = _build_system_message(
        confidence="NONE",
        explicit_lookup=False,
        use_tool_calling=False,
    )
    assert _TOOLS_HINT_MARKER not in msg["content"]


def test_tools_hint_absent_in_voice_mode_even_when_flag_set():
    """Voice mode must never advertise tools — there is no tool loop on the
    voice path, and saying tools exist when they don't is worse than silence."""
    msg = _build_system_message(
        conversation_mode=True,
        confidence="NONE",
        use_tool_calling=True,
    )
    assert _TOOLS_HINT_MARKER not in msg["content"]


# ── _confidence_directive — NONE-in-generative behaviour ────────────────────

def test_confidence_directive_none_in_generative_offers_search_when_factual():
    """Generative mode + NONE + tools enabled + factual query → directive
    nudges the model to offer a web search before answering."""
    directive = _confidence_directive(
        "NONE", chunks=[], mode="generative",
        use_tool_calling=True, factual_query=True,
    )
    assert directive is not None
    assert "offer to search" in directive.lower()
    assert "REFUSE" not in directive
    assert "I cannot" not in directive


def test_confidence_directive_none_in_generative_no_directive_when_casual():
    """Casual / non-factual turns get no directive — the system prompt
    handles 'answer directly' on its own; an extra directive on a greeting
    just primes weird responses."""
    directive = _confidence_directive(
        "NONE", chunks=[], mode="generative",
        use_tool_calling=True, factual_query=False,
    )
    assert directive is None


def test_confidence_directive_none_in_generative_no_search_offer_when_tools_off():
    """If tools aren't available, don't tell the model to offer a search it
    can't deliver — fall through to the system prompt's general guidance."""
    directive = _confidence_directive(
        "NONE", chunks=[], mode="generative",
        use_tool_calling=False, factual_query=True,
    )
    assert directive is None


def test_confidence_directive_low_active_in_both_modes():
    """LOW now produces a directive in both modes, but with different
    intent: lookup mode (only reachable via explicit-lookup override)
    keeps the strict cite-when-used reminder; generative mode (the new
    default for LOW) tells the model to use chunks as background and
    fall back to general knowledge."""
    lookup_directive = _confidence_directive("LOW", chunks=[], mode="lookup")
    generative_directive = _confidence_directive("LOW", chunks=[], mode="generative")
    assert lookup_directive is not None
    assert generative_directive is not None
    assert lookup_directive != generative_directive
    assert "background" in generative_directive.lower()
    assert "general knowledge" in generative_directive.lower()


def test_confidence_directive_low_in_lookup_uses_cite_when_used_language():
    """The new LOW directive must NOT contain the legacy "REFUSE" vocabulary
    or the literal "RETRIEVAL CONFIDENCE:" header — both were priming the
    model to mimic refusal."""
    directive = _confidence_directive("LOW", chunks=[], mode="lookup")
    assert directive is not None
    assert "REFUSE" not in directive
    assert "RETRIEVAL CONFIDENCE" not in directive
    assert "weak" in directive.lower()


def test_confidence_directive_high_no_directive():
    assert _confidence_directive("HIGH", chunks=[], mode="lookup") is None


# ── _finalize_assistant_text (regression #2: <think> leakage) ───────────────

def test_finalize_assistant_text_closes_orphan_think():
    """Truncation at max_tokens can leave an unclosed `<think>` opener — the
    chokepoint must close it so `_strip_thinking`'s regex can erase it."""
    assert _finalize_assistant_text("<think>cut mid-thought") == ""


def test_finalize_assistant_text_strips_closed_think():
    assert _finalize_assistant_text("<think>x</think>visible") == "visible"


def test_finalize_assistant_text_passes_through_clean_text():
    assert _finalize_assistant_text("plain answer") == "plain answer"


def test_finalize_assistant_text_handles_empty():
    assert _finalize_assistant_text("") == ""
    assert _finalize_assistant_text(None) == ""  # type: ignore[arg-type]


def test_finalize_assistant_text_idempotent():
    """Safe to call twice — running through twice yields the same result."""
    once = _finalize_assistant_text("<think>x</think>visible")
    twice = _finalize_assistant_text(once)
    assert once == twice == "visible"


# ── _build_evidence_summary (regression #3: mechanical citations) ───────────

def test_evidence_summary_omitted_when_no_evidence():
    """Empty in → empty out. Previously this returned "EVIDENCE META: No web
    results. No document matches. No attachments." which primed the model to
    cite [S#]/[W#] tags even on casual turns."""
    assert _build_evidence_summary([], []) == ""


def test_evidence_summary_present_when_evidence_exists():
    summary = _build_evidence_summary(
        [], [{"similarity": 0.8, "rerank_score": 0.7}],
    )
    assert summary != ""
    assert "AVAILABLE EVIDENCE" in summary


def test_system_message_for_casual_turn_has_no_evidence_meta():
    """End-to-end: a casual turn (no docs, no web) must produce a system
    message free of the EVIDENCE META block and citation-tag vocabulary
    in that block (the prompt template itself still mentions [S#] in its
    rules — that's fine)."""
    msg = _build_system_message(
        confidence="NONE",
        explicit_lookup=False,
        evidence=None,
    )
    # Sanity: the EVIDENCE META block (now removed) used to contain these
    # exact substrings on empty evidence. They must not appear here.
    assert "EVIDENCE META" not in msg["content"]
    assert "No web results" not in msg["content"]
    assert "No document matches" not in msg["content"]


# ── compute_retrieval_confidence (regression #4: over-refusal) ──────────────

def test_compute_retrieval_confidence_no_refuse_tier():
    """REFUSE was removed — weak hits route to LOW (lookup mode) so the
    chunks are visible to the model under cite-when-used; the model decides
    whether to use them."""
    assert compute_retrieval_confidence([{"rerank_score": 0.25}]) == "LOW"


def test_compute_retrieval_confidence_empty_returns_none():
    assert compute_retrieval_confidence([]) == "NONE"


def test_compute_retrieval_confidence_high_requires_strong_signal():
    chunks = [
        {"rerank_score": 0.8},
        {"rerank_score": 0.7},
        {"rerank_score": 0.6},
    ]
    assert compute_retrieval_confidence(chunks) == "HIGH"


def test_compute_retrieval_confidence_moderate_on_mid_signal():
    chunks = [
        {"rerank_score": 0.55},
        {"rerank_score": 0.5},
    ]
    assert compute_retrieval_confidence(chunks) == "MODERATE"


def test_compute_retrieval_confidence_never_returns_refuse():
    """Belt and braces: under no realistic input shape should the function
    return the legacy REFUSE tier."""
    for chunks in [
        [{"rerank_score": 0.0}],
        [{"rerank_score": 0.1}],
        [{"rerank_score": 0.29}],
        [{"similarity": 0.05}],
    ]:
        assert compute_retrieval_confidence(chunks) != "REFUSE"
