"""Tests for identity-query resolution (the "what do you know about me?" fix).

Self-referential queries ("who am I", "what do you know about me?") are
pronouns + stopwords with no retrievable anchors — neither vector nor BM25
connects "me" to the owner, and the keyword leg can match noise (a note
titled "...Me", e.g. "Can't Hurt Me"). These tests lock in:

  1. ``is_identity_query`` — deterministic detection (no LLM).
  2. ``_retrieval_query`` — expands the *retrieval* text (not the user message)
     with the owner's name + biographical terms when an identity query is
     detected, and is a no-op otherwise.
  3. ``_build_system_message`` — injects the owner card when ``owner_profile``
     is set so the model knows "I"/"me"/"my" refers to that person.
"""

from types import SimpleNamespace

from app.routes.chat import _build_system_message, _retrieval_query
from app.services.intent import is_identity_query


def _settings(expand=True, owner_name="Jimmy Chu"):
    return SimpleNamespace(
        rag_identity_expansion_enabled=expand,
        owner_name=owner_name,
    )


# ── is_identity_query ──────────────────────────────────────────────────────

def test_identity_query_positive_cases():
    for q in [
        "what do you know about me?",
        "so what do you know about me?",
        "who am I",
        "tell me about myself",
        "what's my background",
        "my skills and experience",
        "describe me",
    ]:
        assert is_identity_query(q), q


def test_identity_query_negative_cases():
    for q in [
        "what is the capital of France",
        "explain how RAG works",
        "summarize the meeting notes",
        "remind me to buy milk",   # "remind me to" is not "remind me about myself"
        "",
    ]:
        assert not is_identity_query(q), q


# ── _retrieval_query expansion ─────────────────────────────────────────────

def test_expands_identity_query_with_owner_name():
    out = _retrieval_query("what do you know about me?", _settings())
    assert out != "what do you know about me?"
    assert "Jimmy Chu" in out
    assert "background" in out and "experience" in out
    # original message is preserved inside the expanded query
    assert "what do you know about me?" in out


def test_expansion_without_owner_name_still_adds_bio_terms():
    out = _retrieval_query("who am I", _settings(owner_name=""))
    assert "biography" in out and "profile" in out
    assert "who am I" in out


def test_non_identity_query_is_untouched():
    q = "what is the capital of France"
    assert _retrieval_query(q, _settings()) == q


def test_expansion_disabled_is_noop():
    q = "what do you know about me?"
    assert _retrieval_query(q, _settings(expand=False)) == q


# ── owner card injection ───────────────────────────────────────────────────

def test_owner_profile_injected_into_system_message():
    msg = _build_system_message(
        confidence="NONE",
        owner_profile="Jimmy Chu — AI product builder.",
    )
    content = msg["content"]
    assert "OWNER" in content
    assert "Jimmy Chu — AI product builder." in content
    # the disambiguation hint must be present so the model resolves pronouns
    assert '"me"' in content and '"my"' in content


def test_owner_profile_absent_when_unset():
    msg = _build_system_message(confidence="NONE", owner_profile=None)
    assert "OWNER (the person you are assisting" not in msg["content"]
