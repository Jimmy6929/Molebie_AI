"""Tests for FTS5 stopword hygiene (`_build_fts_match`).

FTS5 has no built-in stopword support, so a bare query like "what do you know
about me" matched noise — "me" matched the note title "Can't Hurt Me". The
match-builder strips function words before constructing the quoted MATCH, while
never collapsing to an empty match.
"""

from app.services.database import _build_fts_match


def test_strips_function_words_keeps_content():
    # "what", "do", "you", "about", "me" are stopwords; "know" is content.
    assert _build_fts_match("what do you know about me") == '"know"'


def test_me_is_dropped_when_content_terms_exist():
    # The bug: "me" matched the note title "Can't Hurt Me". When the query has
    # real content terms, the stopwords ("me", "my", "about", "and") are
    # dropped from the MATCH entirely so they can't surface noise.
    out = _build_fts_match("notes about me and my projects")
    assert '"me"' not in out and '"my"' not in out
    assert '"notes"' in out and '"projects"' in out


def test_content_query_is_unchanged():
    assert _build_fts_match("vault sync scheduler") == '"vault" "sync" "scheduler"'


def test_special_chars_are_treated_as_literals():
    # Punctuation is dropped by the \w+ tokenizer; tokens stay quoted.
    assert _build_fts_match("RAG: pipeline?") == '"RAG" "pipeline"'


def test_all_stopwords_falls_back_not_empty():
    out = _build_fts_match("what is the")
    assert out  # must not collapse to an empty MATCH
    assert out == '"what" "is" "the"'


def test_empty_query_returns_raw():
    assert _build_fts_match("") == ""
