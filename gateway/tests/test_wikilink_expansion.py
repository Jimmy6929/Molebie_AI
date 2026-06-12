"""
Tests for one-hop wikilink expansion (dark-launched behind
rag_wikilink_expansion_enabled).

Three layers, none of which load models:
- find_document_by_title against real SQLite (hand-inserted documents),
- _expand_with_wikilinks against a stub db (caps, dedupe, tagging, error
  isolation),
- format_context budget non-displacement (linked chunks can only use
  leftover budget — pure, mirrors test_format_context_grouping.py).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync
from app.services.markdown_meta import extract_md_metadata
from app.services.rag import RAGService, _expand_with_wikilinks

USER_ID = "00000000-0000-0000-0000-000000000001"


# ───────── Resolution: find_document_by_title (real SQLite) ─────────


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + DB per test (mirrors test_vault_sync.py)."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        monkeypatch.setenv("RAG_ENABLED", "true")

        get_settings.cache_clear()
        init_database_sync(td, embedding_dim=1024, auth_mode="single")

        from app.services import database
        database._db_service = None  # type: ignore[attr-defined]
        database.get_database_service.cache_clear()

        yield td

        async def _close():
            from app.services.database import get_database_service
            await get_database_service().close()
        asyncio.run(_close())
        get_settings.cache_clear()


async def _insert_doc(
    filename: str,
    relative_path: str | None = None,
    doc_status: str = "completed",
    vault_id: str | None = None,
    user_id: str = USER_ID,
) -> dict:
    from app.services.database import get_database_service
    return await get_database_service().insert_document(
        user_id=user_id,
        filename=filename,
        storage_path=f"local://docs/{filename}",
        file_type="text/markdown",
        file_size=10,
        doc_status=doc_status,
        relative_path=relative_path or filename,
        vault_source_id=vault_id,
    )


async def _insert_vault(label: str) -> str:
    from app.services.database import get_database_service
    row = await get_database_service().insert_vault_source(
        user_id=USER_ID, label=label, root_path=f"/fake/{label}"
    )
    return row["id"]


@pytest.mark.asyncio
async def test_resolve_exact_case_insensitive_and_markdown_ext(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()

    note = await _insert_doc("My Note.md")
    longform = await _insert_doc("Longform.markdown")

    assert (await db.find_document_by_title(USER_ID, "My Note"))["id"] == note["id"]
    assert (await db.find_document_by_title(USER_ID, "my note"))["id"] == note["id"]
    assert (await db.find_document_by_title(USER_ID, "LONGFORM"))["id"] == longform["id"]
    assert await db.find_document_by_title(USER_ID, "No Such Note") is None
    assert await db.find_document_by_title(USER_ID, "  ") is None


@pytest.mark.asyncio
async def test_resolve_folder_qualified_prefers_exact_relative_path(
    isolated_data_dir,
):
    from app.services.database import get_database_service
    db = get_database_service()

    await _insert_doc("Note.md", relative_path="other/Note.md")
    in_proj = await _insert_doc("Note.md", relative_path="proj/Note.md")

    hit = await db.find_document_by_title(USER_ID, "proj/Note")
    assert hit["id"] == in_proj["id"]


@pytest.mark.asyncio
async def test_resolve_prefers_same_vault_then_created_at(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()

    vault_a = await _insert_vault("A")
    vault_b = await _insert_vault("B")
    doc_a = await _insert_doc("Shared.md", vault_id=vault_a)
    doc_b = await _insert_doc("Shared.md", relative_path="sub/Shared.md", vault_id=vault_b)

    assert (
        await db.find_document_by_title(USER_ID, "Shared", prefer_vault_id=vault_b)
    )["id"] == doc_b["id"]
    # No vault preference → deterministic created_at ASC winner.
    assert (await db.find_document_by_title(USER_ID, "Shared"))["id"] == doc_a["id"]


@pytest.mark.asyncio
async def test_resolve_excludes_non_completed_and_other_users(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()

    other_user = "00000000-0000-0000-0000-000000000002"
    conn = await db._get_conn()
    await conn.execute(
        "INSERT INTO users (id, email, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (other_user, "other@example.com", "2026-01-01", "2026-01-01"),
    )
    await conn.commit()

    await _insert_doc("Draft.md", doc_status="processing")
    await _insert_doc("Foreign.md", user_id=other_user)

    assert await db.find_document_by_title(USER_ID, "Draft") is None
    assert await db.find_document_by_title(USER_ID, "Foreign") is None


# ───────── Expansion: _expand_with_wikilinks (stub db) ─────────


def _settings(max_targets: int = 5, per_target: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        rag_wikilink_max_targets=max_targets,
        rag_wikilink_chunks_per_target=per_target,
    )


def _parent(
    doc: str,
    links: list[str],
    score: float = 0.5,
    vault: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": f"{doc}-0",
        "document_id": doc,
        "chunk_index": 0,
        "content": f"content {doc}",
        "rerank_score": score,
        "vault_source_id": vault,
        "metadata": {"wikilinks": links},
    }


class _StubDB:
    """Resolves '<Title>' → doc id 'doc:<title lowercased>' unless missing."""

    def __init__(self, missing: set[str] | None = None, raise_on: str | None = None):
        self.missing = missing or set()
        self.raise_on = raise_on
        self.resolve_calls: list[tuple[str, str | None]] = []
        self.range_calls: list[tuple[str, int, int]] = []

    async def find_document_by_title(self, user_id, title, prefer_vault_id=None):
        self.resolve_calls.append((title, prefer_vault_id))
        if title == self.raise_on:
            raise RuntimeError("boom")
        if title in self.missing:
            return None
        return {"id": f"doc:{title.lower()}", "filename": f"{title}.md"}

    async def get_chunks_in_range(self, user_id, doc_id, lo, hi):
        self.range_calls.append((doc_id, lo, hi))
        return [
            {
                "chunk_id": f"{doc_id}-{i}",
                "document_id": doc_id,
                "chunk_index": i,
                "content": f"linked {doc_id} {i}",
            }
            for i in range(lo, hi + 1)
        ]


@pytest.mark.asyncio
async def test_expand_tags_chunks_and_respects_per_target_cap():
    db = _StubDB()
    out = await _expand_with_wikilinks(
        [_parent("d1", ["Target"])], set(), db, USER_ID, _settings(per_target=2)
    )

    assert db.range_calls == [("doc:target", 0, 1)]  # leading chunks 0..k-1
    assert len(out) == 2
    assert all(c["is_parent"] is False for c in out)
    assert all(c["parent_score"] == 0.0 for c in out)
    assert all(c["wikilink_source"] == "Target" for c in out)


@pytest.mark.asyncio
async def test_expand_caps_targets_in_parent_score_order():
    parents = [
        _parent("weak", ["W1", "W2"], score=0.2),
        _parent("strong", ["S1", "S2"], score=0.9),
    ]
    db = _StubDB()
    out = await _expand_with_wikilinks(
        parents, set(), db, USER_ID, _settings(max_targets=3)
    )

    # Strongest parent's links win the cap; only 3 targets resolved.
    assert [c[0] for c in db.resolve_calls] == ["S1", "S2", "W1"]
    assert {c["wikilink_source"] for c in out} == {"S1", "S2", "W1"}


@pytest.mark.asyncio
async def test_expand_dedupes_targets_and_already_retrieved_docs():
    # Doc-level metadata: both parents carry the same link list.
    parents = [
        _parent("d1", ["Hub", "Already Here"], score=0.8),
        _parent("d2", ["hub"], score=0.6),  # same target, different case
    ]
    db = _StubDB()
    out = await _expand_with_wikilinks(
        parents, {"doc:already here"}, db, USER_ID, _settings()
    )

    assert [c[0] for c in db.resolve_calls] == ["Hub", "Already Here"]
    # 'Already Here' resolved to an existing doc → skipped; 'hub' deduped.
    assert {c["wikilink_source"] for c in out} == {"Hub"}


@pytest.mark.asyncio
async def test_expand_passes_majority_vault_preference():
    parents = [
        _parent("d1", ["T"], vault="v1"),
        _parent("d2", [], vault="v2"),
        _parent("d3", [], vault="v1"),
    ]
    db = _StubDB()
    await _expand_with_wikilinks(parents, set(), db, USER_ID, _settings())

    assert db.resolve_calls == [("T", "v1")]


@pytest.mark.asyncio
async def test_expand_error_isolated_and_missing_skipped():
    parents = [_parent("d1", ["Boom", "Ghost", "Fine"])]
    db = _StubDB(missing={"Ghost"}, raise_on="Boom")
    out = await _expand_with_wikilinks(parents, set(), db, USER_ID, _settings())

    assert {c["wikilink_source"] for c in out} == {"Fine"}


@pytest.mark.asyncio
async def test_expand_zero_caps_and_no_links_return_empty():
    db = _StubDB()
    assert await _expand_with_wikilinks(
        [_parent("d1", ["T"])], set(), db, USER_ID, _settings(max_targets=0)
    ) == []
    assert await _expand_with_wikilinks(
        [_parent("d1", [])], set(), db, USER_ID, _settings()
    ) == []
    assert db.resolve_calls == []


# ───────── Budget: linked chunks never displace parents ─────────


def _fmt_chunk(doc: str, content: str, *, parent: bool, score: float) -> dict:
    return {
        "document_id": doc,
        "chunk_index": 0,
        "content": content,
        "filename": f"{doc}.md",
        "metadata": {},
        "is_parent": parent,
        "parent_score": score,
        "rerank_score": score if parent else None,
    }


def _make_service(max_chars: int) -> RAGService:
    svc = RAGService.__new__(RAGService)
    svc.max_context_chars = max_chars
    return svc


def test_tight_budget_drops_linked_group_whole_keeps_parents():
    parents = [
        _fmt_chunk("primary1", "P" * 200, parent=True, score=0.9),
        _fmt_chunk("primary2", "Q" * 200, parent=True, score=0.8),
    ]
    linked = [_fmt_chunk("linked", "L" * 200, parent=False, score=0.0)]

    out = _make_service(max_chars=500).format_context(parents + linked)

    assert "P" * 200 in out and "Q" * 200 in out
    assert "L" not in out  # all-neighbor group dropped whole, no partial leak
    assert out.count("[S") == 2


def test_loose_budget_gives_linked_note_its_own_citation():
    parents = [_fmt_chunk("primary", "P" * 100, parent=True, score=0.9)]
    linked = [_fmt_chunk("linked", "L" * 100, parent=False, score=0.0)]

    out = _make_service(max_chars=12000).format_context(parents + linked)

    assert "P" * 100 in out and "L" * 100 in out
    assert out.count("[S") == 2  # linked note earns its own honest [S#]


# ───────── Regression: link target normalization upstream ─────────


def test_extract_md_metadata_strips_alias_and_heading_forms():
    meta = extract_md_metadata(
        "See [[My Note]], [[My Note#Section]], [[My Note|display]], "
        "and [[Other/Nested Note|x]]."
    )
    assert meta["wikilinks"] == ["My Note", "Other/Nested Note"]
