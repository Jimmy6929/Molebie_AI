"""Tests for user-defined brains (v2): CRUD, folder membership, scoped counts.

DB-backed via a fresh temp DB per test (mirrors test_rag_vector_threshold's
isolated_data_dir). No embeddings needed — brains/folders derive from
documents.relative_path.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync

USER_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + DB per test (mirrors test_rag_vault/threshold)."""
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


async def _seed(db, filename: str, relative_path: str, status: str = "completed"):
    await db.insert_document(
        user_id=USER_ID, filename=filename, storage_path=f"local://{filename}",
        file_type="text/markdown", file_size=10, doc_status=status,
        relative_path=relative_path,
    )


async def test_list_folders_derivation(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()
    await _seed(db, "a.md", "Areas/a.md")
    await _seed(db, "b.md", "Books/b.md")
    await _seed(db, "c.md", "Books/c.md")
    await _seed(db, "root.md", "root.md")           # no folder -> Inbox
    await _seed(db, "wip.md", "Areas/wip.md", status="processing")  # excluded
    folders = {f["folder"]: f["doc_count"] for f in await db.list_folders(USER_ID)}
    assert folders == {"Books": 2, "Areas": 1, "Inbox": 1}


async def test_brain_crud_and_folder_membership(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()
    await _seed(db, "a.md", "Areas/a.md")
    await _seed(db, "b.md", "Books/b.md")

    brain = await db.insert_brain(USER_ID, "Personal")
    bid = brain["id"]
    assert brain["name"] == "Personal"

    # empty brain -> [] (scope to nothing); unknown brain -> None (treat as All)
    assert await db.get_brain_folders(bid, USER_ID) == []
    assert await db.get_brain_folders("no-such-id", USER_ID) is None

    await db.add_brain_folder(bid, USER_ID, "Areas")
    await db.add_brain_folder(bid, USER_ID, "Areas")  # idempotent
    assert await db.get_brain_folders(bid, USER_ID) == ["Areas"]

    [info] = await db.list_brains(USER_ID)
    assert info["folders"] == ["Areas"]
    assert info["doc_count"] == 1            # Areas/a.md
    assert info["missing_folders"] == []

    # a referenced folder with no docs surfaces as missing (re-point case)
    await db.add_brain_folder(bid, USER_ID, "Ghost")
    [info] = await db.list_brains(USER_ID)
    assert "Ghost" in info["missing_folders"]

    await db.remove_brain_folder(bid, USER_ID, "Ghost")
    assert await db.get_brain_folders(bid, USER_ID) == ["Areas"]

    await db.update_brain(bid, USER_ID, "Me")
    assert (await db.get_brain(bid, USER_ID))["name"] == "Me"


async def test_duplicate_brain_name_raises(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()
    await db.insert_brain(USER_ID, "Books")
    with pytest.raises(sqlite3.IntegrityError):
        await db.insert_brain(USER_ID, "Books")


async def test_delete_brain_keeps_documents(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()
    await _seed(db, "a.md", "Areas/a.md")
    brain = await db.insert_brain(USER_ID, "X")
    await db.add_brain_folder(brain["id"], USER_ID, "Areas")

    await db.delete_brain(brain["id"], USER_ID)
    assert await db.get_brain(brain["id"], USER_ID) is None
    assert await db.list_brains(USER_ID) == []
    assert len(await db.list_documents(USER_ID)) == 1   # documents NOT deleted


async def test_many_to_many_folder_in_two_brains(isolated_data_dir):
    from app.services.database import get_database_service
    db = get_database_service()
    await _seed(db, "b.md", "Books/b.md")
    b1 = await db.insert_brain(USER_ID, "Reading")
    b2 = await db.insert_brain(USER_ID, "Motivation")
    await db.add_brain_folder(b1["id"], USER_ID, "Books")
    await db.add_brain_folder(b2["id"], USER_ID, "Books")
    assert await db.get_brain_folders(b1["id"], USER_ID) == ["Books"]
    assert await db.get_brain_folders(b2["id"], USER_ID) == ["Books"]
