"""
DB-backed tests for vector_search_chunks — the L2→cosine conversion, the
similarity threshold, and the completed-documents-only filter.

No embedding model loads: embeddings are hand-built 1024-dim unit vectors
(basis vectors and a 60° rotation), which make the expected cosine
similarities exact: identical → 1.0, orthogonal → 0.0, 60° → 0.5.
"""

from __future__ import annotations

import asyncio
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync

USER_ID = "00000000-0000-0000-0000-000000000001"
DIM = 1024


def _unit(angle_deg: float) -> list[float]:
    """Unit vector in the (e0, e1) plane at the given angle from e0."""
    v = [0.0] * DIM
    v[0] = math.cos(math.radians(angle_deg))
    v[1] = math.sin(math.radians(angle_deg))
    return v


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + DB per test (mirrors test_vault_sync.py)."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        monkeypatch.setenv("RAG_ENABLED", "true")

        get_settings.cache_clear()
        init_database_sync(td, embedding_dim=DIM, auth_mode="single")

        from app.services import database
        database._db_service = None  # type: ignore[attr-defined]
        database.get_database_service.cache_clear()

        yield td

        async def _close():
            from app.services.database import get_database_service
            await get_database_service().close()
        asyncio.run(_close())
        get_settings.cache_clear()


async def _seed_doc_with_chunk(
    filename: str,
    embedding: list[float],
    doc_status: str = "completed",
    metadata: dict | None = None,
):
    from app.services.database import get_database_service
    db = get_database_service()
    doc = await db.insert_document(
        user_id=USER_ID,
        filename=filename,
        storage_path=f"local://docs/{filename}",
        file_type="text/markdown",
        file_size=10,
        doc_status=doc_status,
    )
    await db.insert_chunks([
        {
            "document_id": doc["id"],
            "user_id": USER_ID,
            "content": f"content of {filename}",
            "chunk_index": 0,
            "metadata": metadata,
            "embedding": embedding,
        }
    ])
    return doc


@pytest.mark.asyncio
async def test_l2_to_cosine_conversion_is_exact(isolated_data_dir):
    """sqlite-vec returns L2 distance; the service converts via
    cos = 1 − d²/2 (valid for unit vectors). Verify against known angles."""
    from app.services.database import get_database_service

    await _seed_doc_with_chunk("same.md", _unit(0))      # cos 0°  = 1.0
    await _seed_doc_with_chunk("sixty.md", _unit(60))    # cos 60° = 0.5

    rows = await get_database_service().vector_search_chunks(
        USER_ID, _unit(0), threshold=0.0, limit=10
    )
    sims = {r["filename"]: r["similarity"] for r in rows}

    assert sims["same.md"] == pytest.approx(1.0, abs=1e-4)
    assert sims["sixty.md"] == pytest.approx(0.5, abs=1e-4)
    # Best match first (ORDER BY distance).
    assert [r["filename"] for r in rows] == ["same.md", "sixty.md"]


@pytest.mark.asyncio
async def test_threshold_drops_weak_matches(isolated_data_dir):
    from app.services.database import get_database_service

    await _seed_doc_with_chunk("strong.md", _unit(0))        # sim 1.0
    await _seed_doc_with_chunk("orthogonal.md", _unit(90))   # sim 0.0

    rows = await get_database_service().vector_search_chunks(
        USER_ID, _unit(0), threshold=0.3, limit=10
    )

    assert [r["filename"] for r in rows] == ["strong.md"]


@pytest.mark.asyncio
async def test_non_completed_documents_excluded(isolated_data_dir):
    from app.services.database import get_database_service

    await _seed_doc_with_chunk("done.md", _unit(0), doc_status="completed")
    await _seed_doc_with_chunk("inflight.md", _unit(0), doc_status="processing")

    rows = await get_database_service().vector_search_chunks(
        USER_ID, _unit(0), threshold=0.0, limit=10
    )

    assert [r["filename"] for r in rows] == ["done.md"]


@pytest.mark.asyncio
async def test_metadata_json_decoded_and_row_shape(isolated_data_dir):
    from app.services.database import get_database_service

    meta = {"heading": "Intro", "wikilinks": ["Other Note"], "tags": ["x"]}
    doc = await _seed_doc_with_chunk("note.md", _unit(0), metadata=meta)

    rows = await get_database_service().vector_search_chunks(
        USER_ID, _unit(0), threshold=0.0, limit=10
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["metadata"] == meta  # stored as JSON text, returned as dict
    assert row["document_id"] == doc["id"]
    assert row["chunk_index"] == 0
    assert set(row) >= {
        "chunk_id", "document_id", "filename", "content",
        "chunk_index", "metadata", "similarity",
    }


@pytest.mark.asyncio
async def test_user_isolation(isolated_data_dir):
    from app.services.database import get_database_service

    await _seed_doc_with_chunk("mine.md", _unit(0))
    rows = await get_database_service().vector_search_chunks(
        "00000000-0000-0000-0000-000000000002", _unit(0), threshold=0.0, limit=10
    )
    assert rows == []
