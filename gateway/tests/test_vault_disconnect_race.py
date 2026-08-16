"""
Regression tests for disconnecting a vault while its own sync job is running.

Disconnect deletes every document the vault owns. If the ingest worker keeps
running against that vault it writes `documents` rows tagged with a
`vault_source_id` that no longer resolves — orphans invisible to every vault
query, uncleanable by any later sync or disconnect, with their chunks left in
the RAG index forever. Observed in the wild as a stray
`IntegrityError: FOREIGN KEY constraint failed` on the file that was mid-embed
when the disconnect landed.

Embedding is mocked out throughout — what matters here is the ownership
handshake between the route and the worker, not the pipeline.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync

USER_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def isolated_data_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        monkeypatch.setenv("VAULT_ALLOWED_ROOTS", td)
        monkeypatch.setenv("RAG_ENABLED", "true")
        monkeypatch.setenv("VAULT_SYNC_ENABLED", "true")
        get_settings.cache_clear()
        init_database_sync(td, embedding_dim=1024, auth_mode="single")

        from app.services import database, storage
        database._db_service = None  # type: ignore[attr-defined]
        database.get_database_service.cache_clear()
        storage._storage_service = None  # type: ignore[attr-defined]

        yield td

        async def _close():
            from app.services.database import get_database_service
            await get_database_service().close()
        asyncio.run(_close())


async def _async_noop(*_args, **_kwargs) -> None:
    return None


def _documents(data_dir: str) -> list[dict]:
    db = sqlite3.connect(Path(data_dir) / "molebie.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM documents")]
    db.close()
    return rows


async def _staged_vault(data_dir: str, *, files: dict[str, str]) -> tuple[str, str]:
    """Connect a vault, sync it with the worker mocked out, and return
    (vault_id, job_id) with every file staged in 'uploaded'."""
    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    root = Path(data_dir) / "vault"
    root.mkdir(exist_ok=True)
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    db = get_database_service()
    vault = await db.insert_vault_source(
        user_id=USER_ID,
        label="Disconnected",
        root_path=str(root.resolve()),
        index_attachments=True,
    )
    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault["id"], USER_ID)
    assert report.job_id is not None
    return vault["id"], report.job_id


@pytest.mark.asyncio
async def test_worker_skips_file_whose_vault_was_disconnected(isolated_data_dir):
    """A queued file whose vault is gone must be skipped, not turned into an
    orphan document."""
    from app.services.database import get_database_service
    from app.services.ingest_worker import IngestWorker

    vault_id, job_id = await _staged_vault(isolated_data_dir, files={"a.md": "# A\nbody"})
    db = get_database_service()

    # The disconnect lands while the job is still queued.
    await db.delete_vault_source(vault_id, USER_ID)

    file_row = await db.next_pending_ingest_file(job_id)
    assert file_row is not None

    worker = IngestWorker()
    with patch("app.services.ingest_worker.get_document_processor") as proc:
        await worker._process_one_file(job_id, file_row)
        proc.return_value.process.assert_not_called()

    assert _documents(isolated_data_dir) == [], "no document may be written for a dead vault"

    refreshed = await db.get_ingest_file(file_row["id"])
    assert refreshed["status"] == "skipped"
    assert refreshed["error_message"] == "vault_disconnected"

    job = await db.get_ingest_job(job_id)
    assert job["skipped_files"] == 1
    assert job["failed_files"] == 0


@pytest.mark.asyncio
async def test_worker_skips_when_vault_disappears_mid_embed(isolated_data_dir):
    """The window that actually bit: the vault survives the pre-flight check and
    is disconnected during the (tens of seconds of) embedding."""
    from app.services.database import get_database_service
    from app.services.ingest_worker import IngestWorker

    vault_id, job_id = await _staged_vault(isolated_data_dir, files={"a.md": "# A\nbody"})
    db = get_database_service()

    file_row = await db.next_pending_ingest_file(job_id)
    assert file_row is not None

    async def _disconnect_then_chunk(*_args, **_kwargs):
        # Stands in for the embedder: the user disconnects the vault while the
        # file is being processed, deleting the document row just inserted.
        await db.delete_vault_source(vault_id, USER_ID)
        return [("chunk text", [0.1] * 1024, {"chunk_index": 0})]

    worker = IngestWorker()
    with patch("app.services.ingest_worker.get_document_processor") as proc:
        proc.return_value.process_async = _disconnect_then_chunk
        await worker._process_one_file(job_id, file_row)

    assert _documents(isolated_data_dir) == [], "no orphan document may survive"

    refreshed = await db.get_ingest_file(file_row["id"])
    assert refreshed["status"] == "skipped"
    assert refreshed["error_message"] == "vault_disconnected"


@pytest.mark.asyncio
async def test_failed_file_leaves_no_partial_document(isolated_data_dir):
    """Any failure mid-file must roll back the document row created for it —
    otherwise a permanently 'processing' document is left behind."""
    from app.services.database import get_database_service
    from app.services.ingest_worker import IngestWorker

    _vault_id, job_id = await _staged_vault(isolated_data_dir, files={"a.md": "# A\nbody"})
    db = get_database_service()

    file_row = await db.next_pending_ingest_file(job_id)
    assert file_row is not None

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("embedder exploded")

    worker = IngestWorker()
    with patch("app.services.ingest_worker.get_document_processor") as proc:
        proc.return_value.process_async = _boom
        await worker._process_one_file(job_id, file_row)

    assert _documents(isolated_data_dir) == [], "failed file must not leave a document"

    refreshed = await db.get_ingest_file(file_row["id"])
    assert refreshed["status"] == "failed"
    assert "embedder exploded" in refreshed["error_message"]


@pytest.mark.asyncio
async def test_disconnect_cancels_the_vaults_in_flight_job(isolated_data_dir):
    """Disconnecting must stop the job that is feeding the vault, so the worker
    doesn't keep processing files into a vault that no longer exists."""
    from app.middleware.auth import JWTPayload
    from app.routes.vault import disconnect_vault
    from app.services.database import get_database_service

    vault_id, job_id = await _staged_vault(
        isolated_data_dir, files={"a.md": "# A", "b.md": "# B"}
    )
    db = get_database_service()
    await db.set_ingest_job_running(job_id)

    await disconnect_vault(vault_id, JWTPayload(sub=USER_ID))

    job = await db.get_ingest_job(job_id)
    assert job["status"] == "cancelled"
    assert await db.get_active_ingest_job(USER_ID) is None


@pytest.mark.asyncio
async def test_disconnect_leaves_another_vaults_job_alone(isolated_data_dir):
    """The cancel is scoped to the vault being disconnected — an unrelated
    folder upload or other vault's sync must keep running."""
    from app.middleware.auth import JWTPayload
    from app.routes.vault import disconnect_vault
    from app.services.database import get_database_service

    other_id, job_id = await _staged_vault(isolated_data_dir, files={"a.md": "# A"})
    db = get_database_service()
    await db.set_ingest_job_running(job_id)

    idle = await db.insert_vault_source(
        user_id=USER_ID,
        label="Idle",
        root_path=str((Path(isolated_data_dir) / "idle").resolve()),
        index_attachments=True,
    )

    await disconnect_vault(idle["id"], JWTPayload(sub=USER_ID))

    job = await db.get_ingest_job(job_id)
    assert job["status"] == "running", "unrelated job must survive"
    assert await db.get_vault_source(other_id, USER_ID) is not None
