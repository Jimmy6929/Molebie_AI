"""
End-to-end test for vault sync's file walking + classification + dispatch
staging — without invoking the embedder.

The expensive bits (extract → chunk → embed) live in the IngestWorker which we
mock out. What we exercise here is the part that matters for correctness:

* SHA256 hash diff against existing `documents.file_hash` rows.
* NEW / CHANGED / UNCHANGED / DELETED classification.
* Idempotency: a second sync with no changes is a no-op.
* Path-traversal / extension / iCloud-placeholder filters.
* `vault_source_id` propagation from job → ingest_job_files → documents.

Runs against a real SQLite DB initialised in a tempdir so the schema
migration code path is exercised too.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# The gateway tests/conftest.py expects to be run from gateway/, but importing
# `app.*` requires gateway/ on sys.path. pytest from gateway/ already does
# this via its rootdir. We just import the modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Spin up a fresh data_dir + DB per test. Patch settings.data_dir so the
    DatabaseService and StorageService both use it. Also reset the cached
    settings + db + storage singletons so the patched values take effect."""
    with tempfile.TemporaryDirectory() as td:
        # Point settings at the tempdir
        monkeypatch.setenv("DATA_DIR", td)
        monkeypatch.setenv("VAULT_ALLOWED_ROOTS", td)  # whitelist tempdir
        monkeypatch.setenv("RAG_ENABLED", "true")
        monkeypatch.setenv("VAULT_SYNC_ENABLED", "true")

        # Reset cached settings.
        get_settings.cache_clear()

        # Init the DB.
        init_database_sync(td, embedding_dim=1024, auth_mode="single")

        # Reset cached service singletons so they pick up the new dir.
        from app.services import database, storage
        database._db_service = None  # type: ignore[attr-defined]
        database.get_database_service.cache_clear()
        storage._storage_service = None  # type: ignore[attr-defined]

        yield td

        # Clean up DB connections so the tempdir can be removed.
        async def _close():
            from app.services.database import get_database_service
            await get_database_service().close()
        asyncio.run(_close())


@pytest.fixture
def vault_dir(isolated_data_dir):
    """Build a fresh test vault under data_dir/test_vault/."""
    vault = Path(isolated_data_dir) / "test_vault"
    vault.mkdir()
    return vault


def _write(vault: Path, rel: str, body: str) -> Path:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _all_documents(data_dir: str) -> list[dict]:
    """Read documents directly via sqlite3 to avoid async fixtures."""
    db = sqlite3.connect(Path(data_dir) / "molebie.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM documents")]
    db.close()
    return rows


def _ingest_files(data_dir: str) -> list[dict]:
    db = sqlite3.connect(Path(data_dir) / "molebie.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM ingest_job_files")]
    db.close()
    return rows


# ───────── Tests ─────────


@pytest.mark.asyncio
async def test_sync_classifies_new_changed_unchanged_deleted(
    isolated_data_dir, vault_dir
):
    """Full happy-path: walk → diff → dispatch.

    The embedder is bypassed by patching IngestWorker.ensure_worker_started so
    the worker never actually runs — we only verify that the correct
    `ingest_job_files` rows are queued in the right state.
    """
    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    # 1. Populate vault with three markdown notes.
    _write(vault_dir, "alpha.md", "# Alpha\nfirst note")
    _write(vault_dir, "beta.md", "# Beta\nsecond note")
    _write(vault_dir, "nested/gamma.md", "# Gamma\nthird note")

    # ── Decoy files that must be ignored / skipped ──
    _write(vault_dir, ".obsidian/workspace.json", "{}")  # default ignore
    _write(vault_dir, "image.png", "binary")             # unsupported ext
    placeholder = vault_dir / ".alpha.md.icloud"          # iCloud shadow
    placeholder.touch()

    # 2. Connect a vault source.
    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"
    # Ensure user exists (single-user mode default user)
    rows = await (await db._get_conn()).execute_fetchall(
        "SELECT id FROM users WHERE id = ?", (user_id,)
    )
    assert rows, "default user must be present after init_database_sync"

    vault = await db.insert_vault_source(
        user_id=user_id,
        label="TestVault",
        root_path=str(vault_dir.resolve()),
        index_attachments=True,
    )
    vault_id = vault["id"]

    # 3. First sync — everything new. Mock the worker to avoid embedding.
    with patch(
        "app.services.vault_sync.get_ingest_worker"
    ) as mock_worker_factory:
        mock_worker = mock_worker_factory.return_value
        mock_worker.ensure_worker_started = _async_noop

        report = await sync_vault(vault_id, user_id)

    assert report.new == 3, f"expected 3 new, got {report}"
    assert report.changed == 0
    assert report.unchanged == 0
    assert report.deleted == 0
    assert report.job_id is not None

    # The three accepted files should be queued in 'uploaded' state with
    # vault_source_id propagated through the JOIN.
    files = _ingest_files(isolated_data_dir)
    assert len(files) == 3
    paths = {f["relative_path"] for f in files}
    assert paths == {"alpha.md", "beta.md", "nested/gamma.md"}
    for f in files:
        assert f["status"] == "uploaded", f
        assert f["file_hash"] is not None, f
        assert f["storage_path"] is not None, f


@pytest.mark.asyncio
async def test_sync_idempotent_when_unchanged(isolated_data_dir, vault_dir):
    """A second sync with no changes records `unchanged == total` and
    creates ZERO new ingest_job_files rows."""
    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    _write(vault_dir, "stable.md", "# stable")

    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"
    vault = await db.insert_vault_source(
        user_id=user_id, label="V", root_path=str(vault_dir.resolve()),
    )

    # Pretend the first sync embedded the file by manually inserting a
    # documents row with a matching SHA256. This is faster than actually
    # running the embedder.
    import hashlib
    body = (vault_dir / "stable.md").read_bytes()
    sha = hashlib.sha256(body).hexdigest()
    await db.insert_document(
        user_id=user_id,
        filename="stable.md",
        storage_path="fake/stable.md",
        file_type="text/markdown",
        file_size=len(body),
        doc_status="completed",
        relative_path="stable.md",
        file_hash=sha,
        vault_source_id=vault["id"],
    )

    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault["id"], user_id)

    assert report.new == 0
    assert report.changed == 0
    assert report.unchanged == 1
    assert report.deleted == 0
    assert report.job_id is None, "no job should be created when nothing changed"

    # No new ingest_job_files rows.
    assert _ingest_files(isolated_data_dir) == []


@pytest.mark.asyncio
async def test_sync_detects_changed_and_deleted(isolated_data_dir, vault_dir):
    """Edit one file, delete another, leave one alone — classify accordingly."""
    import hashlib

    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    _write(vault_dir, "a.md", "version-1")
    _write(vault_dir, "b.md", "stable")
    _write(vault_dir, "c.md", "to-be-deleted")

    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"
    vault = await db.insert_vault_source(
        user_id=user_id, label="V", root_path=str(vault_dir.resolve()),
    )
    vault_id = vault["id"]

    # Pre-seed three "already indexed" documents with hashes from the v1 contents.
    for rel in ("a.md", "b.md", "c.md"):
        body = (vault_dir / rel).read_bytes()
        await db.insert_document(
            user_id=user_id,
            filename=rel,
            storage_path=f"fake/{rel}",
            file_type="text/markdown",
            file_size=len(body),
            doc_status="completed",
            relative_path=rel,
            file_hash=hashlib.sha256(body).hexdigest(),
            vault_source_id=vault_id,
        )

    # Now mutate the vault.
    (vault_dir / "a.md").write_text("version-2", encoding="utf-8")
    (vault_dir / "c.md").unlink()

    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault_id, user_id)

    assert report.new == 0
    assert report.changed == 1, report
    assert report.unchanged == 1, report
    assert report.deleted == 1, report

    # The CHANGED document was hard-deleted (so re-ingest creates a fresh row);
    # only "b.md" remains pre-existing. After dispatch staging, no new
    # 'documents' row is created until the worker runs — so we should see only
    # `b.md` in the documents table at this point.
    surviving = [d for d in _all_documents(isolated_data_dir)
                 if d.get("vault_source_id") == vault_id]
    paths = {d["relative_path"] for d in surviving}
    assert paths == {"b.md"}, f"expected only b.md to survive, got {paths}"

    # Exactly one ingest_job_files row queued (for the CHANGED a.md).
    queued = _ingest_files(isolated_data_dir)
    assert len(queued) == 1, queued
    assert queued[0]["relative_path"] == "a.md"
    assert queued[0]["status"] == "uploaded"


@pytest.mark.asyncio
async def test_sync_rejects_root_outside_allowed(isolated_data_dir):
    """A vault root outside VAULT_ALLOWED_ROOTS must fail validation."""
    from app.services.vault_sync import _check_root_allowed

    # Default /tmp and the test isolated_data_dir is whitelisted; /etc is not.
    forbidden = Path("/etc")
    if forbidden.exists():
        reason = _check_root_allowed(forbidden)
        assert reason is not None and "must lie under" in reason


@pytest.mark.asyncio
async def test_sync_adopts_unattached_document_by_hash(
    isolated_data_dir, vault_dir
):
    """A previously folder-uploaded doc with matching SHA256 should be
    adopted into the vault — no re-embedding, just a pointer flip."""
    import hashlib

    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    # The vault contains a single note "shared.md".
    body = b"# shared content"
    _write(vault_dir, "shared.md", body.decode())
    sha = hashlib.sha256(body).hexdigest()

    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"

    # Pre-seed an UNATTACHED document with the same hash but a different
    # relative_path (mimics the case where the user previously imported a
    # parent folder, so the path was prefixed with the parent's name).
    orphan = await db.insert_document(
        user_id=user_id,
        filename="shared.md",
        storage_path="fake/old/shared.md",
        file_type="text/markdown",
        file_size=len(body),
        doc_status="completed",
        relative_path="OldParent/shared.md",
        file_hash=sha,
        vault_source_id=None,
    )

    vault = await db.insert_vault_source(
        user_id=user_id, label="V", root_path=str(vault_dir.resolve()),
    )
    vault_id = vault["id"]

    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault_id, user_id)

    assert report.new == 0, "must NOT re-embed when an orphan with matching hash exists"
    assert report.adopted == 1, report
    assert report.unchanged == 0
    assert report.job_id is None, "no job spawned when nothing needs embedding"

    # The orphan should now be tagged with vault_source_id and the vault's
    # relative_path.
    docs = _all_documents(isolated_data_dir)
    [adopted] = [d for d in docs if d["id"] == orphan["id"]]
    assert adopted["vault_source_id"] == vault_id
    assert adopted["relative_path"] == "shared.md"


@pytest.mark.asyncio
async def test_sync_does_not_adopt_doc_attached_to_other_vault(
    isolated_data_dir, vault_dir
):
    """Hash collision with a doc owned by a DIFFERENT vault must not steal
    that doc — adoption only operates on `vault_source_id IS NULL` rows."""
    import hashlib

    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    body = b"# shared"
    _write(vault_dir, "shared.md", body.decode())
    sha = hashlib.sha256(body).hexdigest()

    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"

    # Pre-seed a doc that ALREADY belongs to vault A.
    other_vault = await db.insert_vault_source(
        user_id=user_id, label="A", root_path=str(vault_dir.resolve()) + "_other",
    )
    await db.insert_document(
        user_id=user_id,
        filename="shared.md",
        storage_path="fake/a/shared.md",
        file_type="text/markdown",
        file_size=len(body),
        doc_status="completed",
        relative_path="shared.md",
        file_hash=sha,
        vault_source_id=other_vault["id"],
    )

    # Now connect vault B (rooted at the actual test dir) and sync.
    vault_b = await db.insert_vault_source(
        user_id=user_id, label="B", root_path=str(vault_dir.resolve()),
    )

    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault_b["id"], user_id)

    assert report.adopted == 0, "must not adopt a doc that belongs to another vault"
    assert report.new == 1, report

    # vault A's doc must still belong to vault A — adoption did not touch it.
    docs = _all_documents(isolated_data_dir)
    by_vault = {d["vault_source_id"]: d for d in docs if d["relative_path"] == "shared.md"}
    assert other_vault["id"] in by_vault


@pytest.mark.asyncio
async def test_sync_does_not_adopt_same_orphan_twice(
    isolated_data_dir, vault_dir
):
    """Two vault files with the same SHA256 must not both adopt one orphan —
    one wins, the other becomes NEW."""
    import hashlib

    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    body = b"# duplicate"
    sha = hashlib.sha256(body).hexdigest()
    _write(vault_dir, "first.md", body.decode())
    _write(vault_dir, "second.md", body.decode())  # identical contents

    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"
    await db.insert_document(
        user_id=user_id,
        filename="dup.md",
        storage_path="fake/dup.md",
        file_type="text/markdown",
        file_size=len(body),
        doc_status="completed",
        relative_path="legacy/dup.md",
        file_hash=sha,
        vault_source_id=None,
    )
    vault = await db.insert_vault_source(
        user_id=user_id, label="V", root_path=str(vault_dir.resolve()),
    )

    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault["id"], user_id)

    assert report.adopted == 1, "exactly one of the two duplicates wins adoption"
    assert report.new == 1, "the other becomes a NEW dispatch"


@pytest.mark.asyncio
async def test_sync_only_indexes_markdown_when_attachments_disabled(
    isolated_data_dir, vault_dir
):
    """index_attachments=False → PDFs and .txt are skipped."""
    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    _write(vault_dir, "note.md", "# md")
    _write(vault_dir, "side.pdf", "%PDF-fake")
    _write(vault_dir, "extra.txt", "plain text")

    db = get_database_service()
    user_id = "00000000-0000-0000-0000-000000000001"
    vault = await db.insert_vault_source(
        user_id=user_id, label="MD", root_path=str(vault_dir.resolve()),
        index_attachments=False,
    )

    with patch("app.services.vault_sync.get_ingest_worker") as mw:
        mw.return_value.ensure_worker_started = _async_noop
        report = await sync_vault(vault["id"], user_id)

    assert report.new == 1, "only the .md should be picked up"
    queued = _ingest_files(isolated_data_dir)
    paths = {f["relative_path"] for f in queued}
    assert paths == {"note.md"}


# ───────── Helpers ─────────


async def _async_noop(*_args, **_kwargs) -> None:
    return None
