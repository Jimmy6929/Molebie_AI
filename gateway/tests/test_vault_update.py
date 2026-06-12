"""
Tests for vault editing (PATCH /documents/vault/{vault_id}) — the recovery
path for a moved/renamed vault folder — plus the computed VaultInfo.status.

Route handlers are plain async functions, so they're called directly with a
JWTPayload (no TestClient plumbing — matches the suite's convention of
asserting on HTTPException.status_code).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.middleware.auth import JWTPayload
from app.schema import init_database_sync

USER_ID = "00000000-0000-0000-0000-000000000001"
OTHER_USER = "00000000-0000-0000-0000-000000000002"
USER = JWTPayload(sub=USER_ID)


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + DB per test (mirrors test_vault_sync.py)."""
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
        get_settings.cache_clear()


@pytest.fixture
def vault_dir(isolated_data_dir):
    vault = Path(isolated_data_dir) / "vault_a"
    vault.mkdir()
    return vault


def _write(vault: Path, rel: str, body: str) -> Path:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


async def _connect(vault_dir: Path, label: str = "Vault A") -> dict:
    from app.services.database import get_database_service
    return await get_database_service().insert_vault_source(
        user_id=USER_ID,
        label=label,
        root_path=str(vault_dir.resolve()),
        index_attachments=True,
    )


async def _seed_document(vault_id: str, rel: str, body: str) -> dict:
    """Insert a completed document row as if a previous sync ingested it."""
    from app.services.database import get_database_service
    return await get_database_service().insert_document(
        user_id=USER_ID,
        filename=rel.rsplit("/", 1)[-1],
        storage_path=f"local://docs/{rel}",
        file_type="text/markdown",
        file_size=len(body),
        doc_status="completed",
        relative_path=rel,
        file_hash=hashlib.sha256(body.encode()).hexdigest(),
        vault_source_id=vault_id,
    )


# ───────── Headline: re-point reconciliation ─────────


@pytest.mark.asyncio
async def test_repoint_then_sync_reconciles_without_reingest(
    isolated_data_dir, vault_dir
):
    """Move the folder, re-point the vault, sync: identical files classify
    UNCHANGED and keep their document rows — zero re-embedding."""
    from app.services.database import get_database_service
    from app.services.vault_sync import sync_vault

    body = "# Alpha\nsame content"
    _write(vault_dir, "alpha.md", body)
    vault = await _connect(vault_dir)
    doc = await _seed_document(vault["id"], "alpha.md", body)

    moved = Path(isolated_data_dir) / "vault_b"
    os.rename(vault_dir, moved)

    db = get_database_service()
    row = await db.update_vault_source(
        vault["id"], USER_ID, root_path=str(moved.resolve())
    )
    assert row is not None and row["root_path"] == str(moved.resolve())

    with patch("app.services.vault_sync.get_ingest_worker"):
        report = await sync_vault(vault["id"], USER_ID)

    assert report.unchanged == 1, f"expected pure reconcile, got {report}"
    assert report.new == report.changed == report.deleted == 0
    surviving = await db.get_document(doc["id"], USER_ID)
    assert surviving is not None, "document must survive the re-point"


@pytest.mark.asyncio
async def test_repoint_then_sync_detects_real_edit(isolated_data_dir, vault_dir):
    """After the move, only a genuinely edited file re-ingests."""
    from app.services.vault_sync import sync_vault

    _write(vault_dir, "alpha.md", "# Alpha v1")
    _write(vault_dir, "beta.md", "# Beta v1")
    vault = await _connect(vault_dir)
    await _seed_document(vault["id"], "alpha.md", "# Alpha v1")
    await _seed_document(vault["id"], "beta.md", "# Beta v1")

    moved = Path(isolated_data_dir) / "vault_b"
    os.rename(vault_dir, moved)
    _write(moved, "beta.md", "# Beta v2 — edited after the move")

    from app.services.database import get_database_service
    await get_database_service().update_vault_source(
        vault["id"], USER_ID, root_path=str(moved.resolve())
    )

    with patch("app.services.vault_sync.get_ingest_worker") as worker_factory:
        worker_factory.return_value.ensure_worker_started = AsyncMock()
        report = await sync_vault(vault["id"], USER_ID)

    assert report.unchanged == 1
    assert report.changed == 1
    assert report.new == report.deleted == 0


# ───────── PATCH endpoint validation ─────────


@pytest.mark.asyncio
async def test_patch_validation_matrix(isolated_data_dir, vault_dir):
    from app.routes.vault import UpdateVaultRequest, update_vault

    vault = await _connect(vault_dir)

    # Nonexistent target path → 400.
    with pytest.raises(HTTPException) as exc:
        await update_vault(
            vault["id"],
            UpdateVaultRequest(root_path=str(Path(isolated_data_dir) / "nope")),
            USER,
        )
    assert exc.value.status_code == 400

    # Path outside VAULT_ALLOWED_ROOTS → 400.
    with pytest.raises(HTTPException) as exc:
        await update_vault(vault["id"], UpdateVaultRequest(root_path="/etc"), USER)
    assert exc.value.status_code == 400

    # Unknown vault id → 404.
    with pytest.raises(HTTPException) as exc:
        await update_vault("not-a-vault", UpdateVaultRequest(label="X"), USER)
    assert exc.value.status_code == 404

    # Another user's vault → 404 (ownership isolation).
    with pytest.raises(HTTPException) as exc:
        await update_vault(
            vault["id"], UpdateVaultRequest(label="X"), JWTPayload(sub=OTHER_USER)
        )
    assert exc.value.status_code == 404

    # Neither field → 400.
    with pytest.raises(HTTPException) as exc:
        await update_vault(vault["id"], UpdateVaultRequest(), USER)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_patch_duplicate_root_conflicts(isolated_data_dir, vault_dir):
    from app.routes.vault import UpdateVaultRequest, update_vault

    other_dir = Path(isolated_data_dir) / "vault_other"
    other_dir.mkdir()
    await _connect(vault_dir, label="First")
    second = await _connect(other_dir, label="Second")

    with pytest.raises(HTTPException) as exc:
        await update_vault(
            second["id"],
            UpdateVaultRequest(root_path=str(vault_dir.resolve())),
            USER,
        )
    assert exc.value.status_code == 409
    assert "First" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_patch_label_only_works_while_root_missing(
    isolated_data_dir, vault_dir
):
    """Renaming must not validate the (possibly gone) current root — that is
    exactly what makes a broken vault still manageable."""
    from app.routes.vault import UpdateVaultRequest, update_vault

    vault = await _connect(vault_dir, label="Old Name")
    shutil.rmtree(vault_dir)

    info = await update_vault(
        vault["id"], UpdateVaultRequest(label="New Name"), USER
    )
    assert info.label == "New Name"
    assert info.status == "path_missing"


@pytest.mark.asyncio
async def test_patch_repoint_returns_ok_status_and_doc_count(
    isolated_data_dir, vault_dir
):
    from app.routes.vault import UpdateVaultRequest, update_vault

    body = "# Alpha"
    _write(vault_dir, "alpha.md", body)
    vault = await _connect(vault_dir)
    await _seed_document(vault["id"], "alpha.md", body)

    moved = Path(isolated_data_dir) / "vault_b"
    os.rename(vault_dir, moved)

    info = await update_vault(
        vault["id"],
        UpdateVaultRequest(root_path=str(moved.resolve())),
        USER,
    )
    assert info.status == "ok"
    assert info.root_path == str(moved.resolve())
    assert info.doc_count == 1


# ───────── Status computation ─────────


@pytest.mark.asyncio
async def test_list_vaults_reports_path_missing(isolated_data_dir, vault_dir):
    from app.routes.vault import list_vaults

    await _connect(vault_dir)
    res = await list_vaults(USER)
    assert res.vaults[0].status == "ok"

    shutil.rmtree(vault_dir)
    res = await list_vaults(USER)
    assert res.vaults[0].status == "path_missing"


# ───────── DB method semantics ─────────


@pytest.mark.asyncio
async def test_update_vault_source_ownership_and_partial_update(
    isolated_data_dir, vault_dir
):
    from app.services.database import get_database_service

    db = get_database_service()
    vault = await _connect(vault_dir, label="Original")

    # Wrong user → None, row untouched.
    assert await db.update_vault_source(vault["id"], OTHER_USER, label="Hax") is None
    fresh = await db.get_vault_source(vault["id"], USER_ID)
    assert fresh is not None and fresh["label"] == "Original"

    # Label-only update leaves root_path untouched.
    row = await db.update_vault_source(vault["id"], USER_ID, label="Renamed")
    assert row is not None
    assert row["label"] == "Renamed"
    assert row["root_path"] == vault["root_path"]
