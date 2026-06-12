"""
Tests for the vault auto-sync scheduler — tick gating, signature pre-scan,
and the busy/missing-root skip paths.

``sync_vault`` itself is covered by test_vault_sync.py; here it is mocked so
these tests assert pure scheduling behavior: WHEN the engine is invoked, not
what it does. The signature scan runs for real against a tempdir vault.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync

USER_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + DB per test (mirrors test_vault_sync.py)."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        monkeypatch.setenv("VAULT_ALLOWED_ROOTS", td)
        monkeypatch.setenv("RAG_ENABLED", "true")
        monkeypatch.setenv("VAULT_SYNC_ENABLED", "true")
        monkeypatch.setenv("VAULT_AUTO_SYNC_ENABLED", "true")

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
    vault = Path(isolated_data_dir) / "test_vault"
    vault.mkdir()
    return vault


def _write(vault: Path, rel: str, body: str) -> Path:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


async def _connect_vault(vault_dir: Path, label: str = "TestVault") -> dict:
    from app.services.database import get_database_service
    db = get_database_service()
    return await db.insert_vault_source(
        user_id=USER_ID,
        label=label,
        root_path=str(vault_dir.resolve()),
        index_attachments=True,
    )


def _scheduler():
    from app.services.vault_auto_sync import VaultAutoSyncScheduler
    return VaultAutoSyncScheduler()


# ───────── Tick gating ─────────


@pytest.mark.asyncio
async def test_tick_disabled_flag(isolated_data_dir, monkeypatch):
    monkeypatch.setenv("VAULT_AUTO_SYNC_ENABLED", "false")
    get_settings.cache_clear()

    with patch("app.services.vault_auto_sync.sync_vault") as mock_sync:
        result = await _scheduler()._tick()

    assert result.ran is False
    assert result.reason == "disabled"
    mock_sync.assert_not_called()


@pytest.mark.asyncio
async def test_tick_no_vaults(isolated_data_dir):
    with patch("app.services.vault_auto_sync.sync_vault") as mock_sync:
        result = await _scheduler()._tick()

    assert result.ran is False
    assert result.reason == "no_vaults"
    mock_sync.assert_not_called()


# ───────── Signature-driven sync decisions ─────────


@pytest.mark.asyncio
async def test_tick_syncs_once_then_skips_unchanged(isolated_data_dir, vault_dir):
    _write(vault_dir, "alpha.md", "# Alpha")
    await _connect_vault(vault_dir)
    scheduler = _scheduler()

    with patch(
        "app.services.vault_auto_sync.sync_vault", new=AsyncMock()
    ) as mock_sync:
        first = await scheduler._tick()
        second = await scheduler._tick()

    assert first.ran and first.synced == 1
    assert second.ran and second.synced == 0 and second.skipped_unchanged == 1
    assert mock_sync.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: _write(v, "alpha.md", "# Alpha — edited, new size"),
        lambda v: _write(v, "fresh.md", "# Fresh"),
        lambda v: (v / "alpha.md").unlink(),
        lambda v: os.utime(v / "alpha.md", (1000000000, 1000000000)),
    ],
    ids=["edited", "added", "deleted", "touched"],
)
async def test_tick_resyncs_on_change(isolated_data_dir, vault_dir, mutate):
    _write(vault_dir, "alpha.md", "# Alpha")
    await _connect_vault(vault_dir)
    scheduler = _scheduler()

    with patch(
        "app.services.vault_auto_sync.sync_vault", new=AsyncMock()
    ) as mock_sync:
        await scheduler._tick()          # baseline: caches signature
        mutate(vault_dir)
        result = await scheduler._tick()

    assert result.synced == 1, f"expected re-sync after mutation, got {result}"
    assert mock_sync.await_count == 2


@pytest.mark.asyncio
async def test_tick_ignores_non_indexable_changes(isolated_data_dir, vault_dir):
    """Changes inside .obsidian/ or to unsupported extensions never trigger."""
    _write(vault_dir, "alpha.md", "# Alpha")
    await _connect_vault(vault_dir)
    scheduler = _scheduler()

    with patch(
        "app.services.vault_auto_sync.sync_vault", new=AsyncMock()
    ) as mock_sync:
        await scheduler._tick()
        _write(vault_dir, ".obsidian/workspace.json", "{}")
        _write(vault_dir, "image.png", "binary")
        result = await scheduler._tick()

    assert result.skipped_unchanged == 1
    assert mock_sync.await_count == 1


# ───────── Skip paths ─────────


@pytest.mark.asyncio
async def test_tick_missing_root_skips_quietly(isolated_data_dir, vault_dir):
    _write(vault_dir, "alpha.md", "# Alpha")
    await _connect_vault(vault_dir)
    shutil.rmtree(vault_dir)

    with patch("app.services.vault_auto_sync.sync_vault") as mock_sync:
        result = await _scheduler()._tick()

    assert result.ran is True
    assert result.skipped_missing == 1
    assert result.errors == []
    mock_sync.assert_not_called()


@pytest.mark.asyncio
async def test_tick_busy_user_skips_and_retries(isolated_data_dir, vault_dir):
    """An active ingest job defers auto-sync, and the signature is NOT cached
    so the next tick (job finished) picks the change up."""
    from app.services.database import get_database_service
    db = get_database_service()

    _write(vault_dir, "alpha.md", "# Alpha")
    vault = await _connect_vault(vault_dir)
    job = await db.create_ingest_job(
        USER_ID,
        "manual-upload",
        [{"relative_path": "x.md", "file_size": 1, "content_type": "text/markdown"}],
    )
    scheduler = _scheduler()

    with patch(
        "app.services.vault_auto_sync.sync_vault", new=AsyncMock()
    ) as mock_sync:
        busy = await scheduler._tick()
        assert busy.skipped_busy == 1
        assert vault["id"] not in scheduler._sig_cache
        mock_sync.assert_not_awaited()

        # Finish the job → next tick must sync.
        conn = await db._get_conn()
        await conn.execute(
            "UPDATE ingest_jobs SET status='completed' WHERE id = ?", (job["id"],)
        )
        await conn.commit()

        after = await scheduler._tick()

    assert after.synced == 1
    assert mock_sync.await_count == 1


@pytest.mark.asyncio
async def test_tick_one_vault_error_does_not_block_others(
    isolated_data_dir, vault_dir
):
    other_dir = Path(isolated_data_dir) / "other_vault"
    other_dir.mkdir()
    _write(vault_dir, "alpha.md", "# Alpha")
    _write(other_dir, "beta.md", "# Beta")
    bad = await _connect_vault(vault_dir, label="BadVault")

    async def _sync_side_effect(vault_id: str, user_id: str):
        if vault_id == bad["id"]:
            raise RuntimeError("boom")

    await _connect_vault(other_dir, label="GoodVault")

    with patch(
        "app.services.vault_auto_sync.sync_vault",
        new=AsyncMock(side_effect=_sync_side_effect),
    ):
        result = await _scheduler()._tick()

    assert result.checked == 2
    assert result.synced == 1
    assert len(result.errors) == 1 and "BadVault" in result.errors[0]


# ───────── Signature scan (pure) ─────────


def test_scan_signature_stable_and_filtered(tmp_path):
    from app.services.vault_sync import _scan_vault_signature

    _write(tmp_path, "a.md", "# A")
    _write(tmp_path, "nested/b.md", "# B")
    _write(tmp_path, ".obsidian/workspace.json", "{}")
    _write(tmp_path, "image.png", "binary")

    sig1 = _scan_vault_signature(tmp_path, [".obsidian"], ["*.png"], True)
    sig2 = _scan_vault_signature(tmp_path, [".obsidian"], ["*.png"], True)
    assert sig1 == sig2

    # Ignored / non-indexable churn does not move the signature.
    _write(tmp_path, ".obsidian/cache.json", "xyz")
    _write(tmp_path, "photo.png", "more binary")
    assert _scan_vault_signature(tmp_path, [".obsidian"], ["*.png"], True) == sig1

    # Indexable churn does.
    _write(tmp_path, "c.md", "# C")
    assert _scan_vault_signature(tmp_path, [".obsidian"], ["*.png"], True) != sig1


def test_scan_signature_detects_delete_and_touch(tmp_path):
    from app.services.vault_sync import _scan_vault_signature

    _write(tmp_path, "a.md", "# A")
    _write(tmp_path, "b.md", "# B")
    base = _scan_vault_signature(tmp_path, [], [], True)

    os.utime(tmp_path / "a.md", (1000000000, 1000000000))
    touched = _scan_vault_signature(tmp_path, [], [], True)
    assert touched != base

    (tmp_path / "b.md").unlink()
    assert _scan_vault_signature(tmp_path, [], [], True) != touched


# ───────── Lifecycle ─────────


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_completes(isolated_data_dir):
    scheduler = _scheduler()
    await scheduler.start(interval=3600.0)
    task = scheduler._task
    await scheduler.start(interval=3600.0)
    assert scheduler._task is task, "second start must not spawn a new task"
    await scheduler.stop()
    assert scheduler._task is None
