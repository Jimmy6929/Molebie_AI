"""Tests for StorageMigrationScheduler — slice 9.4b.

We exercise ``_tick()`` directly rather than spinning a real loop, the
same way ``test_backend_probe.py`` calls ``_probe_one`` directly. Disk
fullness is controlled by monkeypatching ``shutil.disk_usage`` on the
scheduler module. The 9.4a engine is exercised through an injected
``mover_factory`` that wires in the existing ``_AlwaysOkClient`` fake.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync
from app.services import storage_scheduler as scheduler_mod
from app.services.storage import LocalStorageService
from app.services.storage_mover import StorageMover
from app.services.storage_scheduler import (
    MigrationTickResult,
    StorageMigrationScheduler,
)
from tests.test_storage_mover import (  # reuse 9.4a helpers (not fixtures)
    _AlwaysOkClient,
    _doc_storage_path,
    _register_satellite,
    _satellite_blobs,
    _seed_document,
)


@pytest.fixture
def data_dir():
    """Local tempdir + initialized SQLite — mirrors the 9.4a mover fixture,
    defined here so ``from ... import data_dir`` doesn't trip F811."""
    with tempfile.TemporaryDirectory() as td:
        init_database_sync(td, embedding_dim=1024, auth_mode="single")
        yield td

_Usage = namedtuple("_Usage", ["total", "used", "free"])
_CONTENT = b"scheduler integration content"


# ─────────────────────── helpers ───────────────────────


def _set_disk_used_fraction(monkeypatch, fraction: float) -> None:
    total = 1_000_000
    used = int(total * fraction)
    monkeypatch.setattr(
        scheduler_mod.shutil,
        "disk_usage",
        lambda *_a, **_k: _Usage(total, used, total - used),
    )


def _override_settings(monkeypatch, data_dir_path: str, **kwargs) -> None:
    """Force settings fields by patching the cached Settings instance.

    ``data_dir`` must be set so module-level helpers
    (``_has_registered_satellites``) hit the test tempdir's SQLite, not
    the repo default ``data/`` directory.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", data_dir_path)
    for k, v in kwargs.items():
        monkeypatch.setattr(settings, k, v)


def _ok_mover_factory(data_dir: str) -> Callable[[str], StorageMover]:
    """Factory returning a real StorageMover wired to _AlwaysOkClient."""
    shared = _AlwaysOkClient()

    def _build(identity: str) -> StorageMover:
        return StorageMover(
            LocalStorageService(data_dir), identity, data_dir,
            http_client_factory=lambda: shared,
        )
    return _build


# ─────────────────────── gate ordering ───────────────────────


class TestGates:
    @pytest.mark.asyncio
    async def test_disabled_short_circuits(self, data_dir, monkeypatch):
        _override_settings(monkeypatch, data_dir, storage_auto_migrate_enabled=False)
        sched = StorageMigrationScheduler(
            data_dir, mover_factory=lambda _i: pytest.fail("mover must not be built"),
        )
        result = await sched._tick()
        assert result == MigrationTickResult(ran=False, reason="disabled")

    @pytest.mark.asyncio
    async def test_no_satellite_skips_before_disk_syscall(self, data_dir, monkeypatch):
        _override_settings(monkeypatch, data_dir, storage_auto_migrate_enabled=True)
        # If disk_usage gets called, fail — the gate ordering must short-circuit first.
        monkeypatch.setattr(
            scheduler_mod.shutil, "disk_usage",
            lambda *_a, **_k: pytest.fail("disk_usage called before satellite gate"),
        )
        sched = StorageMigrationScheduler(data_dir)
        result = await sched._tick()
        assert result.ran is False
        assert result.reason == "no_satellite"

    @pytest.mark.asyncio
    async def test_no_identity_skips(self, data_dir, monkeypatch):
        _override_settings(monkeypatch, data_dir, storage_auto_migrate_enabled=True)
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        monkeypatch.setattr(scheduler_mod, "get_operator_identity", lambda: None)
        sched = StorageMigrationScheduler(data_dir)
        result = await sched._tick()
        assert result.ran is False
        assert result.reason == "no_identity"

    @pytest.mark.asyncio
    async def test_disk_below_high_water_skips(self, data_dir, monkeypatch):
        _override_settings(
            monkeypatch, data_dir,
            storage_auto_migrate_enabled=True,
            storage_auto_migrate_high_water=0.85,
        )
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        monkeypatch.setattr(scheduler_mod, "get_operator_identity", lambda: "op@x")
        _set_disk_used_fraction(monkeypatch, 0.50)
        sched = StorageMigrationScheduler(
            data_dir,
            mover_factory=lambda _i: pytest.fail("mover must not be built"),
        )
        result = await sched._tick()
        assert result.ran is False
        assert result.reason == "disk_ok"
        assert result.used_fraction == pytest.approx(0.50)


# ─────────────────────── migration path ───────────────────────


class TestMigration:
    @pytest.mark.asyncio
    async def test_migrates_when_disk_above_high_water(self, data_dir, monkeypatch):
        _override_settings(
            monkeypatch, data_dir,
            storage_auto_migrate_enabled=True,
            storage_auto_migrate_high_water=0.85,
            storage_auto_migrate_batch_limit=10,
        )
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        monkeypatch.setattr(scheduler_mod, "get_operator_identity", lambda: "op@x")
        _set_disk_used_fraction(monkeypatch, 0.90)

        sched = StorageMigrationScheduler(
            data_dir, mover_factory=_ok_mover_factory(data_dir),
        )
        result = await sched._tick()

        assert result.ran is True
        assert result.migrated == 1
        assert result.skipped == 0
        assert _doc_storage_path(data_dir, doc_id).startswith("satellite://node-a/")
        assert len(_satellite_blobs(data_dir)) == 1

    @pytest.mark.asyncio
    async def test_batch_limit_respected(self, data_dir, monkeypatch):
        _override_settings(
            monkeypatch, data_dir,
            storage_auto_migrate_enabled=True,
            storage_auto_migrate_high_water=0.0,  # always trigger
            storage_auto_migrate_batch_limit=2,
        )
        local = LocalStorageService(data_dir)
        ids = [_seed_document(data_dir, local, _CONTENT + bytes([i])) for i in range(3)]
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        monkeypatch.setattr(scheduler_mod, "get_operator_identity", lambda: "op@x")
        _set_disk_used_fraction(monkeypatch, 0.99)

        sched = StorageMigrationScheduler(
            data_dir, mover_factory=_ok_mover_factory(data_dir),
        )
        result = await sched._tick()

        assert result.ran is True
        assert result.migrated == 2
        remote = [i for i in ids if _doc_storage_path(data_dir, i).startswith("satellite://")]
        assert len(remote) == 2


# ─────────────────────── loop resilience + lifecycle ───────────────────────


class TestLoop:
    @pytest.mark.asyncio
    async def test_run_swallows_tick_errors(self, data_dir, monkeypatch):
        """A raising mover_factory must not kill the loop — _run records
        the error in ``latest`` and continues to the next sleep."""
        _override_settings(
            monkeypatch, data_dir,
            storage_auto_migrate_enabled=True,
            storage_auto_migrate_high_water=0.0,
        )
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        monkeypatch.setattr(scheduler_mod, "get_operator_identity", lambda: "op@x")
        _set_disk_used_fraction(monkeypatch, 0.99)

        def _boom(_identity):
            raise RuntimeError("boom")

        sched = StorageMigrationScheduler(data_dir, mover_factory=_boom)

        async def _short_sleep(_):
            sched._stopping = True  # break the loop after one iteration

        monkeypatch.setattr(scheduler_mod.asyncio, "sleep", _short_sleep)
        await sched._run(interval=1.0)  # must not raise

        assert sched._latest is not None
        assert sched._latest.ran is False
        assert sched._latest.reason and sched._latest.reason.startswith("error:")

    @pytest.mark.asyncio
    async def test_latest_returns_last_tick(self, data_dir, monkeypatch):
        _override_settings(monkeypatch, data_dir, storage_auto_migrate_enabled=False)
        sched = StorageMigrationScheduler(data_dir)
        assert sched.latest() is None
        result = await sched._tick()
        sched._latest = result
        assert sched.latest() is result

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, data_dir, monkeypatch):
        _override_settings(monkeypatch, data_dir, storage_auto_migrate_enabled=False)
        sched = StorageMigrationScheduler(data_dir)
        await sched.start(interval=3600.0)
        first_task = sched._task
        await sched.start(interval=3600.0)
        assert sched._task is first_task
        await sched.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_cleanly(self, data_dir, monkeypatch):
        _override_settings(monkeypatch, data_dir, storage_auto_migrate_enabled=False)
        sched = StorageMigrationScheduler(data_dir)
        await sched.start(interval=3600.0)
        # Give the task one chance to enter its first sleep.
        await asyncio.sleep(0)
        await sched.stop()
        assert sched._task is None
