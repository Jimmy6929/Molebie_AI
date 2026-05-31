"""Background storage-migration scheduler — turns the 9.4a engine on.

Slice 9.4a shipped ``StorageMover`` plus a loopback ``POST
/fleet/storage/migrate`` so an operator could move files by hand. This
slice (9.4b) adds the *policy + loop* that calls the same engine
automatically when the primary's disk crosses a high-water mark.

The engine is unchanged — this module is pure policy-on-top. The tick
sequence (gates ordered cheapest-first; single-machine pays nothing):

    enabled? → satellites registered? → operator identity present?
        → disk above high-water?  → migrate one batch

Each tick migrates at most ``batch_limit`` documents and then sleeps;
under sustained pressure the loop drains gradually over successive
ticks, which keeps each iteration bounded and shutdown-responsive. The
9.4a engine is failure-safe (HEAD-verify before delete-local), so the
scheduler never opens a data-loss window even on partial failures.

Lifecycle mirrors ``StorageProbe`` (idempotent ``start``, graceful-cancel
``stop``). ``mover_factory`` is injectable so tests can supply a mover
wired with a fake httpx client.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.services.storage import LocalStorageService, _has_registered_satellites
from app.services.storage_mover import StorageMover
from app.services.tailscale_outbound import get_operator_identity

_DEFAULT_INTERVAL_SEC = 300.0


@dataclass(slots=True)
class MigrationTickResult:
    ran: bool
    reason: str | None = None
    used_fraction: float | None = None
    migrated: int = 0
    skipped: int = 0


class StorageMigrationScheduler:
    """Periodically migrates local documents to satellites under disk pressure."""

    def __init__(
        self,
        data_dir: str,
        mover_factory: Callable[[str], StorageMover] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._mover_factory = mover_factory or self._default_mover_factory
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._latest: MigrationTickResult | None = None

    def latest(self) -> MigrationTickResult | None:
        return self._latest

    async def start(self, interval: float = _DEFAULT_INTERVAL_SEC) -> None:
        """Idempotent — kicks off the background tick loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(interval))

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                # Shutdown path: we initiated the cancel, so swallow both the
                # expected CancelledError and any final error from the task to
                # guarantee stop() always completes.
                pass
            self._task = None

    async def _run(self, interval: float) -> None:
        while not self._stopping:
            try:
                result = await self._tick()
            except Exception as exc:
                # Soft fail — a single bad tick must not kill the loop.
                result = MigrationTickResult(ran=False, reason=f"error: {exc}")
            self._latest = result
            if result.ran:
                print(
                    f"[storage-scheduler] migrated={result.migrated} "
                    f"skipped={result.skipped} used={result.used_fraction:.2f}"
                )
            await asyncio.sleep(interval)

    async def _tick(self) -> MigrationTickResult:
        # Late import keeps this module decoupled from config load order
        # (mirrors the pattern in storage.py).
        from app.config import get_settings
        settings = get_settings()

        if not settings.storage_auto_migrate_enabled:
            return MigrationTickResult(ran=False, reason="disabled")

        # Single-machine fast path — no disk syscall, no mover built.
        if not _has_registered_satellites():
            return MigrationTickResult(ran=False, reason="no_satellite")

        identity = get_operator_identity()
        if identity is None:
            return MigrationTickResult(ran=False, reason="no_identity")

        usage = _measure_disk_usage(self._data_dir)
        used_fraction = usage.used / usage.total
        if used_fraction < settings.storage_auto_migrate_high_water:
            return MigrationTickResult(
                ran=False, reason="disk_ok", used_fraction=used_fraction
            )

        mover = self._mover_factory(identity)
        results = await asyncio.to_thread(
            mover.migrate_documents, settings.storage_auto_migrate_batch_limit
        )
        migrated = sum(1 for r in results if r.migrated)
        return MigrationTickResult(
            ran=True,
            used_fraction=used_fraction,
            migrated=migrated,
            skipped=len(results) - migrated,
        )

    def _default_mover_factory(self, identity: str) -> StorageMover:
        return StorageMover(
            LocalStorageService(self._data_dir), identity, self._data_dir
        )


def _measure_disk_usage(data_dir: str):
    """``shutil.disk_usage`` with a parent-fallback when data_dir hasn't
    been created yet — mirrors ``satellite_storage/routes/capacity.py``."""
    target = Path(data_dir)
    if not target.exists():
        target = target.parent
    return shutil.disk_usage(target)


_scheduler: StorageMigrationScheduler | None = None


def get_storage_scheduler(data_dir: str | None = None) -> StorageMigrationScheduler:
    """Process-wide singleton. First caller must pass ``data_dir``."""
    global _scheduler
    if _scheduler is None:
        if data_dir is None:
            raise RuntimeError(
                "StorageMigrationScheduler not yet initialised; "
                "pass data_dir on first call"
            )
        _scheduler = StorageMigrationScheduler(data_dir)
    return _scheduler


def reset_storage_scheduler() -> None:
    """Test-only helper."""
    global _scheduler
    _scheduler = None
