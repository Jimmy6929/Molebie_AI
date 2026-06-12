"""Background vault auto-sync scheduler.

Vault sync shipped a manual hash-diff engine behind ``POST
/documents/vault/{id}/sync``. This module adds the freshness loop on top:
every ``vault_auto_sync_interval_sec`` it stat-scans each registered vault
(size + mtime signature — no reads, no hashes) and re-runs the existing
``sync_vault`` engine only when the signature changed.

Tick gates, ordered cheapest-first (mirrors ``StorageMigrationScheduler``):

    enabled? → any vaults? → root exists? → signature changed?
        → user not mid-ingest? → sync_vault()

The engine is unchanged — a false-positive signature change (mtime-only
touch) costs one no-dispatch sync in which hash-diff classifies everything
UNCHANGED, never a re-embed.

The signature cache is in-memory by design: the first tick after boot runs
one full (idempotent) sync per vault, which also picks up edits made while
the gateway was down.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from app.services.database import get_database_service
from app.services.vault_sync import (
    _scan_vault_signature,
    _vault_ignore_tokens,
    sync_vault,
)

_DEFAULT_INTERVAL_SEC = 300.0


@dataclass(slots=True)
class VaultAutoSyncTickResult:
    ran: bool
    reason: str | None = None
    checked: int = 0
    synced: int = 0
    skipped_unchanged: int = 0
    skipped_busy: int = 0
    skipped_missing: int = 0
    errors: list[str] = field(default_factory=list)


class VaultAutoSyncScheduler:
    """Periodically re-syncs registered vaults whose contents changed on disk."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._latest: VaultAutoSyncTickResult | None = None
        self._sig_cache: dict[str, str] = {}

    def latest(self) -> VaultAutoSyncTickResult | None:
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
                result = VaultAutoSyncTickResult(ran=False, reason=f"error: {exc}")
            self._latest = result
            if result.synced or result.errors:
                print(
                    f"[vault-auto-sync] synced={result.synced} "
                    f"unchanged={result.skipped_unchanged} "
                    f"errors={len(result.errors)}"
                )
            await asyncio.sleep(interval)

    async def _tick(self) -> VaultAutoSyncTickResult:
        # Late import keeps this module decoupled from config load order
        # (mirrors the pattern in storage_scheduler.py).
        from app.config import get_settings
        settings = get_settings()

        if not (
            settings.vault_auto_sync_enabled
            and settings.vault_sync_enabled
            and settings.rag_enabled
        ):
            return VaultAutoSyncTickResult(ran=False, reason="disabled")

        db = get_database_service()
        vaults = await db.list_all_vault_sources()
        if not vaults:
            return VaultAutoSyncTickResult(ran=False, reason="no_vaults")

        result = VaultAutoSyncTickResult(ran=True, checked=len(vaults))
        for vault in vaults:
            vault_id = vault["id"]
            try:
                root = Path(vault["root_path"]).expanduser()
                if not root.is_dir():
                    # Vault folder moved/deleted. Surfacing this is the vault
                    # status field's job — skipping quietly here avoids one
                    # log line per tick forever.
                    result.skipped_missing += 1
                    continue

                ignore_names, ignore_globs = _vault_ignore_tokens(vault, settings)
                sig = await asyncio.to_thread(
                    _scan_vault_signature,
                    root,
                    ignore_names,
                    ignore_globs,
                    bool(vault.get("index_attachments", 1)),
                )
                if sig == self._sig_cache.get(vault_id):
                    result.skipped_unchanged += 1
                    continue

                if await db.get_active_ingest_job(vault["user_id"]):
                    # A manual sync or folder upload is mid-flight. Skip
                    # WITHOUT caching the signature so the next tick retries.
                    result.skipped_busy += 1
                    continue

                await sync_vault(vault_id, vault["user_id"])
                # Cache the pre-sync signature: an edit landing mid-sync just
                # causes one cheap re-sync next tick (conservative direction).
                self._sig_cache[vault_id] = sig
                result.synced += 1
            except Exception as exc:  # noqa: BLE001
                # One bad vault never aborts the tick for the rest.
                result.errors.append(f"{vault.get('label', vault_id)}: {exc}")
        return result


_scheduler: VaultAutoSyncScheduler | None = None


def get_vault_auto_sync_scheduler() -> VaultAutoSyncScheduler:
    """Process-wide singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = VaultAutoSyncScheduler()
    return _scheduler


def reset_vault_auto_sync_scheduler() -> None:
    """Test-only helper."""
    global _scheduler
    _scheduler = None
