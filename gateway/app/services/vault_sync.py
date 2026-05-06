"""
Vault sync — hash-diff an external folder (Obsidian vault) against the
documents already in Molebie's index, then dispatch only the deltas.

Design:
- New / changed files → uploaded into local storage and enqueued into the
  existing folder-ingest worker (same extract → chunk → embed path).
- Changed files first have the old `documents` row hard-deleted via
  `db.delete_document` so chunks/vec rows are cleaned in one shot.
- Deleted files → straight `db.delete_document` per document.

The vault root never moves into storage as a unit. Only file bytes that
need (re)embedding are copied — unchanged files are skipped entirely.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.database import get_database_service
from app.services.ingest_worker import get_ingest_worker
from app.services.storage import get_storage_service


# Markdown is always indexed. PDFs and plaintext join when index_attachments=1.
_MARKDOWN_EXTS = frozenset({".md", ".markdown"})
_ATTACHMENT_EXTS = frozenset({".pdf", ".txt"})

_EXT_TO_MIME = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}


@dataclass
class SyncReport:
    job_id: str | None
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    adopted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _split_ignore_tokens(s: str) -> tuple[list[str], list[str]]:
    """Split a comma-separated ignore string into (segment_names, glob_patterns).

    Mirrors `folder_ingest._split_globs` — keeping behavior consistent so a
    user's mental model from the folder-upload flow carries over.
    """
    names: list[str] = []
    globs: list[str] = []
    for raw in s.split(","):
        token = raw.strip()
        if not token:
            continue
        if "*" in token or "?" in token:
            globs.append(token)
        else:
            names.append(token)
    return names, globs


def _path_is_ignored(rel: str, names: list[str], globs: list[str]) -> bool:
    segments = rel.split("/")
    if any(s in names for s in segments):
        return True
    base = segments[-1] if segments else rel
    return any(fnmatch.fnmatch(base, g) for g in globs)


def _ext_of(rel: str) -> str:
    dot = rel.rfind(".")
    return rel[dot:].lower() if dot >= 0 else ""


def _is_indexable(rel: str, ext: str, index_attachments: bool) -> bool:
    if ext in _MARKDOWN_EXTS:
        return True
    if index_attachments and ext in _ATTACHMENT_EXTS:
        return True
    return False


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_under(root: Path, child: Path) -> bool:
    """True iff `child.resolve()` is contained within `root.resolve()`.

    Guards against symlink escapes — a vault that contains a symlink to
    /etc/passwd should not result in /etc/passwd being read and embedded.
    """
    try:
        return child.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _check_root_allowed(root: Path) -> str | None:
    """Return None if the resolved root is under VAULT_ALLOWED_ROOTS (or the
    user's home when that setting is empty). Otherwise return a reason string
    suitable for an HTTP 400 detail."""
    settings = get_settings()
    raw = (settings.vault_allowed_roots or "").strip()
    allowed: list[Path] = []
    if raw:
        for token in raw.split(","):
            token = token.strip()
            if token:
                allowed.append(Path(token).expanduser().resolve())
    else:
        try:
            allowed.append(Path.home().resolve())
        except RuntimeError:
            return "Cannot resolve user home; set VAULT_ALLOWED_ROOTS"

    try:
        resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        return f"Cannot resolve root_path: {exc}"
    for prefix in allowed:
        try:
            if resolved.is_relative_to(prefix):
                return None
        except ValueError:
            continue
    return f"root_path must lie under one of: {', '.join(str(p) for p in allowed)}"


# ───────────────────────────── Walking ─────────────────────────────


@dataclass
class _VaultFile:
    relative_path: str
    abs_path: Path
    size: int
    sha256: str
    content_type: str


def _walk_vault_sync(
    root: Path,
    ignore_names: list[str],
    ignore_globs: list[str],
    index_attachments: bool,
    max_size: int,
) -> tuple[dict[str, _VaultFile], list[str]]:
    """Synchronous vault walk — runs off-thread. Returns (files_by_path, skipped).

    `files_by_path` is keyed on `relative_path` (vault-root-relative POSIX),
    one entry per indexable file. `skipped` is a list of human-readable reasons
    (e.g. ``"oversize: foo.pdf (62MB)"``) suitable for surfacing in the SyncReport.
    """
    files: dict[str, _VaultFile] = {}
    skipped: list[str] = []
    root_resolved = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune ignored directories before recursing — saves walking node_modules
        # and the like. We rebuild dirnames in place per os.walk's protocol.
        pruned = []
        for d in dirnames:
            full_segments = os.path.relpath(
                os.path.join(dirpath, d), str(root)
            ).replace(os.sep, "/")
            if _path_is_ignored(full_segments, ignore_names, ignore_globs):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        for fname in filenames:
            abs_path = Path(dirpath) / fname
            try:
                rel = str(abs_path.relative_to(root)).replace(os.sep, "/")
            except ValueError:
                continue

            if _path_is_ignored(rel, ignore_names, ignore_globs):
                continue

            ext = _ext_of(rel)
            if not _is_indexable(rel, ext, index_attachments):
                continue

            # Skip iCloud "evicted" placeholders (size 0, dotfile shadow).
            try:
                stat = abs_path.lstat()
            except OSError:
                continue
            if stat.st_size == 0 and fname.startswith("."):
                continue

            # Reject symlinks that point outside the vault root.
            if abs_path.is_symlink() and not _resolve_under(root_resolved, abs_path):
                skipped.append(f"symlink_escape: {rel}")
                continue

            size = int(stat.st_size)
            if size > max_size:
                skipped.append(f"too_large: {rel} ({size} bytes)")
                continue

            try:
                data = abs_path.read_bytes()
            except OSError as exc:
                skipped.append(f"read_error: {rel} ({exc})")
                continue

            sha = _hash_bytes(data)
            content_type = _EXT_TO_MIME.get(ext, "text/plain")
            files[rel] = _VaultFile(
                relative_path=rel,
                abs_path=abs_path,
                size=size,
                sha256=sha,
                content_type=content_type,
            )

    return files, skipped


# ───────────────────────────── Public API ─────────────────────────────


async def sync_vault(vault_id: str, user_id: str) -> SyncReport:
    """Run a hash-diff sync of a vault against existing documents.

    Returns a SyncReport. If any new/changed files are dispatched, the report
    carries the `job_id` of the spawned ingest job — the caller should stream
    `/documents/folder/{job_id}/events` for live progress.
    """
    settings = get_settings()
    if not settings.vault_sync_enabled:
        raise RuntimeError("vault sync disabled (VAULT_SYNC_ENABLED=false)")
    if not settings.rag_enabled:
        raise RuntimeError("RAG must be enabled to sync vaults")

    db = get_database_service()
    storage = get_storage_service()

    vault = await db.get_vault_source(vault_id, user_id)
    if not vault:
        raise FileNotFoundError(f"vault {vault_id} not found")

    root = Path(vault["root_path"]).expanduser()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"root_path no longer exists: {root}")

    # Build ignore set: defaults + per-vault overrides.
    import json
    extra: list[str] = []
    if vault.get("exclude_globs"):
        try:
            extra = list(json.loads(vault["exclude_globs"]) or [])
        except (json.JSONDecodeError, TypeError):
            extra = []
    ignore_str = settings.vault_default_ignore + (
        "," + ",".join(extra) if extra else ""
    )
    ignore_names, ignore_globs = _split_ignore_tokens(ignore_str)
    index_attachments = bool(vault.get("index_attachments", 1))

    # 1. Walk the vault off-thread (file I/O + hashing).
    files, skipped = await asyncio.to_thread(
        _walk_vault_sync,
        root,
        ignore_names,
        ignore_globs,
        index_attachments,
        settings.document_max_file_size,
    )

    # 2. Pull existing vault documents for hash diff.
    existing_rows = await db.list_vault_documents(vault_id, user_id)
    existing_by_path: dict[str, dict[str, Any]] = {
        r["relative_path"]: r for r in existing_rows if r.get("relative_path")
    }

    # 3. Classify. Before declaring a file NEW, try to adopt an existing
    # unattached document whose SHA256 matches — this is what kicks in the
    # first time a user connects a vault that was previously imported via
    # the regular folder-upload flow. Adoption avoids re-embedding identical
    # content; the doc just changes its `vault_source_id` and relative_path.
    new_paths: list[str] = []
    changed_paths: list[str] = []
    unchanged_paths: list[str] = []
    adopted_paths: list[str] = []
    adopted_doc_ids: set[str] = set()  # never adopt the same doc twice in one sync

    for rel, vf in files.items():
        prior = existing_by_path.get(rel)
        if prior is None:
            candidate = await db.find_unattached_document_by_hash(user_id, vf.sha256)
            if candidate and candidate["id"] not in adopted_doc_ids:
                adopted = await db.adopt_document_into_vault(
                    candidate["id"], vault_id, rel, user_id
                )
                if adopted:
                    adopted_doc_ids.add(candidate["id"])
                    adopted_paths.append(rel)
                    continue
            new_paths.append(rel)
        elif (prior.get("file_hash") or "") != vf.sha256:
            changed_paths.append(rel)
        else:
            unchanged_paths.append(rel)

    deleted_paths: list[str] = [
        rel for rel in existing_by_path if rel not in files
    ]

    # 4. Apply size guardrail to NEW + CHANGED.
    to_dispatch = new_paths + changed_paths
    if len(to_dispatch) > settings.vault_max_files_per_sync:
        raise ValueError(
            f"Sync would touch {len(to_dispatch)} files, exceeds "
            f"VAULT_MAX_FILES_PER_SYNC={settings.vault_max_files_per_sync}. "
            "Narrow exclude_globs or split the vault."
        )

    report = SyncReport(
        job_id=None,
        new=len(new_paths),
        changed=len(changed_paths),
        unchanged=len(unchanged_paths),
        deleted=0,
        adopted=len(adopted_paths),
        skipped=len(skipped),
        errors=skipped,
    )

    # 5. DELETED files first — keeps the live index honest even if dispatch fails.
    for rel in deleted_paths:
        prior = existing_by_path.get(rel)
        if not prior:
            continue
        try:
            doc = await db.get_document(prior["id"], user_id)
            await db.delete_document(prior["id"], user_id)
            if doc and doc.get("storage_path"):
                try:
                    await asyncio.to_thread(
                        storage.delete_document, doc["storage_path"]
                    )
                except OSError:
                    pass
            report.deleted += 1
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"delete_failed: {rel} ({exc})")

    # 6. CHANGED files: drop the old documents before re-ingesting. The new
    # job will create fresh document rows.
    for rel in changed_paths:
        prior = existing_by_path.get(rel)
        if not prior:
            continue
        try:
            doc = await db.get_document(prior["id"], user_id)
            await db.delete_document(prior["id"], user_id)
            if doc and doc.get("storage_path"):
                try:
                    await asyncio.to_thread(
                        storage.delete_document, doc["storage_path"]
                    )
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"replace_cleanup_failed: {rel} ({exc})")

    # 7. NEW + CHANGED dispatch via the existing folder-ingest worker.
    if to_dispatch:
        accepted_for_db: list[dict[str, Any]] = []
        for rel in to_dispatch:
            vf = files[rel]
            accepted_for_db.append({
                "relative_path": rel,
                "file_size": vf.size,
                "content_type": vf.content_type,
            })

        job = await db.create_ingest_job(
            user_id,
            f"vault:{vault['label']}",
            accepted_for_db,
            vault_source_id=vault_id,
        )
        job_id = job["id"]
        report.job_id = job_id

        # Stage bytes into storage and flip each file row to 'uploaded' so the
        # worker picks it up. We re-read the file via the cached dataclass
        # (already in memory) — avoids a second disk read.
        file_id_by_path = {
            a["relative_path"]: a["file_id"] for a in job["accepted_files"]
        }
        for rel in to_dispatch:
            vf = files[rel]
            file_id = file_id_by_path.get(rel)
            if not file_id:
                continue
            try:
                # Upload bytes into local storage. Read once more here to
                # avoid keeping all vault bytes resident across the dispatch
                # phase — for a large vault this matters.
                data = await asyncio.to_thread(vf.abs_path.read_bytes)
                filename = rel.rsplit("/", 1)[-1] or rel
                storage_path = await asyncio.to_thread(
                    storage.upload_document,
                    user_id,
                    filename,
                    data,
                    vf.content_type,
                )
                await db.update_ingest_file_status(
                    file_id,
                    "uploaded",
                    storage_path=storage_path,
                    file_hash=vf.sha256,
                    content_type=vf.content_type,
                )
            except Exception as exc:  # noqa: BLE001
                await db.update_ingest_file_status(
                    file_id, "failed", error_message=f"stage_failed: {exc}"
                )
                report.errors.append(f"stage_failed: {rel} ({exc})")

        # Wake the worker.
        await get_ingest_worker().ensure_worker_started(job_id)

    # 8. Stamp the sync time, even if no dispatch happened — so the UI can
    # show "last synced 2s ago" for a no-op run.
    await db.update_vault_source_synced_at(vault_id)

    return report
