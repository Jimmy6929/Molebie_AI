"""
Vault source endpoints — connect / list / sync / disconnect Obsidian-style
external folders. Sync delegates to `app.services.vault_sync.sync_vault`,
which reuses the folder-ingest worker so progress streams through the
existing `/documents/folder/{job_id}/events` SSE channel.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.middleware.auth import JWTPayload, get_current_user
from app.services.database import get_database_service
from app.services.vault_sync import _check_root_allowed, sync_vault

router = APIRouter(prefix="/documents/vault", tags=["Vault Sync"])


# ── Schemas ──────────────────────────────────────────────────────────────


class ConnectVaultRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    root_path: str = Field(..., min_length=1)
    exclude_globs: list[str] | None = None
    index_attachments: bool = True


class UpdateVaultRequest(BaseModel):
    label: str | None = Field(None, min_length=1, max_length=200)
    root_path: str | None = Field(None, min_length=1)


class VaultInfo(BaseModel):
    id: str
    label: str
    root_path: str
    kind: str
    last_sync_at: str | None
    index_attachments: bool
    doc_count: int
    status: str = "ok"  # "ok" | "path_missing"


class VaultListResponse(BaseModel):
    vaults: list[VaultInfo]


class SyncClassification(BaseModel):
    new: int
    changed: int
    unchanged: int
    deleted: int
    adopted: int
    skipped: int


class SyncVaultResponse(BaseModel):
    job_id: str | None
    classification: SyncClassification
    errors: list[str] = []


class DisconnectResponse(BaseModel):
    deleted_documents: int


# ── Helpers ──────────────────────────────────────────────────────────────


def _row_to_info(row: dict) -> VaultInfo:
    return VaultInfo(
        id=row["id"],
        label=row["label"],
        root_path=row["root_path"],
        kind=row.get("kind") or "obsidian",
        last_sync_at=row.get("last_sync_at"),
        index_attachments=bool(row.get("index_attachments", 1)),
        doc_count=int(row.get("doc_count", 0)),
        # One stat per vault at list time — lets the UI flag a moved/renamed
        # folder instead of waiting for the next sync to 400.
        status=(
            "ok"
            if Path(row["root_path"]).expanduser().is_dir()
            else "path_missing"
        ),
    )


def _validate_root_or_raise(root_str: str) -> Path:
    settings = get_settings()
    if not settings.vault_sync_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault sync disabled (VAULT_SYNC_ENABLED=false)",
        )
    if not settings.rag_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault sync requires RAG to be enabled.",
        )
    root = Path(root_str).expanduser()
    if not root.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"root_path does not exist: {root}",
        )
    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"root_path is not a directory: {root}",
        )
    reason = _check_root_allowed(root)
    if reason:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
    return root


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("/connect", response_model=VaultInfo)
async def connect_vault(
    body: ConnectVaultRequest,
    user: JWTPayload = Depends(get_current_user),
):
    """Register an external folder as a vault source. Idempotent on root_path:
    reconnecting the same path returns the existing vault row."""
    root = _validate_root_or_raise(body.root_path)
    db = get_database_service()
    user_id = user.user_id
    canonical = str(root.resolve())

    existing = await db.get_vault_source_by_root(user_id, canonical)
    if existing:
        existing["doc_count"] = 0  # backfilled by the GET /documents/vault list
        return _row_to_info(existing)

    row = await db.insert_vault_source(
        user_id=user_id,
        label=body.label.strip(),
        root_path=canonical,
        kind="obsidian",
        exclude_globs=body.exclude_globs,
        index_attachments=body.index_attachments,
    )
    row["doc_count"] = 0
    return _row_to_info(row)


@router.get("", response_model=VaultListResponse)
async def list_vaults(user: JWTPayload = Depends(get_current_user)):
    db = get_database_service()
    rows = await db.list_vault_sources(user.user_id)
    return VaultListResponse(vaults=[_row_to_info(r) for r in rows])


@router.post("/{vault_id}/sync", response_model=SyncVaultResponse)
async def trigger_sync(
    vault_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    """Hash-diff the vault against existing documents, dispatch deltas
    through the folder-ingest worker. Client should subscribe to
    `/documents/folder/{job_id}/events` for live progress when `job_id` is
    non-null."""
    db = get_database_service()
    user_id = user.user_id

    vault = await db.get_vault_source(vault_id, user_id)
    if not vault:
        raise HTTPException(status_code=404, detail="Vault not found")

    # Guard against an existing in-flight ingest job (folder upload OR vault
    # sync) — the worker is single-active per user. Lets the client know to
    # wait or cancel rather than silently double-running.
    active = await db.get_active_ingest_job(user_id)
    if active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "An ingest job is already running for this user",
                "job_id": active["id"],
                "status": active["status"],
                "root_label": active.get("root_label"),
            },
        )

    try:
        report = await sync_vault(vault_id, user_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return SyncVaultResponse(
        job_id=report.job_id,
        classification=SyncClassification(
            new=report.new,
            changed=report.changed,
            unchanged=report.unchanged,
            deleted=report.deleted,
            adopted=report.adopted,
            skipped=report.skipped,
        ),
        errors=report.errors,
    )


@router.patch("/{vault_id}", response_model=VaultInfo)
async def update_vault(
    vault_id: str,
    body: UpdateVaultRequest,
    user: JWTPayload = Depends(get_current_user),
):
    """Update a vault's label and/or root_path. Re-pointing root_path is the
    recovery path for a moved/renamed vault folder: documents are kept, and
    the next sync's hash-diff classifies identical files UNCHANGED — nothing
    re-embeds. Label-only updates never validate the current root, so a vault
    whose folder is missing can still be renamed."""
    db = get_database_service()
    user_id = user.user_id

    vault = await db.get_vault_source(vault_id, user_id)
    if not vault:
        raise HTTPException(status_code=404, detail="Vault not found")

    if body.label is None and body.root_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide label and/or root_path",
        )

    new_root: str | None = None
    if body.root_path is not None:
        root = _validate_root_or_raise(body.root_path)
        new_root = str(root.resolve())
        other = await db.get_vault_source_by_root(user_id, new_root)
        if other and other["id"] != vault_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Path already registered to vault '{other['label']}'",
            )

    try:
        row = await db.update_vault_source(
            vault_id,
            user_id,
            label=body.label.strip() if body.label else None,
            root_path=new_root,
        )
    except sqlite3.IntegrityError:
        # TOCTOU between the pre-check above and the UPDATE — the unique
        # (user_id, root_path) index is the source of truth.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Path already registered to another vault",
        )
    if not row:
        raise HTTPException(status_code=404, detail="Vault not found")

    docs = await db.list_vault_documents(vault_id, user_id)
    row["doc_count"] = len(docs)
    return _row_to_info(row)


@router.delete("/{vault_id}", response_model=DisconnectResponse)
async def disconnect_vault(
    vault_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    """Remove the vault source AND every document/chunk it owns."""
    db = get_database_service()
    vault = await db.get_vault_source(vault_id, user.user_id)
    if not vault:
        raise HTTPException(status_code=404, detail="Vault not found")
    deleted = await db.delete_vault_source(vault_id, user.user_id)
    return DisconnectResponse(deleted_documents=deleted)
