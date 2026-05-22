"""Blob storage endpoints — PUT/GET/HEAD/DELETE.

Every endpoint is Tailscale-identity-gated. The digest in the URL must
be a 64-char lowercase hex string; FastAPI rejects mis-shaped paths with
422 before we ever touch the filesystem.

Streaming on both sides:
* PUT consumes ``request.stream()`` chunk-by-chunk via CAS helper.
* GET returns ``FileResponse`` (zero-copy sendfile when available).
* HEAD is a stat-only existence check.

All endpoints share the prefix ``/v1/storage`` so the primary's future
``TieredStorageService`` can compose URLs like
``http://<satellite>:8090/v1/storage/blobs/<sha256>`` without any per-
satellite quirks.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse

from satellite_storage.config import get_settings
from satellite_storage.middleware.tailscale_identity import (
    TailscaleIdentity,
    get_tailscale_identity,
)
from satellite_storage.services.cas import (
    HashMismatch,
    blob_exists,
    blob_path,
    delete_blob,
    write_blob,
)

router = APIRouter(prefix="/v1/storage", tags=["Storage"])

_DIGEST_REGEX = r"^[0-9a-f]{64}$"


@router.put("/blobs/{digest}")
async def put_blob(
    request: Request,
    digest: str = Path(..., pattern=_DIGEST_REGEX),
    _identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> Response:
    """Upload a blob. Idempotent: re-uploading the same digest returns 200."""
    settings = get_settings()
    try:
        created, size = await write_blob(settings.data_dir, digest, request.stream())
    except HashMismatch as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    payload = {"digest": digest, "size_bytes": size, "created": created}
    return JSONResponse(
        content=payload,
        status_code=(status.HTTP_201_CREATED if created else status.HTTP_200_OK),
    )


@router.get("/blobs/{digest}")
async def get_blob(
    digest: str = Path(..., pattern=_DIGEST_REGEX),
    _identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> FileResponse:
    """Stream a blob's bytes back to the caller. 404 when absent."""
    settings = get_settings()
    path = blob_path(settings.data_dir, digest)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="blob not found")
    return FileResponse(path, media_type="application/octet-stream")


@router.head("/blobs/{digest}")
async def head_blob(
    digest: str = Path(..., pattern=_DIGEST_REGEX),
    _identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> Response:
    """Existence + size check, no body."""
    settings = get_settings()
    exists, size = blob_exists(settings.data_dir, digest)
    if not exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="blob not found")
    return Response(
        status_code=status.HTTP_200_OK,
        headers={"Content-Length": str(size)},
    )


@router.delete("/blobs/{digest}")
async def delete_blob_route(
    digest: str = Path(..., pattern=_DIGEST_REGEX),
    _identity: TailscaleIdentity = Depends(get_tailscale_identity),
) -> Response:
    """Delete a blob. 204 on success, 404 if absent."""
    settings = get_settings()
    deleted = delete_blob(settings.data_dir, digest)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="blob not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
