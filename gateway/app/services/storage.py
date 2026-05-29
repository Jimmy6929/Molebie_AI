"""
Filesystem + fleet storage services.

``LocalStorageService`` — single-machine mode. Reads/writes the local
``data/`` directory. ``storage_path`` values are bare paths (legacy) or
``local://`` URIs (slice 9.2 made it tolerant of both).

``TieredStorageService`` (slice 9.3) — fleet mode. Wraps a
``LocalStorageService`` and dispatches reads/deletes by URI scheme:
``local://`` goes to the local filesystem, ``satellite://<node>/<sha256>``
goes to that satellite's ``molebie-storage`` service over HTTP. Writes
always go local (local-first design — see the Storage Extension doc); the
background mover (slice 9.4) is what later migrates files to satellites
and rewrites their ``storage_path`` to ``satellite://`` URIs.

``get_storage_service`` picks the implementation: Local when no satellites
are registered (single-machine, unchanged), Tiered when ≥1 is.
"""

import uuid
from collections.abc import Callable
from pathlib import Path

import httpx

from app.services import storage_uri

_SATELLITE_STORAGE_PORT = 8090
_SATELLITE_HTTP_TIMEOUT_SEC = 10.0


class BlobNotFoundError(Exception):
    """A satellite returned 404 for a requested blob."""


class BlobAuthError(Exception):
    """A satellite rejected the request's Tailscale identity (401)."""


class BlobUnreachableError(Exception):
    """A satellite couldn't be reached, or returned an unexpected status."""


def _resolve_local_path(storage_path: str, base: Path) -> Path:
    """Convert a stored ``storage_path`` (bare path or ``local://`` URI)
    into a concrete filesystem path under ``base``.

    Raises ``NotImplementedError`` for ``satellite://`` URIs —
    ``LocalStorageService`` can't dispatch over HTTP; that's
    ``TieredStorageService``'s job (slice 9.3).
    """
    uri = storage_uri.parse(storage_path)
    if uri.scheme == "local":
        return base / uri.path_or_digest
    raise NotImplementedError(
        f"satellite:// storage requires TieredStorageService (slice 9.3); "
        f"got {storage_path!r}"
    )


class LocalStorageService:
    """Store and retrieve files from the local filesystem."""

    def __init__(self, data_dir: str):
        self.documents_dir = Path(data_dir) / "documents"
        self.images_dir = Path(data_dir) / "images"

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def upload_document(
        self, user_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        """Save a document file. Returns the storage_path."""
        user_dir = self.documents_dir / user_id
        self._ensure_dir(user_dir)
        storage_name = f"{uuid.uuid4().hex}_{filename}"
        storage_path = f"{user_id}/{storage_name}"
        (user_dir / storage_name).write_bytes(data)
        return storage_path

    def download_document(self, storage_path: str) -> bytes:
        """Read document bytes from storage."""
        full_path = _resolve_local_path(storage_path, self.documents_dir)
        return full_path.read_bytes()

    def delete_document(self, storage_path: str) -> None:
        """Delete a document file. Silently ignores missing files."""
        full_path = _resolve_local_path(storage_path, self.documents_dir)
        if full_path.exists():
            full_path.unlink()

    def upload_image(
        self, user_id: str, raw_bytes: bytes, mime_type: str, filename: str = "image"
    ) -> str:
        """Save an image file. Returns the storage_path."""
        user_dir = self.images_dir / user_id
        self._ensure_dir(user_dir)
        ext = mime_type.split("/")[-1]
        if ext == "jpeg":
            ext = "jpg"
        storage_name = f"{uuid.uuid4().hex}_{filename}.{ext}"
        storage_path = f"{user_id}/{storage_name}"
        (user_dir / storage_name).write_bytes(raw_bytes)
        return storage_path

    def download_image(self, storage_path: str) -> bytes:
        """Read image bytes from storage."""
        full_path = _resolve_local_path(storage_path, self.images_dir)
        return full_path.read_bytes()

    def delete_image(self, storage_path: str) -> None:
        """Delete an image file. Silently ignores missing files."""
        full_path = _resolve_local_path(storage_path, self.images_dir)
        if full_path.exists():
            full_path.unlink()


class TieredStorageService:
    """Fleet-mode storage. Wraps a ``LocalStorageService`` and dispatches
    reads/deletes by the URI scheme stored in ``storage_path``.

    Local-first: writes always delegate to the wrapped local service
    (so uploads never depend on a satellite being awake). Only the
    read/delete paths branch — ``satellite://`` URIs are dispatched to
    the owning satellite's ``molebie-storage`` HTTP service.

    ``resolve_satellite_host(node_id)`` maps a satellite node id to its
    Tailscale host (or None if unknown). ``operator_identity`` is the
    ``Tailscale-User-Login`` value sent on every outbound satellite call.
    ``http_client_factory`` is injectable for tests; defaults to a fresh
    ``httpx.Client`` per call.
    """

    def __init__(
        self,
        local: LocalStorageService,
        resolve_satellite_host: Callable[[str], str | None],
        operator_identity: str,
        http_client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.local = local
        self.resolve_satellite_host = resolve_satellite_host
        self.operator_identity = operator_identity
        self._client_factory = http_client_factory or httpx.Client

    # ----- writes: always local (local-first) -----

    def upload_document(
        self, user_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        return self.local.upload_document(user_id, filename, data, content_type)

    def upload_image(
        self, user_id: str, raw_bytes: bytes, mime_type: str, filename: str = "image"
    ) -> str:
        return self.local.upload_image(user_id, raw_bytes, mime_type, filename)

    # ----- reads / deletes: dispatch by URI scheme -----

    def download_document(self, storage_path: str) -> bytes:
        return self._read(storage_path, local_read=self.local.download_document)

    def download_image(self, storage_path: str) -> bytes:
        return self._read(storage_path, local_read=self.local.download_image)

    def delete_document(self, storage_path: str) -> None:
        self._delete(storage_path, local_delete=self.local.delete_document)

    def delete_image(self, storage_path: str) -> None:
        self._delete(storage_path, local_delete=self.local.delete_image)

    # ----- internals -----

    def _read(self, storage_path: str, local_read: Callable[[str], bytes]) -> bytes:
        uri = storage_uri.parse(storage_path)
        if uri.scheme == "local":
            return local_read(storage_path)
        # satellite://<node>/<digest>
        url = self._satellite_url(uri.node_id, uri.path_or_digest)
        resp = self._satellite_request("GET", url)
        return resp.content

    def _delete(self, storage_path: str, local_delete: Callable[[str], None]) -> None:
        uri = storage_uri.parse(storage_path)
        if uri.scheme == "local":
            local_delete(storage_path)
            return
        url = self._satellite_url(uri.node_id, uri.path_or_digest)
        # 404 on delete is fine — the blob's already gone (idempotent).
        self._satellite_request("DELETE", url, ok_missing=True)

    def _satellite_url(self, node_id: str | None, digest: str) -> str:
        host = self.resolve_satellite_host(node_id) if node_id else None
        if host is None:
            raise BlobUnreachableError(f"unknown satellite node {node_id!r}")
        return f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/blobs/{digest}"

    def _satellite_request(
        self, method: str, url: str, *, ok_missing: bool = False
    ) -> httpx.Response:
        headers = {"Tailscale-User-Login": self.operator_identity}
        try:
            with self._client_factory() as client:
                resp = client.request(
                    method, url, headers=headers, timeout=_SATELLITE_HTTP_TIMEOUT_SEC
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise BlobUnreachableError(f"satellite call failed: {exc}") from exc
        if resp.status_code == 404:
            if ok_missing:
                return resp
            raise BlobNotFoundError(f"blob not found on satellite: {url}")
        if resp.status_code == 401:
            raise BlobAuthError(f"satellite rejected identity: {url}")
        if resp.status_code >= 400:
            raise BlobUnreachableError(
                f"satellite returned HTTP {resp.status_code}: {url}"
            )
        return resp


_storage_service: "LocalStorageService | TieredStorageService | None" = None


def _has_registered_satellites() -> bool:
    """True if at least one satellite is registered in fleet_satellites."""
    import sqlite3

    from app.config import get_settings
    from app.schema import _get_db_path

    settings = get_settings()
    data_dir = getattr(settings, "data_dir", "data")
    try:
        conn = sqlite3.connect(_get_db_path(data_dir))
        try:
            row = conn.execute(
                "SELECT 1 FROM fleet_satellites LIMIT 1"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _resolve_satellite_host_from_db(node_id: str) -> str | None:
    """Look up a satellite's Tailscale host by node id from fleet_satellites."""
    import sqlite3

    from app.config import get_settings
    from app.schema import _get_db_path

    settings = get_settings()
    data_dir = getattr(settings, "data_dir", "data")
    try:
        conn = sqlite3.connect(_get_db_path(data_dir))
        try:
            row = conn.execute(
                "SELECT host FROM fleet_satellites WHERE id = ?", (node_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def get_storage_service() -> "LocalStorageService | TieredStorageService":
    """Get the cached storage service.

    Single-machine (no satellites registered) → ``LocalStorageService``,
    unchanged. Fleet mode (≥1 satellite) → ``TieredStorageService``, unless
    the primary has no Tailscale identity to authenticate outbound calls,
    in which case we fall back to local-only and log a warning.

    The chosen implementation is cached for the process lifetime, so a
    satellite that registers *after* a zero-satellite boot won't switch the
    gateway Local→Tiered until restart. Acceptable for now: no code path
    writes ``satellite://`` URIs until the slice 9.4 background mover, so a
    pre-mover gateway never needs the tiered read path mid-run.
    """
    global _storage_service
    if _storage_service is not None:
        return _storage_service

    from app.config import get_settings
    settings = get_settings()
    data_dir = getattr(settings, "data_dir", "data")
    local = LocalStorageService(data_dir)

    if not _has_registered_satellites():
        _storage_service = local
        return _storage_service

    from app.services.tailscale_outbound import get_operator_identity
    identity = get_operator_identity()
    if identity is None:
        print(
            "[storage] satellites registered but no Tailscale identity available "
            "— falling back to LocalStorageService (satellite reads will fail)"
        )
        _storage_service = local
        return _storage_service

    _storage_service = TieredStorageService(
        local=local,
        resolve_satellite_host=_resolve_satellite_host_from_db,
        operator_identity=identity,
    )
    print(f"[storage] using TieredStorageService (operator={identity})")
    return _storage_service


def reset_storage_service() -> None:
    """Drop the cached singleton — used by tests."""
    global _storage_service
    _storage_service = None
