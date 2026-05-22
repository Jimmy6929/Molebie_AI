"""
Local filesystem storage service.

Replaces Supabase Storage with simple file I/O in the data/ directory.
Documents stored in data/documents/{user_id}/, images in data/images/{user_id}/.

URI-tolerance (added in slice 9.2): ``storage_path`` arguments accept both
bare paths (the legacy shape: ``"<user-id>/<filename>"``) and ``local://``
URIs. Writes still produce bare paths — the URI-on-write flip happens in
slice 9.3 when ``TieredStorageService`` takes over routing. ``satellite://``
URIs raise ``NotImplementedError`` here; that's the contract slice 9.3
fulfills.
"""

import uuid
from pathlib import Path

from app.services import storage_uri


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


_storage_service: LocalStorageService | None = None


def get_storage_service() -> LocalStorageService:
    """Get cached LocalStorageService instance."""
    global _storage_service
    if _storage_service is None:
        from app.config import get_settings
        settings = get_settings()
        data_dir = getattr(settings, "data_dir", "data")
        _storage_service = LocalStorageService(data_dir)
    return _storage_service
