"""
Local filesystem storage service.

Replaces Supabase Storage with simple file I/O in the data/ directory.
Documents stored in data/documents/{user_id}/, images in data/images/{user_id}/.
"""

import uuid
from pathlib import Path


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
        full_path = self.documents_dir / storage_path
        return full_path.read_bytes()

    def delete_document(self, storage_path: str) -> None:
        """Delete a document file. Silently ignores missing files."""
        full_path = self.documents_dir / storage_path
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
        full_path = self.images_dir / storage_path
        return full_path.read_bytes()

    def delete_image(self, storage_path: str) -> None:
        """Delete an image file. Silently ignores missing files."""
        full_path = self.images_dir / storage_path
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
