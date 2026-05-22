"""Tests for ``LocalStorageService`` URI tolerance (added in slice 9.2).

Verifies:
1. Bare paths still work (current behavior, no regression).
2. ``local://`` URIs work identically — same file, same bytes.
3. ``satellite://`` URIs raise ``NotImplementedError`` — that's the
   contract slice 9.3's ``TieredStorageService`` will fulfill.
4. Both bare and URI formats consistent across read + delete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.storage import LocalStorageService


@pytest.fixture
def storage_with_file(tmp_path):
    """LocalStorageService with one document already on disk."""
    svc = LocalStorageService(str(tmp_path))
    # Plant the file directly to avoid coupling the test to upload_*'s
    # uuid-prefixed naming.
    user_dir = svc.documents_dir / "user-abc"
    user_dir.mkdir(parents=True)
    (user_dir / "file.pdf").write_bytes(b"hello molebie")
    yield svc, tmp_path


class TestUriTolerance:
    def test_download_document_with_bare_path(self, storage_with_file):
        svc, _ = storage_with_file
        assert svc.download_document("user-abc/file.pdf") == b"hello molebie"

    def test_download_document_with_local_uri(self, storage_with_file):
        svc, _ = storage_with_file
        # Same file, same bytes, accessed via URI form.
        assert svc.download_document("local://user-abc/file.pdf") == b"hello molebie"

    def test_download_document_with_satellite_uri_raises(self, storage_with_file):
        svc, _ = storage_with_file
        with pytest.raises(NotImplementedError, match="TieredStorageService"):
            svc.download_document("satellite://node-a/abc123")

    def test_delete_document_accepts_both_formats(self, storage_with_file):
        svc, tmp_path = storage_with_file
        target = tmp_path / "documents" / "user-abc" / "file.pdf"

        # First delete via bare path.
        svc.delete_document("user-abc/file.pdf")
        assert not target.exists()

        # Re-create and delete via local:// URI — same result.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"again")
        svc.delete_document("local://user-abc/file.pdf")
        assert not target.exists()
