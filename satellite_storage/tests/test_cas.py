"""Unit tests for the content-addressable storage helper."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from satellite_storage.services.cas import (
    HashMismatch,
    blob_exists,
    blob_path,
    delete_blob,
    write_blob,
)


async def _chunks(data: bytes, chunk_size: int = 64):
    """Async iterable that yields ``data`` in fixed-size chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestCAS:
    def test_write_blob_creates_file_in_fanout_layout(self, tmp_path):
        body = b"hello molebie"
        digest = _digest(body)
        created, size = asyncio.run(write_blob(tmp_path, digest, _chunks(body)))
        assert created is True
        assert size == len(body)
        expected = tmp_path / "blobs" / digest[:2] / digest
        assert expected.exists()
        assert expected.read_bytes() == body

    def test_write_blob_is_idempotent_on_second_write(self, tmp_path):
        body = b"hello molebie"
        digest = _digest(body)
        first = asyncio.run(write_blob(tmp_path, digest, _chunks(body)))
        second = asyncio.run(write_blob(tmp_path, digest, _chunks(body)))
        assert first == (True, len(body))
        assert second == (False, len(body))

    def test_write_blob_rejects_hash_mismatch(self, tmp_path):
        body = b"this is the real content"
        wrong_digest = "0" * 64  # not the actual hash
        with pytest.raises(HashMismatch) as exc_info:
            asyncio.run(write_blob(tmp_path, wrong_digest, _chunks(body)))
        assert exc_info.value.expected == wrong_digest
        # Partial file removed.
        partial = tmp_path / "blobs" / wrong_digest[:2] / f".{wrong_digest}.partial"
        assert not partial.exists()
        # Target obviously not present.
        target = blob_path(tmp_path, wrong_digest)
        assert not target.exists()

    def test_write_blob_creates_fanout_dir(self, tmp_path):
        body = b"x"
        digest = _digest(body)
        # Parent doesn't exist yet.
        assert not (tmp_path / "blobs" / digest[:2]).exists()
        asyncio.run(write_blob(tmp_path, digest, _chunks(body)))
        assert (tmp_path / "blobs" / digest[:2]).is_dir()

    def test_delete_blob_returns_true_when_present(self, tmp_path):
        body = b"delete me"
        digest = _digest(body)
        asyncio.run(write_blob(tmp_path, digest, _chunks(body)))
        assert delete_blob(tmp_path, digest) is True
        assert not blob_path(tmp_path, digest).exists()

    def test_delete_blob_returns_false_when_missing(self, tmp_path):
        digest = "a" * 64
        assert delete_blob(tmp_path, digest) is False

    def test_blob_exists_reports_size(self, tmp_path):
        body = b"size matters"
        digest = _digest(body)
        assert blob_exists(tmp_path, digest) == (False, 0)
        asyncio.run(write_blob(tmp_path, digest, _chunks(body)))
        assert blob_exists(tmp_path, digest) == (True, len(body))

    def test_write_blob_handles_empty_chunks(self, tmp_path):
        body = b"real content"
        digest = _digest(body)

        async def _with_empty():
            yield b""
            yield body[:5]
            yield b""
            yield body[5:]
            yield b""

        created, size = asyncio.run(write_blob(tmp_path, digest, _with_empty()))
        assert created is True
        assert size == len(body)
        assert blob_path(tmp_path, digest).read_bytes() == body
