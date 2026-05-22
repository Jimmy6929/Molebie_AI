"""Content-addressable storage on the satellite's filesystem.

Files live at ``<data_dir>/blobs/<sha256[:2]>/<sha256>`` — Git-style fanout
so any single directory stays bounded to ~16K files at very large scale.

Two invariants this module enforces:

1. **The on-disk name is the content's SHA-256.** Two upload attempts of
   the same file produce the same name; the second is a no-op (free dedup,
   retry safety). The hash is verified during writes — mismatched bytes
   are rejected.

2. **Writes are atomic.** Streamed into ``.<digest>.partial`` next to the
   target, fsync'd, then renamed. A power-loss mid-write leaves either
   the partial (which the next probe will ignore / clean) or the final
   target — never a half-written blob with a valid-looking name.

No SQLite or other database on the satellite — the filesystem layout IS
the inventory. The primary's ``satellite_blobs`` table (slice 9.2) is
the authoritative record of *which satellite holds which blob* across
the fleet; this module only knows its own local files.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterable
from pathlib import Path


class HashMismatch(Exception):
    """Raised by ``write_blob`` when the streamed body doesn't hash to
    the expected digest. The partial file is unlinked before the
    exception is raised so the CAS layout never holds bad data."""

    def __init__(self, expected: str, got: str) -> None:
        super().__init__(f"hash mismatch: expected {expected!r}, got {got!r}")
        self.expected = expected
        self.got = got


def blob_path(data_dir: Path, digest: str) -> Path:
    """Where on disk does blob ``digest`` live?"""
    return data_dir / "blobs" / digest[:2] / digest


def blob_exists(data_dir: Path, digest: str) -> tuple[bool, int]:
    """Return ``(exists, size_bytes)``. Size is 0 when absent."""
    path = blob_path(data_dir, digest)
    if not path.exists():
        return (False, 0)
    return (True, path.stat().st_size)


async def write_blob(
    data_dir: Path,
    digest: str,
    body: AsyncIterable[bytes],
) -> tuple[bool, int]:
    """Stream ``body`` into the CAS layout at ``digest``.

    Returns ``(created, size_bytes)`` — ``created=False`` if the blob was
    already present (no write performed; idempotent), ``True`` if newly
    written. ``size_bytes`` is always the on-disk size after the call.

    Raises ``HashMismatch`` when the streamed bytes don't hash to
    ``digest``. The partial file is removed before raising.
    """
    target = blob_path(data_dir, digest)
    if target.exists():
        return (False, target.stat().st_size)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{digest}.partial"
    hasher = hashlib.sha256()
    bytes_written = 0

    # Sync file IO inside async route — v0.2 scale (single satellite,
    # one operator) doesn't justify aiofiles. Revisit if perf matters.
    with tmp.open("wb") as f:
        async for chunk in body:
            if not chunk:
                continue
            hasher.update(chunk)
            f.write(chunk)
            bytes_written += len(chunk)
        f.flush()
        os.fsync(f.fileno())

    computed = hasher.hexdigest()
    if computed != digest:
        tmp.unlink(missing_ok=True)
        raise HashMismatch(expected=digest, got=computed)

    # Atomic rename within the same FS — POSIX guarantees no half-states.
    tmp.replace(target)
    return (True, bytes_written)


def delete_blob(data_dir: Path, digest: str) -> bool:
    """Remove a blob. Returns True if a file was deleted, False if absent."""
    path = blob_path(data_dir, digest)
    if not path.exists():
        return False
    path.unlink()
    return True
