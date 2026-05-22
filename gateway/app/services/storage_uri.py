"""URI scheme for `documents.storage_path` / `message_images.storage_path`.

Pure functions. No I/O, no DB, no config. The shape this module captures:

* **`local://<relative-path>`** — file lives in the primary's local data
  dir, at ``data/documents/<relative-path>`` or
  ``data/images/<relative-path>`` depending on the table.
* **`satellite://<node-id>/<sha256>`** — file lives on the satellite
  identified by ``node-id``, addressable by its SHA-256 content digest.

Bare paths (the legacy `storage_path` shape: ``"<user-id>/<filename>"``)
are tolerated by ``parse`` for backward compatibility — they're treated
as `local://<value>`. This lets the codebase add URI-aware logic *now*
and migrate existing row values *later* (slice 9.3) without a flag-day.

This module is the contract that slice 9.3's ``TieredStorageService``
will build on: it'll dispatch reads to local-fs vs satellite-HTTP based
on ``parse(storage_path).scheme``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

Scheme = Literal["local", "satellite"]

_RECOGNIZED_SCHEMES = ("local", "satellite")


@dataclass(frozen=True)
class StorageURI:
    """Parsed form of a ``storage_path`` column value.

    * ``scheme`` — ``"local"`` or ``"satellite"``.
    * ``path_or_digest`` — for ``local``, the relative path under the
      data directory; for ``satellite``, the SHA-256 digest of the blob.
    * ``node_id`` — populated only for ``satellite`` URIs (the host
      component of the URI); ``None`` for ``local``.
    """

    scheme: Scheme
    path_or_digest: str
    node_id: str | None


def is_uri(value: str) -> bool:
    """True if ``value`` starts with one of the recognized schemes."""
    for scheme in _RECOGNIZED_SCHEMES:
        if value.startswith(f"{scheme}://"):
            return True
    return False


def parse(value: str) -> StorageURI:
    """Parse a stored ``storage_path``. Bare paths → ``local://<value>``.

    Raises ``ValueError`` on any URI-shaped input (``<scheme>://...``)
    whose scheme isn't ``local`` or ``satellite``. We treat unknown
    schemes as honest errors rather than silently coercing to bare-path
    — a caller using ``foo://x`` clearly meant a URI; masking that as
    relative path "foo://x" would hide a real bug.
    """
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme == "local":
            # ``local://user/file.pdf`` → netloc="user", path="/file.pdf".
            # Rejoin so the caller sees a single relative path "user/file.pdf".
            path = parsed.netloc + parsed.path
            return StorageURI(scheme="local", path_or_digest=path, node_id=None)
        if parsed.scheme == "satellite":
            # ``satellite://node-a/abc123`` → netloc="node-a", path="/abc123".
            digest = parsed.path.lstrip("/")
            return StorageURI(
                scheme="satellite",
                path_or_digest=digest,
                node_id=parsed.netloc,
            )
        raise ValueError(f"unknown storage URI scheme: {value!r}")

    # No "://" → bare path (legacy shape). Treat as local.
    return StorageURI(scheme="local", path_or_digest=value, node_id=None)


def build_local(relative_path: str) -> str:
    """Build a ``local://`` URI from a relative path under the data dir."""
    return f"local://{relative_path}"


def build_satellite(node_id: str, digest: str) -> str:
    """Build a ``satellite://`` URI naming the satellite + content digest."""
    return f"satellite://{node_id}/{digest}"
