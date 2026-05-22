"""Pure-function tests for the storage URI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.storage_uri import (
    StorageURI,
    build_local,
    build_satellite,
    is_uri,
    parse,
)


class TestParse:
    def test_bare_path_treated_as_local(self):
        # Legacy `storage_path` shape: "<user-id>/<filename>".
        # Must parse as a local URI to preserve backward compatibility.
        uri = parse("user-abc/file.pdf")
        assert uri == StorageURI(
            scheme="local", path_or_digest="user-abc/file.pdf", node_id=None
        )

    def test_local_uri(self):
        uri = parse("local://user-abc/file.pdf")
        assert uri == StorageURI(
            scheme="local", path_or_digest="user-abc/file.pdf", node_id=None
        )

    def test_satellite_uri(self):
        uri = parse("satellite://node-a/abc123")
        assert uri == StorageURI(
            scheme="satellite", path_or_digest="abc123", node_id="node-a"
        )

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="unknown storage URI scheme"):
            parse("unknown://x")

    def test_local_uri_with_nested_path(self):
        # Two-level relative path round-trips cleanly through urlparse.
        uri = parse("local://user-abc/deep/nested/file.pdf")
        assert uri.scheme == "local"
        assert uri.path_or_digest == "user-abc/deep/nested/file.pdf"


class TestIsUri:
    def test_recognizes_local(self):
        assert is_uri("local://anything") is True

    def test_recognizes_satellite(self):
        assert is_uri("satellite://node/digest") is True

    def test_rejects_bare_path(self):
        assert is_uri("user/file.pdf") is False
        assert is_uri("just-a-filename") is False


class TestBuilders:
    def test_build_local(self):
        assert build_local("user-abc/file.pdf") == "local://user-abc/file.pdf"

    def test_build_satellite(self):
        assert build_satellite("node-a", "abc123") == "satellite://node-a/abc123"

    def test_round_trip_local(self):
        # parse(build_local(x)).path_or_digest == x
        for path in ("user/file", "u/deep/path/file.pdf", "x/y"):
            assert parse(build_local(path)).path_or_digest == path

    def test_round_trip_satellite(self):
        node = "home-server"
        digest = "a" * 64
        uri = parse(build_satellite(node, digest))
        assert uri.scheme == "satellite"
        assert uri.node_id == node
        assert uri.path_or_digest == digest
