"""Integration tests for the blob-storage HTTP routes.

Uses FastAPI's TestClient. Each test gets a fresh tempdir via the
``tempdir_data_dir`` fixture and a fresh app via ``client``.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from satellite_storage.app import create_app


@pytest.fixture
def client(tempdir_data_dir):
    """FastAPI TestClient bound to a fresh app + tempdir per test."""
    app = create_app()
    with TestClient(app) as c:
        yield c


_AUTH = {"Tailscale-User-Login": "operator@example.com"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestPutBlob:
    def test_first_put_creates_blob(self, client, tempdir_data_dir):
        body = b"hello molebie"
        digest = _digest(body)
        r = client.put(f"/v1/storage/blobs/{digest}", content=body, headers=_AUTH)
        assert r.status_code == 201
        j = r.json()
        assert j == {"digest": digest, "size_bytes": len(body), "created": True}
        on_disk = tempdir_data_dir / "blobs" / digest[:2] / digest
        assert on_disk.exists()
        assert on_disk.read_bytes() == body

    def test_repeated_put_is_idempotent(self, client):
        body = b"hello molebie"
        digest = _digest(body)
        r1 = client.put(f"/v1/storage/blobs/{digest}", content=body, headers=_AUTH)
        r2 = client.put(f"/v1/storage/blobs/{digest}", content=body, headers=_AUTH)
        assert r1.status_code == 201
        assert r2.status_code == 200
        assert r2.json()["created"] is False

    def test_put_with_wrong_digest_returns_400(self, client):
        body = b"real content"
        wrong = "0" * 64
        r = client.put(f"/v1/storage/blobs/{wrong}", content=body, headers=_AUTH)
        assert r.status_code == 400
        assert "hash mismatch" in r.json()["detail"].lower()

    def test_put_without_identity_returns_401(self, client):
        body = b"x"
        digest = _digest(body)
        r = client.put(f"/v1/storage/blobs/{digest}", content=body)
        assert r.status_code == 401

    def test_put_malformed_digest_returns_422(self, client):
        body = b"x"
        # Path validator requires 64 hex chars; "not-a-hash" doesn't qualify.
        r = client.put("/v1/storage/blobs/not-a-hash", content=body, headers=_AUTH)
        assert r.status_code == 422


class TestGetBlob:
    def test_get_happy_path_returns_body(self, client):
        body = b"hello molebie"
        digest = _digest(body)
        client.put(f"/v1/storage/blobs/{digest}", content=body, headers=_AUTH)
        r = client.get(f"/v1/storage/blobs/{digest}", headers=_AUTH)
        assert r.status_code == 200
        assert r.content == body

    def test_get_missing_returns_404(self, client):
        digest = _digest(b"never uploaded")
        r = client.get(f"/v1/storage/blobs/{digest}", headers=_AUTH)
        assert r.status_code == 404

    def test_get_without_identity_returns_401(self, client):
        digest = _digest(b"x")
        r = client.get(f"/v1/storage/blobs/{digest}")
        assert r.status_code == 401


class TestHeadBlob:
    def test_head_present_returns_200_with_content_length(self, client):
        body = b"hello molebie"
        digest = _digest(body)
        client.put(f"/v1/storage/blobs/{digest}", content=body, headers=_AUTH)
        r = client.head(f"/v1/storage/blobs/{digest}", headers=_AUTH)
        assert r.status_code == 200
        assert r.headers["content-length"] == str(len(body))
        assert r.content == b""

    def test_head_missing_returns_404(self, client):
        digest = _digest(b"never")
        r = client.head(f"/v1/storage/blobs/{digest}", headers=_AUTH)
        assert r.status_code == 404


class TestDeleteBlob:
    def test_delete_present_returns_204(self, client, tempdir_data_dir):
        body = b"hello molebie"
        digest = _digest(body)
        client.put(f"/v1/storage/blobs/{digest}", content=body, headers=_AUTH)
        r = client.delete(f"/v1/storage/blobs/{digest}", headers=_AUTH)
        assert r.status_code == 204
        assert not (tempdir_data_dir / "blobs" / digest[:2] / digest).exists()

    def test_delete_missing_returns_404(self, client):
        digest = _digest(b"never")
        r = client.delete(f"/v1/storage/blobs/{digest}", headers=_AUTH)
        assert r.status_code == 404
