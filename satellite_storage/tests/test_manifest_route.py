"""Tests for /v1/storage/manifest — slice 9.5."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from satellite_storage.app import create_app


@pytest.fixture
def client(tempdir_data_dir):
    app = create_app()
    with TestClient(app) as c:
        yield c


_AUTH = {"Tailscale-User-Login": "operator@example.com"}


def _plant_blob(data_dir: Path, content: bytes) -> tuple[str, int]:
    """Write a fake blob directly into the CAS layout; returns (sha256, size)."""
    digest = hashlib.sha256(content).hexdigest()
    target = data_dir / "blobs" / digest[:2] / digest
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return digest, len(content)


class TestManifest:
    def test_empty_manifest_when_no_blobs(self, client, tempdir_data_dir):
        r = client.get("/v1/storage/manifest", headers=_AUTH)
        assert r.status_code == 200
        j = r.json()
        assert j["blobs"] == []
        assert "generated_at" in j

    def test_populated_manifest_sorted_by_sha256(self, client, tempdir_data_dir):
        # Plant 3 blobs in non-sorted order; manifest must come back sorted.
        contents = [b"alpha", b"bravo", b"charlie"]
        planted = [_plant_blob(tempdir_data_dir, c) for c in contents]

        r = client.get("/v1/storage/manifest", headers=_AUTH)
        assert r.status_code == 200
        blobs = r.json()["blobs"]
        assert len(blobs) == 3

        # Sorted ascending by sha256
        shas = [b["sha256"] for b in blobs]
        assert shas == sorted(shas)

        # Each entry has the right size
        expected = {d: s for d, s in planted}
        for b in blobs:
            assert b["size_bytes"] == expected[b["sha256"]]

    def test_partial_files_excluded(self, client, tempdir_data_dir):
        # A real blob alongside a .partial in-flight file.
        digest, _ = _plant_blob(tempdir_data_dir, b"completed")
        partial = tempdir_data_dir / "blobs" / digest[:2] / f".{digest}.partial"
        partial.write_bytes(b"in-flight bytes")

        r = client.get("/v1/storage/manifest", headers=_AUTH)
        assert r.status_code == 200
        blobs = r.json()["blobs"]
        assert len(blobs) == 1
        assert blobs[0]["sha256"] == digest

    def test_manifest_requires_identity(self, client):
        r = client.get("/v1/storage/manifest")
        assert r.status_code == 401
