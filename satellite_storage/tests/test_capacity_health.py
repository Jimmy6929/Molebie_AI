"""Tests for /v1/storage/capacity and /v1/storage/health."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from satellite_storage.app import create_app


@pytest.fixture
def client(tempdir_data_dir):
    app = create_app()
    with TestClient(app) as c:
        yield c


_AUTH = {"Tailscale-User-Login": "operator@example.com"}


class TestCapacity:
    def test_capacity_returns_disk_stats(self, client, tempdir_data_dir):
        r = client.get("/v1/storage/capacity", headers=_AUTH)
        assert r.status_code == 200
        j = r.json()
        assert set(j.keys()) == {"total_bytes", "used_bytes", "free_bytes", "data_dir"}
        assert j["total_bytes"] > 0
        assert j["free_bytes"] > 0
        assert j["data_dir"] == str(tempdir_data_dir)

    def test_capacity_requires_identity(self, client):
        r = client.get("/v1/storage/capacity")
        assert r.status_code == 401


class TestHealth:
    def test_health_unauthenticated(self, client):
        r = client.get("/v1/storage/health")
        assert r.status_code == 200
        j = r.json()
        assert "version" in j
        assert "uptime_sec" in j
        assert isinstance(j["uptime_sec"], int)

    def test_health_works_without_identity_header(self, client):
        # Same as above but explicit: no headers at all → still 200.
        r = client.get("/v1/storage/health", headers={})
        assert r.status_code == 200
