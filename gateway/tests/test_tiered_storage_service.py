"""Tests for TieredStorageService — URI-aware read/delete dispatch.

Local reads/deletes delegate to a real LocalStorageService over a
tempdir. Satellite reads/deletes are exercised by mocking the injected
httpx client factory, so no real network is involved. Satellite URIs are
planted directly (no production code path writes them yet — that's slice
9.4's background mover).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.storage import (
    BlobAuthError,
    BlobNotFoundError,
    BlobUnreachableError,
    LocalStorageService,
    TieredStorageService,
)


@pytest.fixture
def local(tmp_path):
    """Real LocalStorageService over a tempdir, with one document planted."""
    svc = LocalStorageService(str(tmp_path))
    user_dir = svc.documents_dir / "user-abc"
    user_dir.mkdir(parents=True)
    (user_dir / "file.pdf").write_bytes(b"local bytes")
    return svc


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


class _FakeClient:
    """Captures the last request; returns a programmed response."""

    def __init__(self, response: _FakeResponse | Exception):
        self._response = response
        self.calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def request(self, method, url, headers=None, timeout=None):
        self.calls.append((method, url, headers))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _tiered(local, response=None, host_map=None):
    """Build a TieredStorageService with a mocked client + host resolver."""
    client = _FakeClient(response) if response is not None else _FakeClient(_FakeResponse(200))
    host_map = host_map or {"node-a": "100.64.0.9"}
    svc = TieredStorageService(
        local=local,
        resolve_satellite_host=lambda nid: host_map.get(nid),
        operator_identity="operator@example.com",
        http_client_factory=lambda: client,
    )
    return svc, client


_DIGEST = "a" * 64


class TestLocalDispatch:
    def test_download_bare_path_delegates_local(self, local):
        svc, client = _tiered(local)
        assert svc.download_document("user-abc/file.pdf") == b"local bytes"
        assert client.calls == []  # no HTTP

    def test_download_local_uri_delegates_local(self, local):
        svc, client = _tiered(local)
        assert svc.download_document("local://user-abc/file.pdf") == b"local bytes"
        assert client.calls == []

    def test_upload_always_delegates_local(self, local, tmp_path):
        svc, client = _tiered(local)
        path = svc.upload_document("user-x", "new.pdf", b"data", "application/pdf")
        # Bare-path return (local-first; writes don't go to satellites).
        assert path.startswith("user-x/")
        assert client.calls == []


class TestSatelliteDispatch:
    def test_download_satellite_uri_makes_get(self, local):
        svc, client = _tiered(local, response=_FakeResponse(200, b"remote bytes"))
        result = svc.download_document(f"satellite://node-a/{_DIGEST}")
        assert result == b"remote bytes"
        method, url, headers = client.calls[0]
        assert method == "GET"
        assert url == f"http://100.64.0.9:8090/v1/storage/blobs/{_DIGEST}"
        assert headers == {"Tailscale-User-Login": "operator@example.com"}

    def test_download_image_satellite(self, local):
        svc, client = _tiered(local, response=_FakeResponse(200, b"img"))
        assert svc.download_image(f"satellite://node-a/{_DIGEST}") == b"img"
        assert client.calls[0][0] == "GET"

    def test_satellite_404_raises_not_found(self, local):
        svc, _ = _tiered(local, response=_FakeResponse(404))
        with pytest.raises(BlobNotFoundError):
            svc.download_document(f"satellite://node-a/{_DIGEST}")

    def test_satellite_401_raises_auth(self, local):
        svc, _ = _tiered(local, response=_FakeResponse(401))
        with pytest.raises(BlobAuthError):
            svc.download_document(f"satellite://node-a/{_DIGEST}")

    def test_satellite_500_raises_unreachable(self, local):
        svc, _ = _tiered(local, response=_FakeResponse(503))
        with pytest.raises(BlobUnreachableError):
            svc.download_document(f"satellite://node-a/{_DIGEST}")

    def test_unknown_node_raises_unreachable(self, local):
        svc, _ = _tiered(local, response=_FakeResponse(200), host_map={})
        with pytest.raises(BlobUnreachableError, match="unknown satellite node"):
            svc.download_document(f"satellite://ghost-node/{_DIGEST}")

    def test_transport_error_raises_unreachable(self, local):
        svc, _ = _tiered(local, response=httpx.ConnectError("refused"))
        with pytest.raises(BlobUnreachableError):
            svc.download_document(f"satellite://node-a/{_DIGEST}")

    def test_delete_satellite_uri_makes_delete(self, local):
        svc, client = _tiered(local, response=_FakeResponse(204))
        svc.delete_document(f"satellite://node-a/{_DIGEST}")
        assert client.calls[0][0] == "DELETE"

    def test_delete_satellite_404_is_silent(self, local):
        # 404 on delete is idempotent-OK — must not raise.
        svc, _ = _tiered(local, response=_FakeResponse(404))
        svc.delete_document(f"satellite://node-a/{_DIGEST}")  # no exception

    def test_delete_local_uri_delegates(self, local, tmp_path):
        svc, client = _tiered(local)
        target = tmp_path / "documents" / "user-abc" / "file.pdf"
        assert target.exists()
        svc.delete_document("local://user-abc/file.pdf")
        assert not target.exists()
        assert client.calls == []


class TestFactory:
    def test_factory_returns_local_when_no_satellites(self, tmp_path, monkeypatch):
        from app.services import storage
        storage.reset_storage_service()
        monkeypatch.setattr(storage, "_has_registered_satellites", lambda: False)

        class _FakeSettings:
            data_dir = str(tmp_path)
        monkeypatch.setattr("app.config.get_settings", lambda: _FakeSettings())

        svc = storage.get_storage_service()
        assert isinstance(svc, LocalStorageService)
        storage.reset_storage_service()

    def test_factory_returns_tiered_when_satellites_and_identity(self, tmp_path, monkeypatch):
        from app.services import storage
        storage.reset_storage_service()
        monkeypatch.setattr(storage, "_has_registered_satellites", lambda: True)
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity",
            lambda: "op@example.com",
        )

        class _FakeSettings:
            data_dir = str(tmp_path)
        monkeypatch.setattr("app.config.get_settings", lambda: _FakeSettings())

        svc = storage.get_storage_service()
        assert isinstance(svc, TieredStorageService)
        storage.reset_storage_service()

    def test_factory_falls_back_to_local_without_identity(self, tmp_path, monkeypatch):
        from app.services import storage
        storage.reset_storage_service()
        monkeypatch.setattr(storage, "_has_registered_satellites", lambda: True)
        monkeypatch.setattr(
            "app.services.tailscale_outbound.get_operator_identity",
            lambda: None,
        )

        class _FakeSettings:
            data_dir = str(tmp_path)
        monkeypatch.setattr("app.config.get_settings", lambda: _FakeSettings())

        svc = storage.get_storage_service()
        assert isinstance(svc, LocalStorageService)
        storage.reset_storage_service()
