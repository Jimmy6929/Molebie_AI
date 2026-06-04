"""Tests for the cross-platform service-install dispatcher."""

from __future__ import annotations

import pytest

from satellite_storage.cli import service


def test_render_template_happy(monkeypatch=None):
    out = service.render_template("hello __NAME__", {"NAME": "world"})
    assert out == "hello world"


def test_render_template_passes_underscores_in_value():
    """A value like ``/Users/__test__`` must not trip the drift check."""
    out = service.render_template("home is __HOME__", {"HOME": "/Users/__test__"})
    assert out == "home is /Users/__test__"


def test_render_template_raises_on_missing_key():
    with pytest.raises(KeyError, match="__MISSING__"):
        service.render_template("hi __MISSING__ bye", {})


def test_dispatcher_routes_to_macos(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "darwin")
    mod = service._platform_module()
    assert mod.__name__.endswith("service_macos")


def test_dispatcher_routes_to_linux(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "linux")
    mod = service._platform_module()
    assert mod.__name__.endswith("service_linux")


def test_dispatcher_routes_to_windows(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "win32")
    mod = service._platform_module()
    assert mod.__name__.endswith("service_windows")


def test_dispatcher_raises_on_unknown_platform(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "haiku")
    with pytest.raises(service.PlatformNotSupportedError):
        service._platform_module()


def test_default_data_dir_under_home():
    p = service.default_data_dir()
    assert ".molebie" in str(p)
    assert "satellite-storage" in str(p)
