"""Tests for the per-tier extra-payload override (Layer C of the
backend-agnostic streaming plan).

Operators set ``INFERENCE_<MODE>_EXTRA_PAYLOAD`` to a JSON object that the
gateway merges into the request payload last (last-write-wins). This is
the escape hatch for backend-specific quirks like the mlx_vlm
field-presence streaming-buffer flip — keeping the gateway code free of
``if mlx_vlm: ...`` branches.
"""

import json

import pytest

from app.config import Settings

# ── Parser: get_extra_payload_for_mode ──────────────────────────────────────

@pytest.mark.parametrize("mode", ["instant", "thinking", "thinking_harder"])
def test_extra_payload_unset_returns_empty_dict(mode):
    settings = Settings()
    assert settings.get_extra_payload_for_mode(mode) == {}


@pytest.mark.parametrize("mode,attr", [
    ("instant", "inference_instant_extra_payload"),
    ("thinking", "inference_thinking_extra_payload"),
    ("thinking_harder", "inference_thinking_harder_extra_payload"),
])
def test_extra_payload_well_formed_json_parses(mode, attr):
    raw = json.dumps({"thinking_budget": 0, "thinking_start_token": "<think>"})
    settings = Settings(**{attr: raw})
    parsed = settings.get_extra_payload_for_mode(mode)
    assert parsed == {"thinking_budget": 0, "thinking_start_token": "<think>"}


def test_extra_payload_empty_string_returns_empty_dict():
    settings = Settings(inference_instant_extra_payload="   ")
    assert settings.get_extra_payload_for_mode("instant") == {}


def test_extra_payload_malformed_json_raises_at_parse():
    settings = Settings(inference_instant_extra_payload="{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        settings.get_extra_payload_for_mode("instant")


def test_extra_payload_non_object_raises():
    """JSON list / scalar must be rejected — payload merge requires a dict."""
    settings = Settings(inference_instant_extra_payload='[1, 2, 3]')
    with pytest.raises(ValueError, match="must be a JSON object"):
        settings.get_extra_payload_for_mode("instant")


def test_extra_payload_per_mode_isolation():
    """Setting one tier's extra payload must not leak into another tier."""
    settings = Settings(
        inference_instant_extra_payload='{"thinking_budget":0}',
    )
    assert settings.get_extra_payload_for_mode("instant") == {"thinking_budget": 0}
    assert settings.get_extra_payload_for_mode("thinking") == {}
    assert settings.get_extra_payload_for_mode("thinking_harder") == {}


# ── Eager validation in get_settings ─────────────────────────────────────────

def test_get_settings_validates_extra_payloads_eagerly(monkeypatch):
    """Malformed JSON in .env must fail at startup (first get_settings()
    call), not silently per-request. Invariant: a bad .env crashes the
    process loudly with a useful error message, not on the user's first
    chat turn hours after deploy."""
    monkeypatch.setenv("INFERENCE_INSTANT_EXTRA_PAYLOAD", "{still invalid")
    # Bust the lru_cache so a fresh Settings is constructed.
    from app import config as config_module
    config_module.get_settings.cache_clear()
    with pytest.raises(ValueError, match="INFERENCE_INSTANT_EXTRA_PAYLOAD"):
        config_module.get_settings()
    config_module.get_settings.cache_clear()


def test_get_settings_succeeds_with_well_formed_extras(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_INSTANT_EXTRA_PAYLOAD",
        '{"thinking_budget":0,"thinking_start_token":"<think>"}',
    )
    from app import config as config_module
    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()
    assert settings.get_extra_payload_for_mode("instant")["thinking_budget"] == 0
    config_module.get_settings.cache_clear()
