"""Tests for the v3 to v4 schema migration.

v4 adds the fleet model: a `satellites: list[SatelliteNode]` field on
MolebieConfig. The migration must:
  - Default new and single-machine v3 configs to an empty satellites list.
  - Synthesize one COMPUTE satellite from v3 distributed configs that pointed
    at a remote LLM machine (run_inference=False, gateway+webapp local,
    inference_host set to a non-localhost value).
  - Leave already-v4 configs untouched.
  - Round-trip cleanly through load_config() + save_config().
"""

from __future__ import annotations

import json

from cli.models.config import MolebieConfig, SatelliteRole, SatelliteStatus
from cli.services import config_manager
from cli.services.config_manager import _migrate_v3_to_v4


def _base_v3() -> dict:
    return {
        "version": 3,
        "setup_type": "single",
        "run_inference": True,
        "run_gateway": True,
        "run_webapp": True,
        "inference_host": "localhost",
        "gateway_host": "localhost",
        "webapp_host": "localhost",
        "inference_backend": "mlx",
        "model_profile": "balanced",
        "voice_enabled": False,
        "search_enabled": True,
        "rag_enabled": True,
        "installed": True,
    }


def test_v3_single_machine_migrates_to_empty_satellites():
    data = _base_v3()
    out = _migrate_v3_to_v4(dict(data))
    assert out["version"] == 4
    assert out["satellites"] == []


def test_v3_distributed_with_remote_llm_creates_compute_satellite():
    data = _base_v3()
    data["setup_type"] = "distributed"
    data["run_inference"] = False
    data["inference_host"] = "100.64.0.5"

    out = _migrate_v3_to_v4(dict(data))

    assert out["version"] == 4
    assert len(out["satellites"]) == 1
    sat = out["satellites"][0]
    assert sat["host"] == "100.64.0.5"
    assert sat["role"] == "compute"
    assert sat["status"] == "active"
    assert sat["label"] == "migrated-from-v3"
    # joined_at should be an ISO 8601 timestamp the model can parse
    assert "T" in sat["joined_at"]


def test_v3_distributed_with_localhost_inference_host_stays_empty():
    """A misconfigured-but-loadable v3 distributed setup pointing at localhost
    shouldn't become a fake satellite — there's no real second machine."""
    data = _base_v3()
    data["setup_type"] = "distributed"
    data["run_inference"] = False
    data["inference_host"] = "localhost"

    out = _migrate_v3_to_v4(dict(data))

    assert out["version"] == 4
    assert out["satellites"] == []


def test_v3_llm_server_only_stays_empty_satellites():
    """If THIS machine was the LLM server in v3 (gateway/webapp remote),
    the v4 mapping is unclear — defer to the user re-running install."""
    data = _base_v3()
    data["setup_type"] = "distributed"
    data["run_inference"] = True
    data["run_gateway"] = False
    data["run_webapp"] = False
    data["gateway_host"] = "100.64.0.6"
    data["webapp_host"] = "100.64.0.6"

    out = _migrate_v3_to_v4(dict(data))

    assert out["version"] == 4
    assert out["satellites"] == []


def test_v4_config_is_not_remigrated():
    """Already-v4 data must pass through untouched."""
    data = {"version": 4, "setup_type": "single", "satellites": []}
    out = _migrate_v3_to_v4(dict(data))
    assert out == data


def test_v4_with_existing_satellite_is_not_overwritten():
    """If a v4 config already has satellites, migration must not clobber them."""
    sat = {
        "host": "100.64.0.7",
        "role": "storage",
        "capabilities": {"disk_gb": 500},
        "status": "active",
        "joined_at": "2026-05-20T14:00:00+00:00",
        "label": "nas-box",
    }
    data = {"version": 4, "setup_type": "single", "satellites": [sat]}
    out = _migrate_v3_to_v4(dict(data))
    assert out["satellites"] == [sat]


def test_migrated_satellite_validates_as_pydantic_model():
    """The synthesized satellite dict must be valid for MolebieConfig.model_validate."""
    data = _base_v3()
    data["setup_type"] = "distributed"
    data["run_inference"] = False
    data["inference_host"] = "100.64.0.5"

    out = _migrate_v3_to_v4(dict(data))
    config = MolebieConfig.model_validate(out)

    assert config.version == 4
    assert len(config.satellites) == 1
    assert config.satellites[0].role == SatelliteRole.COMPUTE
    assert config.satellites[0].status == SatelliteStatus.ACTIVE


def test_load_config_persists_migration_to_disk(tmp_path, monkeypatch):
    """load_config() should run the chain and save the upgraded config back."""
    cfg_dir = tmp_path / ".molebie"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.json"

    v3_on_disk = _base_v3()
    v3_on_disk["setup_type"] = "distributed"
    v3_on_disk["run_inference"] = False
    v3_on_disk["inference_host"] = "100.64.0.5"
    cfg_path.write_text(json.dumps(v3_on_disk))

    monkeypatch.setattr(config_manager, "get_config_path", lambda: cfg_path)
    monkeypatch.setattr(config_manager, "get_config_dir", lambda: cfg_dir)

    config = config_manager.load_config()

    assert config.version == 4
    assert len(config.satellites) == 1
    # Confirm it was persisted, not just held in memory
    persisted = json.loads(cfg_path.read_text())
    assert persisted["version"] == 4
    assert len(persisted["satellites"]) == 1


def test_load_config_idempotent_for_v4_on_disk(tmp_path, monkeypatch):
    """A v4 config on disk should load without rewriting the file."""
    cfg_dir = tmp_path / ".molebie"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.json"

    v4_on_disk = MolebieConfig().model_dump()
    cfg_path.write_text(json.dumps(v4_on_disk))
    mtime_before = cfg_path.stat().st_mtime_ns

    monkeypatch.setattr(config_manager, "get_config_path", lambda: cfg_path)
    monkeypatch.setattr(config_manager, "get_config_dir", lambda: cfg_dir)

    config = config_manager.load_config()
    mtime_after = cfg_path.stat().st_mtime_ns

    assert config.version == 4
    assert mtime_before == mtime_after, "v4 config should not be rewritten on load"
