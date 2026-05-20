"""Tests for the collapsed install wizard step 2 (_ask_setup_type).

Before this change, step 2 forked into four topology choices. After: a single
Y/n confirmation. The user can't intentionally produce a DISTRIBUTED config from
the install wizard anymore — the only way to get one is migrating from v3.
"""

from __future__ import annotations

import inspect

import pytest
import typer

from cli.commands import install as install_module
from cli.commands.install import _ask_setup_type
from cli.models.config import MolebieConfig, SetupType


def test_confirm_yes_produces_single_node_primary(monkeypatch):
    """Y to the confirmation should set up a SINGLE primary with all services local."""
    monkeypatch.setattr(install_module, "ask_confirm", lambda *a, **kw: True)

    config = MolebieConfig()
    _ask_setup_type(config)

    assert config.setup_type == SetupType.SINGLE
    assert config.run_inference is True
    assert config.run_gateway is True
    assert config.run_webapp is True


def test_confirm_no_aborts_install_cleanly(monkeypatch):
    """N to the confirmation should raise typer.Exit(0) — a clean abort."""
    monkeypatch.setattr(install_module, "ask_confirm", lambda *a, **kw: False)

    config = MolebieConfig()
    with pytest.raises(typer.Exit) as excinfo:
        _ask_setup_type(config)
    assert excinfo.value.exit_code == 0


def test_no_topology_choice_prompt_in_source():
    """The old 4-topology ask_choice prompt must not be reachable from _ask_setup_type."""
    src = inspect.getsource(_ask_setup_type)
    assert "ask_choice" not in src, (
        "Step 2 must no longer ask the 4-topology question — that's the whole point of the collapse."
    )
    assert "All-in-one" not in src, "The old topology labels should be gone."
    assert "LLM server" not in src, "The old topology labels should be gone."


def test_dead_helper_removed():
    """_show_other_machine_instructions was only called from the old topology fork.
    It must be deleted along with the fork to avoid dead code."""
    assert not hasattr(install_module, "_show_other_machine_instructions"), (
        "_show_other_machine_instructions is dead code after the wizard collapse and "
        "must be removed."
    )


def test_distributed_setup_still_supported_via_migration():
    """The collapse removes the wizard's distributed path, but the MolebieConfig
    model itself must still support DISTRIBUTED — v3 distributed configs migrate
    to v4 keeping setup_type='distributed'."""
    config = MolebieConfig(setup_type=SetupType.DISTRIBUTED, run_inference=False)
    assert config.setup_type == SetupType.DISTRIBUTED


def test_aborted_install_does_not_mutate_config(monkeypatch):
    """If the user says N, the partially-mutated config should not be left
    in an inconsistent state."""
    monkeypatch.setattr(install_module, "ask_confirm", lambda *a, **kw: False)

    config = MolebieConfig()
    original_setup = config.setup_type
    with pytest.raises(typer.Exit):
        _ask_setup_type(config)

    assert config.setup_type == original_setup
