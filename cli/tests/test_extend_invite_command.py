"""Tests for ``molebie-ai extend invite`` — the primary-side one-liner generator.

Mirrors the mocking style of test_extend_commands.py (monkeypatched
get_tailscale_ip + capture stdout via capsys). The invite command is
output-only — no HTTP, no DB — so the tests check that the printed
copyable string contains the right parts (primary IP, role, label,
``pipx install``, install command).
"""

from __future__ import annotations

import pytest
import typer

from cli.commands import extend as extend_module


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Default: Tailscale IP resolves to a known value. Tests that need
    to override can do so per-case."""
    monkeypatch.setattr(extend_module, "get_tailscale_ip", lambda: "100.104.193.59")
    yield


def _flatten(text: str) -> str:
    """rich wraps long lines at the captured terminal width; collapse all
    whitespace runs to single spaces so most substring asserts survive."""
    return " ".join(text.split())


def _squashed(text: str) -> str:
    """Strip ALL whitespace — for substrings rich may hard-wrap mid-word
    (e.g. ``satellite_storage`` becomes ``satellite_sto\\nrage`` in the
    captured output)."""
    return "".join(text.split())


class TestExtendInvite:
    def test_prints_pipx_install_with_primary_ip_and_role(self, capsys):
        extend_module.invite_satellite(role="storage", label=None, repo_ref="main")
        raw = capsys.readouterr().out
        flat = _flatten(raw)
        squashed = _squashed(raw)
        # Short strings that won't wrap survive whitespace-collapse.
        assert "pipx install" in flat
        assert "100.104.193.59" in flat
        assert "--role storage" in flat
        assert "github.com/Jimmy6929/Molebie_AI.git" in flat
        # Long identifiers may hard-wrap mid-word — strip all whitespace.
        assert "subdirectory=satellite_storage" in squashed
        # Requirements footnote names the three things the new satellite needs.
        assert "Tailscale" in flat
        assert "Python" in flat

    def test_label_appears_in_one_liner_when_provided(self, capsys):
        extend_module.invite_satellite(role="storage", label="nas", repo_ref="main")
        out = _flatten(capsys.readouterr().out)
        assert "--label" in out
        assert "nas" in out

    def test_repo_ref_overrides_default_main(self, capsys):
        extend_module.invite_satellite(
            role="storage", label=None, repo_ref="fix/some-branch",
        )
        out = _squashed(capsys.readouterr().out)
        assert "@fix/some-branch" in out
        assert "@main#" not in out

    def test_no_tailscale_ip_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(extend_module, "get_tailscale_ip", lambda: None)
        with pytest.raises(typer.Exit) as exc:
            extend_module.invite_satellite(
                role="storage", label=None, repo_ref="main",
            )
        assert exc.value.exit_code == 1
        out = capsys.readouterr().out
        assert "Tailscale IP" in out or "tailscale" in out.lower()
