"""Shared types + helpers for the satellite service modules.

This is a *leaf* module — it imports nothing from elsewhere in
``satellite_storage.cli``. Both the cross-platform dispatcher in
``service.py`` and the platform implementations (``service_macos``,
``service_linux``, ``service_windows``) import from here, which breaks
the static dependency cycle CodeQL otherwise flags between
``service.py`` and ``service_<platform>.py``.

The public names defined here are re-exported by ``service.py`` so
existing callers continue to work unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SATELLITE_SERVICE_LABEL = "com.molebieai.satellite"


@dataclass(frozen=True)
class ServiceConfig:
    """Everything any platform module needs to install the satellite as a service.

    ``label`` is platform-flavoured (launchd uses dotted reverse-DNS,
    systemd uses ``molebie-satellite``, Windows tasks use ``MolebieSatellite``)
    so each platform module overrides as needed; this default is the
    macOS-style canonical form.
    """

    satellite_bin: Path
    data_dir: Path
    log_dir: Path
    home_dir: Path
    label: str = SATELLITE_SERVICE_LABEL


class ServiceInstallError(Exception):
    """Raised when the platform-specific install/uninstall fails."""


def render_template(template_text: str, mapping: dict[str, str]) -> str:
    """Substitute ``__KEY__`` placeholders. Drift-detection is precise:

    Snapshots the placeholders **in the input** before substitution, then
    verifies every one of them appeared in ``mapping``. This avoids
    false-positives on legitimate values that happen to contain ``__`` —
    e.g., a home directory like ``/Users/__test__`` would otherwise trip a
    naive "any leftover ``__`` in output" check.

    Missing keys raise ``KeyError`` so template/code drift fails loudly.
    """
    expected = set(re.findall(r"__[A-Z_]+__", template_text))
    provided = {f"__{key}__" for key in mapping}
    missing = expected - provided
    if missing:
        raise KeyError(
            f"render_template: template needs {sorted(missing)} but they "
            f"were not supplied in mapping"
        )

    out = template_text
    for key, value in mapping.items():
        out = out.replace(f"__{key}__", value)
    return out
