# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for molebie-ai CLI standalone binary.

Produces a single-file executable that bundles the entire CLI package
plus all Python dependencies. No Python installation required on target.

Build:  pyinstaller packaging/pyinstaller/molebie-ai.spec
Output: dist/molebie-ai (or dist/molebie-ai.exe on Windows)
"""

import sys
from pathlib import Path

block_cipher = None

# Project root is two levels up from this spec file
PROJECT_ROOT = str(Path(SPECPATH).parent.parent)

a = Analysis(
    [str(Path(PROJECT_ROOT) / "cli" / "main.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # Include .env.example so the install wizard can generate .env.local
        (str(Path(PROJECT_ROOT) / ".env.example"), "."),
    ],
    hiddenimports=[
        # Typer / Click ecosystem — PyInstaller misses these
        "typer",
        "typer.main",
        "typer.core",
        "click",
        "click.core",
        "click.decorators",
        # Rich (used by typer[all])
        "rich",
        "rich.console",
        "rich.panel",
        "rich.table",
        "rich.progress",
        "rich.markup",
        # Pydantic v2
        "pydantic",
        "pydantic.main",
        "pydantic._internal",
        "pydantic._internal._core_utils",
        # httpx and its transports
        "httpx",
        "httpcore",
        "h11",
        "anyio",
        "anyio._backends",
        "anyio._backends._asyncio",
        "sniffio",
        # python-dotenv
        "dotenv",
        # CLI submodules — explicit to guarantee inclusion
        "cli",
        "cli.__init__",
        "cli.main",
        "cli.commands",
        "cli.commands.install",
        "cli.commands.run",
        "cli.commands.doctor",
        "cli.commands.status",
        "cli.commands.config_cmd",
        "cli.commands.feature",
        "cli.commands.model_cmd",
        "cli.models",
        "cli.models.config",
        "cli.services",
        "cli.services.backend_setup",
        "cli.services.config_manager",
        "cli.services.env_generator",
        "cli.services.feature_setup",
        "cli.services.prerequisite_checker",
        "cli.services.service_manager",
        "cli.services.system_info",
        "cli.services.deep_checker",
        "cli.ui",
        "cli.ui.console",
        "cli.ui.prompts",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages the CLI doesn't need at runtime
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "PIL",
        "cv2",
        "tkinter",
        "test",
        "unittest",
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="molebie-ai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=sys.platform != "win32",  # Strip symbols on macOS/Linux, not Windows
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Build for current architecture
    codesign_identity=None,
    entitlements_file=None,
)
