#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Build molebie-ai standalone binary using PyInstaller
# ══════════════════════════════════════════════════════════════
# Output: dist/molebie-ai (or dist/molebie-ai.exe on Windows)
#
# Usage:
#   bash packaging/pyinstaller/build.sh
#
# On Windows, run from Git Bash or WSL:
#   bash packaging/pyinstaller/build.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Building molebie-ai standalone binary ==="

# Ensure venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Determine venv python path
if [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PY=".venv/Scripts/python.exe"
else
    VENV_PY=".venv/bin/python"
fi

# Install project + PyInstaller
echo "Installing dependencies..."
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -e . --quiet
"$VENV_PY" -m pip install pyinstaller --quiet

# Build
echo "Running PyInstaller..."
"$VENV_PY" -m PyInstaller packaging/pyinstaller/molebie-ai.spec \
    --distpath dist \
    --workpath build/pyinstaller \
    --noconfirm

# Verify
BINARY="dist/molebie-ai"
if [ -f "$BINARY.exe" ]; then
    BINARY="$BINARY.exe"
fi

if [ -f "$BINARY" ]; then
    SIZE=$(du -sh "$BINARY" | cut -f1)
    echo ""
    echo "=== Build successful ==="
    echo "Binary: $BINARY ($SIZE)"
    "$BINARY" --version
else
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi
