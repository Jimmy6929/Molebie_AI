#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Build molebie-ai AppImage
# ══════════════════════════════════════════════════════════════
# Requires: the PyInstaller binary at dist/molebie-ai (run build.sh first)
# Output:   dist/Molebie_AI-x86_64.AppImage
#
# Usage: bash packaging/appimage/build-appimage.sh
# Note:  Linux only. Downloads appimagetool automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BINARY="$PROJECT_ROOT/dist/molebie-ai"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run 'bash packaging/pyinstaller/build.sh' first."
    exit 1
fi

echo "=== Building AppImage ==="

# Determine architecture
ARCH=$(uname -m)

# Create AppDir structure
APPDIR="$PROJECT_ROOT/build/appimage/Molebie_AI.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"

# Copy files
cp "$SCRIPT_DIR/AppRun" "$APPDIR/"
chmod +x "$APPDIR/AppRun"

cp "$SCRIPT_DIR/molebie-ai.desktop" "$APPDIR/molebie-ai.desktop"
cp "$BINARY" "$APPDIR/usr/bin/molebie-ai"
chmod +x "$APPDIR/usr/bin/molebie-ai"

# Create a simple icon (placeholder SVG)
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cat > "$APPDIR/molebie-ai.svg" << 'ICON'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect width="256" height="256" rx="32" fill="#2563eb"/>
  <text x="128" y="160" font-family="Arial,sans-serif" font-size="120" font-weight="bold" fill="white" text-anchor="middle">M</text>
</svg>
ICON
cp "$APPDIR/molebie-ai.svg" "$APPDIR/usr/share/icons/hicolor/256x256/apps/"

# Download appimagetool if not present
TOOL="$PROJECT_ROOT/build/appimage/appimagetool"
if [ ! -f "$TOOL" ]; then
    echo "Downloading appimagetool..."
    TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
    curl -fsSL "$TOOL_URL" -o "$TOOL"
    chmod +x "$TOOL"
fi

# Build AppImage
export ARCH
OUTPUT="$PROJECT_ROOT/dist/Molebie_AI-${ARCH}.AppImage"
"$TOOL" "$APPDIR" "$OUTPUT" 2>/dev/null || \
    "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUTPUT"

echo ""
echo "=== AppImage built ==="
echo "Output: $OUTPUT"
echo "Run: chmod +x $OUTPUT && ./$OUTPUT --version"
