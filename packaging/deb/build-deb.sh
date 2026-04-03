#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Build molebie-ai .deb package
# ══════════════════════════════════════════════════════════════
# Requires: the PyInstaller binary at dist/molebie-ai (run build.sh first)
# Output:   dist/molebie-ai_0.1.0_amd64.deb
#
# Usage: bash packaging/deb/build-deb.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="0.1.0"
ARCH="amd64"
PKG_NAME="molebie-ai_${VERSION}_${ARCH}"

BINARY="$PROJECT_ROOT/dist/molebie-ai"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run 'bash packaging/pyinstaller/build.sh' first."
    exit 1
fi

echo "=== Building .deb package ==="

# Create package directory structure
BUILD_DIR="$PROJECT_ROOT/build/deb/$PKG_NAME"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/local/bin"

# Copy control files
cp "$SCRIPT_DIR/control" "$BUILD_DIR/DEBIAN/control"
cp "$SCRIPT_DIR/postinst" "$BUILD_DIR/DEBIAN/postinst"
chmod 0755 "$BUILD_DIR/DEBIAN/postinst"

# Update version and architecture in control file
sed -i "s/^Version:.*/Version: $VERSION/" "$BUILD_DIR/DEBIAN/control" 2>/dev/null || \
    sed -i '' "s/^Version:.*/Version: $VERSION/" "$BUILD_DIR/DEBIAN/control"

# Detect architecture
MACHINE=$(uname -m)
if [ "$MACHINE" = "aarch64" ] || [ "$MACHINE" = "arm64" ]; then
    ARCH="arm64"
    sed -i "s/^Architecture:.*/Architecture: arm64/" "$BUILD_DIR/DEBIAN/control" 2>/dev/null || \
        sed -i '' "s/^Architecture:.*/Architecture: arm64/" "$BUILD_DIR/DEBIAN/control"
fi

# Copy binary
cp "$BINARY" "$BUILD_DIR/usr/local/bin/molebie-ai"
chmod 0755 "$BUILD_DIR/usr/local/bin/molebie-ai"

# Calculate installed size (in KB)
INSTALLED_SIZE=$(du -sk "$BUILD_DIR/usr" | cut -f1)
echo "Installed-Size: $INSTALLED_SIZE" >> "$BUILD_DIR/DEBIAN/control"

# Build .deb
OUTPUT="$PROJECT_ROOT/dist/${PKG_NAME}.deb"
dpkg-deb --build "$BUILD_DIR" "$OUTPUT"

echo ""
echo "=== .deb package built ==="
echo "Output: $OUTPUT"
echo "Install: sudo dpkg -i $OUTPUT"
