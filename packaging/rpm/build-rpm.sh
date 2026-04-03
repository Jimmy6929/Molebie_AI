#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Build molebie-ai .rpm package
# ══════════════════════════════════════════════════════════════
# Requires: the PyInstaller binary at dist/molebie-ai (run build.sh first)
# Output:   dist/molebie-ai-0.1.0-1.x86_64.rpm
#
# Usage: bash packaging/rpm/build-rpm.sh
# Note:  Requires rpmbuild (install: dnf install rpm-build)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="0.1.0"

BINARY="$PROJECT_ROOT/dist/molebie-ai"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run 'bash packaging/pyinstaller/build.sh' first."
    exit 1
fi

if ! command -v rpmbuild &>/dev/null; then
    echo "ERROR: rpmbuild not found. Install with: dnf install rpm-build"
    exit 1
fi

echo "=== Building .rpm package ==="

# Set up rpmbuild directory structure
RPM_ROOT="$PROJECT_ROOT/build/rpm"
rm -rf "$RPM_ROOT"
mkdir -p "$RPM_ROOT"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Copy binary as source
cp "$BINARY" "$RPM_ROOT/SOURCES/molebie-ai"

# Copy spec
cp "$SCRIPT_DIR/molebie-ai.spec" "$RPM_ROOT/SPECS/"

# Build
rpmbuild --define "_topdir $RPM_ROOT" -bb "$RPM_ROOT/SPECS/molebie-ai.spec"

# Copy output to dist/
ARCH=$(uname -m)
RPM_FILE=$(find "$RPM_ROOT/RPMS" -name "*.rpm" | head -1)
if [ -n "$RPM_FILE" ]; then
    OUTPUT="$PROJECT_ROOT/dist/molebie-ai-${VERSION}-1.${ARCH}.rpm"
    cp "$RPM_FILE" "$OUTPUT"
    echo ""
    echo "=== .rpm package built ==="
    echo "Output: $OUTPUT"
    echo "Install: sudo dnf install $OUTPUT"
else
    echo "ERROR: No .rpm file produced"
    exit 1
fi
