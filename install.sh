#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Molebie AI — Bootstrap Installer
# ══════════════════════════════════════════════════════════════
# Ensures Python 3.10+ is available, creates a virtual environment,
# installs the molebie-ai CLI, then launches the interactive wizard.
#
# Usage:  ./install.sh           (interactive)
#         ./install.sh --quick   (auto-select defaults)
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Molebie AI — Installer${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 1: Find Python 3.10+
# ──────────────────────────────────────────────────────────────
info "Looking for Python 3.10+..."

PYTHON_CMD=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_MINOR=$("$candidate" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$PY_MINOR" -ge 10 ] 2>/dev/null; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    warn "Python 3.10+ not found."
    if command -v brew &>/dev/null; then
        echo ""
        read -rp "Install Python via Homebrew? [Y/n] " INSTALL_PY
        INSTALL_PY="${INSTALL_PY:-Y}"
        if [[ "$INSTALL_PY" =~ ^[Yy] ]]; then
            brew install python@3.12
            PYTHON_CMD="python3.12"
            if ! command -v "$PYTHON_CMD" &>/dev/null; then
                PYTHON_CMD="python3"
            fi
        else
            fail "Python 3.10+ is required. Install it and re-run ./install.sh"
        fi
    elif command -v apt-get &>/dev/null; then
        echo ""
        read -rp "Install Python via apt? [Y/n] " INSTALL_PY
        INSTALL_PY="${INSTALL_PY:-Y}"
        if [[ "$INSTALL_PY" =~ ^[Yy] ]]; then
            sudo apt-get update -qq && sudo apt-get install -y python3 python3-venv python3-pip
            PYTHON_CMD="python3"
        else
            fail "Python 3.10+ is required. Install it and re-run ./install.sh"
        fi
    else
        fail "Python 3.10+ is required. Install it and re-run ./install.sh"
    fi
fi

PY_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VERSION ($PYTHON_CMD)"

# ──────────────────────────────────────────────────────────────
# Step 2: Create virtual environment
# ──────────────────────────────────────────────────────────────
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
    ok "Virtual environment exists (.venv/)"
else
    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv .venv
    ok "Virtual environment created (.venv/)"
fi

# Upgrade pip quietly
.venv/bin/python -m pip install --upgrade pip --quiet 2>/dev/null

# ──────────────────────────────────────────────────────────────
# Step 3: Install CLI
# ──────────────────────────────────────────────────────────────
info "Installing molebie-ai CLI..."
.venv/bin/pip install -e . --quiet 2>/dev/null

if .venv/bin/molebie-ai --version &>/dev/null; then
    ok "CLI installed"
else
    fail "CLI installation failed. Check Python and try again."
fi

# ──────────────────────────────────────────────────────────────
# Step 4: Create bin/molebie-ai wrapper
# ──────────────────────────────────────────────────────────────
mkdir -p bin
cat > bin/molebie-ai << 'WRAPPER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$SCRIPT_DIR/.venv/bin/molebie-ai" "$@"
WRAPPER
chmod +x bin/molebie-ai
ok "Created bin/molebie-ai wrapper"

# ──────────────────────────────────────────────────────────────
# Step 5: Launch the interactive wizard
# ──────────────────────────────────────────────────────────────
echo ""
info "Launching setup wizard..."
echo ""

exec .venv/bin/molebie-ai install "$@"
