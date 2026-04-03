#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Molebie AI — Bootstrap Installer
# ══════════════════════════════════════════════════════════════
# Ensures Python 3.10+ is available, creates a virtual environment,
# installs the molebie-ai CLI, then launches the interactive wizard.
#
# Usage:
#   Local:   ./install.sh                          (interactive)
#            ./install.sh --quick                   (auto-select defaults)
#   Remote:  curl -fsSL https://molebieai.com/install.sh | bash
#            curl -fsSL https://raw.githubusercontent.com/Jimmy6929/Molebie_AI/main/install.sh | bash
#            curl -fsSL https://molebieai.com/install.sh | bash -s -- --quick
#            curl -fsSL https://molebieai.com/install.sh | bash -s -- --install-dir ~/my-molebie
# ══════════════════════════════════════════════════════════════

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

# ──────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────
INSTALL_DIR=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore remaining args for the wizard
set -- "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"

# ──────────────────────────────────────────────────────────────
# Detect execution mode: local (inside repo) vs remote (curl pipe)
# ──────────────────────────────────────────────────────────────
REMOTE_MODE=0

if [[ -z "${BASH_SOURCE[0]:-}" ]] \
   || [[ "${BASH_SOURCE[0]}" == "/dev/stdin" ]] \
   || [[ "${BASH_SOURCE[0]}" == "bash" ]]; then
    REMOTE_MODE=1
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ ! -f "$SCRIPT_DIR/pyproject.toml" ]] || [[ ! -d "$SCRIPT_DIR/gateway" ]]; then
        REMOTE_MODE=1
    fi
fi

# ──────────────────────────────────────────────────────────────
# Remote mode: clone the repo first, then re-exec local script
# ──────────────────────────────────────────────────────────────
if [[ "$REMOTE_MODE" -eq 1 ]]; then
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Molebie AI — Remote Installer${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo ""

    # Check git
    if ! command -v git &>/dev/null; then
        fail "git is required for remote install. Install it first:\n  macOS:  xcode-select --install\n  Linux:  sudo apt-get install git"
    fi

    # Determine install directory
    if [[ -z "$INSTALL_DIR" ]]; then
        INSTALL_DIR="$HOME/Molebie_AI"
    fi
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

    if [[ -d "$INSTALL_DIR" ]]; then
        if [[ -f "$INSTALL_DIR/pyproject.toml" ]] && [[ -d "$INSTALL_DIR/gateway" ]]; then
            info "Existing installation found at $INSTALL_DIR"
            info "Updating with git pull..."
            git -C "$INSTALL_DIR" pull --ff-only || warn "git pull failed — continuing with existing code"
            ok "Repository updated"
        else
            fail "$INSTALL_DIR already exists but is not a Molebie AI installation.\n  Use --install-dir to specify a different path."
        fi
    else
        info "Cloning into $INSTALL_DIR..."
        git clone --depth 1 https://github.com/Jimmy6929/Molebie_AI.git "$INSTALL_DIR"
        ok "Repository cloned"
    fi

    # Re-exec the local install.sh (clean terminal context, no pipe on stdin)
    info "Handing off to local installer..."
    exec "$INSTALL_DIR/install.sh" --quick "$@"
fi

# ──────────────────────────────────────────────────────────────
# Local mode: we are inside the repo
# ──────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

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
            brew install python@3.12 </dev/null
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
            sudo apt-get update -qq </dev/null && sudo apt-get install -y python3 python3-venv python3-pip </dev/null
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
.venv/bin/python -m pip install --upgrade pip --quiet </dev/null 2>/dev/null

# ──────────────────────────────────────────────────────────────
# Step 2b: Validate Python compatibility
# ──────────────────────────────────────────────────────────────
# Probe: install the PINNED pydantic version from requirements.txt using
# only pre-built wheels (--only-binary :all:).  pydantic depends on
# pydantic-core (compiled Rust), so if no wheel exists for this Python
# version, pip fails instantly.  This auto-detects incompatible Python
# without hardcoded version caps.
info "Checking package compatibility..."
PYDANTIC_PIN=$(grep -E '^pydantic==' gateway/requirements.txt 2>/dev/null | head -1)
PROBE_PKG="${PYDANTIC_PIN:-pydantic}"
if .venv/bin/pip install "$PROBE_PKG" --only-binary :all: --force-reinstall --quiet </dev/null 2>/dev/null; then
    ok "Package compatibility verified"
else
    warn "Python $PY_VERSION has no pre-built packages for key dependencies."
    warn "This usually means the Python version is too new for the package ecosystem."
    info "Looking for a compatible Python..."

    FALLBACK_CMD=""
    for fb in python3.13 python3.12 python3.11 python3.10; do
        if command -v "$fb" &>/dev/null; then
            FB_MINOR=$("$fb" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
            if [ "$FB_MINOR" -ge 10 ] 2>/dev/null; then
                FALLBACK_CMD="$fb"
                break
            fi
        fi
    done

    if [ -z "$FALLBACK_CMD" ]; then
        # No compatible Python installed — try to install one automatically.
        # Strategy: Homebrew → python.org .pkg (macOS) → apt/dnf/pacman (Linux)

        # Attempt 1: Homebrew (macOS & Linux)
        if [ -z "$FALLBACK_CMD" ] && command -v brew &>/dev/null; then
            info "Installing Python 3.12 via Homebrew (this may take several minutes)..."
            if brew install python@3.12 </dev/null; then
                FALLBACK_CMD="$(brew --prefix)/bin/python3.12"
                if ! command -v "$FALLBACK_CMD" &>/dev/null; then
                    FALLBACK_CMD="python3.12"
                fi
                command -v "$FALLBACK_CMD" &>/dev/null || FALLBACK_CMD=""
            else
                warn "Homebrew install failed — trying next method..."
            fi
        fi

        # Attempt 2: python.org official installer (macOS only)
        if [ -z "$FALLBACK_CMD" ] && [[ "$OSTYPE" == darwin* ]]; then
            info "Downloading Python 3.12 from python.org..."
            PY_PKG="/tmp/molebie-python-3.12.pkg"
            if curl -fsSL "https://www.python.org/ftp/python/3.12.13/python-3.12.13-macos11.pkg" -o "$PY_PKG" </dev/null 2>/dev/null; then
                info "Installing Python 3.12 (may ask for your password)..."
                if sudo installer -pkg "$PY_PKG" -target / </dev/null 2>/dev/null; then
                    # python.org installer puts python3.12 in /usr/local/bin (Intel)
                    # or /Library/Frameworks/Python.framework/Versions/3.12/bin (universal)
                    for pypath in /usr/local/bin/python3.12 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12; do
                        if [ -x "$pypath" ]; then
                            FALLBACK_CMD="$pypath"
                            break
                        fi
                    done
                    if [ -z "$FALLBACK_CMD" ] && command -v python3.12 &>/dev/null; then
                        FALLBACK_CMD="python3.12"
                    fi
                    [ -n "$FALLBACK_CMD" ] && ok "Python 3.12 installed from python.org"
                else
                    warn "python.org installer failed — trying next method..."
                fi
                rm -f "$PY_PKG"
            else
                warn "Download failed — trying next method..."
            fi
        fi

        # Attempt 3: System package manager (Linux)
        if [ -z "$FALLBACK_CMD" ] && [[ "$OSTYPE" == linux* ]]; then
            if command -v apt-get &>/dev/null; then
                info "Installing Python 3.12 via apt..."
                if sudo apt-get update -qq </dev/null && sudo apt-get install -y python3.12 python3.12-venv </dev/null 2>/dev/null; then
                    FALLBACK_CMD="python3.12"
                fi
            elif command -v dnf &>/dev/null; then
                info "Installing Python 3.12 via dnf..."
                if sudo dnf install -y python3.12 </dev/null 2>/dev/null; then
                    FALLBACK_CMD="python3.12"
                fi
            elif command -v pacman &>/dev/null; then
                info "Installing Python 3.12 via pacman..."
                if sudo pacman -S --noconfirm python </dev/null 2>/dev/null; then
                    FALLBACK_CMD="python3"
                fi
            fi
        fi

        # All automatic methods exhausted
        if [ -z "$FALLBACK_CMD" ]; then
            echo ""
            fail "Could not install a compatible Python automatically.\n\n  Install Python 3.12 manually:\n    macOS:  https://www.python.org/downloads/release/python-31213/\n    Linux:  https://www.python.org/downloads/\n\n  Then re-run this installer."
        fi
    fi

    PYTHON_CMD="$FALLBACK_CMD"
    PY_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Switching to Python $PY_VERSION ($PYTHON_CMD)..."
    rm -rf .venv
    "$PYTHON_CMD" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip --quiet </dev/null 2>/dev/null

    # Re-verify the fallback Python works
    if .venv/bin/pip install "$PROBE_PKG" --only-binary :all: --force-reinstall --quiet </dev/null 2>/dev/null; then
        ok "Python $PY_VERSION — package compatibility verified"
    else
        fail "Python $PY_VERSION also failed the compatibility check.\n  Install Python 3.12 from https://www.python.org/downloads/ and re-run."
    fi
fi

# ──────────────────────────────────────────────────────────────
# Step 3: Install CLI
# ──────────────────────────────────────────────────────────────
info "Installing molebie-ai CLI..."
.venv/bin/pip install -e . --quiet </dev/null 2>/dev/null

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
