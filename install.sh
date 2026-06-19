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

    # Re-exec the local install.sh with stdin reconnected to the terminal.
    # This allows the interactive wizard to work even when invoked via curl|bash.
    # Falls back to --quick if no terminal is available (e.g. CI, containers).
    info "Handing off to local installer..."
    if [ -t 0 ] || [ -e /dev/tty ]; then
        exec "$INSTALL_DIR/install.sh" "$@" </dev/tty
    else
        warn "No terminal detected — running in non-interactive mode"
        exec "$INSTALL_DIR/install.sh" --quick "$@"
    fi
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
# Python fitness helpers
# ──────────────────────────────────────────────────────────────
# sqlite-vec is mandatory and loads as a runtime extension, which requires
# conn.enable_load_extension(). That method is ABSENT on Pythons built without
# --enable-loadable-sqlite-extensions (python.org macOS builds, some pyenv/conda
# builds); Homebrew and Linux distro pythons have it. A "fit" Python therefore
# needs BOTH a usable version AND extension support. Wheel availability is a
# third requirement, verified separately against the created venv because it
# depends on pinned packages.

py_supports_sqlite_ext() {  # $1 = python executable; 0 if it can load extensions
    "$1" -c "import sqlite3,sys; sys.exit(0 if hasattr(sqlite3.connect(':memory:'),'enable_load_extension') else 1)" 2>/dev/null
}

py_is_fit() {  # $1 = python executable; 0 if version>=3.10 AND extensions supported
    command -v "$1" &>/dev/null || return 1
    local minor
    minor=$("$1" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    [ "$minor" -ge 10 ] 2>/dev/null && py_supports_sqlite_ext "$1"
}

# Rebuild .venv from $1 and verify BOTH wheel availability and SQLite extension
# support. 0 on success. Uses $PROBE_PKG (set in Step 2b). An incapable provider
# (e.g. a python.org macOS build) fails the extension check and is rejected here.
rebuild_and_verify_venv() {  # $1 = python executable
    rm -rf .venv
    "$1" -m venv .venv 2>/dev/null || return 1
    .venv/bin/python -m pip install --upgrade pip --quiet </dev/null 2>/dev/null
    .venv/bin/pip install "$PROBE_PKG" --only-binary :all: --force-reinstall --quiet </dev/null 2>/dev/null || return 1
    py_supports_sqlite_ext .venv/bin/python
}

# ──────────────────────────────────────────────────────────────
# Step 1: Find a fit Python 3.10+
# ──────────────────────────────────────────────────────────────
info "Looking for a Python 3.10+ with SQLite extension support..."

PYTHON_CMD=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if py_is_fit "$candidate"; then
        PYTHON_CMD="$candidate"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    warn "No suitable Python 3.10+ found (sqlite-vec needs SQLite extension support)."
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
        if [[ "$OSTYPE" == darwin* ]]; then
            fail "No suitable Python found. On macOS, sqlite-vec needs Homebrew Python (python.org builds will NOT work):\n  1. Install Homebrew:  https://brew.sh\n  2. brew install python@3.12\n  3. ./install.sh"
        else
            fail "Python 3.10+ with SQLite extension support is required. Install it and re-run ./install.sh"
        fi
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
# Step 2b: Validate Python compatibility (wheels + SQLite extensions)
# ──────────────────────────────────────────────────────────────
# Two independent requirements:
#   1. Wheels: install the PINNED pydantic (compiled pydantic-core) with
#      --only-binary :all: — fails instantly if no wheel exists for this Python.
#   2. SQLite extensions: sqlite-vec needs conn.enable_load_extension().
# If EITHER fails, reroute to a Python that satisfies BOTH. This also re-checks a
# .venv that Step 2 may have REUSED, catching a previously-built bad venv.
info "Checking package compatibility..."
PYDANTIC_PIN=$(grep -E '^pydantic==' gateway/requirements.txt 2>/dev/null | head -1)
PROBE_PKG="${PYDANTIC_PIN:-pydantic}"

WHEELS_OK=0
.venv/bin/pip install "$PROBE_PKG" --only-binary :all: --force-reinstall --quiet </dev/null 2>/dev/null && WHEELS_OK=1
EXT_OK=0
py_supports_sqlite_ext .venv/bin/python && EXT_OK=1

if [ "$WHEELS_OK" -eq 1 ] && [ "$EXT_OK" -eq 1 ]; then
    ok "Package compatibility verified (wheels + SQLite extensions)"
else
    [ "$WHEELS_OK" -ne 1 ] && warn "Python $PY_VERSION has no pre-built packages for key dependencies."
    [ "$EXT_OK" -ne 1 ] && warn "Python $PY_VERSION cannot load SQLite extensions (required by sqlite-vec)."
    info "Looking for a Python that satisfies both..."

    # Try candidates in order. Each is fully verified (wheels + extensions) after
    # rebuilding the venv, so an incapable provider is rejected and the next tried.
    REROUTED=0

    # Candidates already installed on the system
    for fb in python3.13 python3.12 python3.11 python3.10; do
        py_is_fit "$fb" || continue
        info "Trying $fb..."
        if rebuild_and_verify_venv "$fb"; then
            PYTHON_CMD="$fb"; REROUTED=1; break
        fi
    done

    # Install a capable Python via Homebrew (macOS & Linux)
    if [ "$REROUTED" -ne 1 ] && command -v brew &>/dev/null; then
        info "Installing Python 3.12 via Homebrew (this may take several minutes)..."
        if brew install python@3.12 </dev/null; then
            BREW_PY="$(brew --prefix)/bin/python3.12"
            command -v "$BREW_PY" &>/dev/null || BREW_PY="python3.12"
            if rebuild_and_verify_venv "$BREW_PY"; then
                PYTHON_CMD="$BREW_PY"; REROUTED=1
            fi
        else
            warn "Homebrew install failed — trying next method..."
        fi
    fi

    # Install via a Linux system package manager
    if [ "$REROUTED" -ne 1 ] && [[ "$OSTYPE" == linux* ]]; then
        LINUX_PY=""
        if command -v apt-get &>/dev/null; then
            info "Installing Python 3.12 via apt..."
            sudo apt-get update -qq </dev/null && sudo apt-get install -y python3.12 python3.12-venv </dev/null 2>/dev/null && LINUX_PY="python3.12"
        elif command -v dnf &>/dev/null; then
            info "Installing Python 3.12 via dnf..."
            sudo dnf install -y python3.12 </dev/null 2>/dev/null && LINUX_PY="python3.12"
        elif command -v pacman &>/dev/null; then
            info "Installing Python via pacman..."
            sudo pacman -S --noconfirm python </dev/null 2>/dev/null && LINUX_PY="python3"
        fi
        if [ -n "$LINUX_PY" ] && rebuild_and_verify_venv "$LINUX_PY"; then
            PYTHON_CMD="$LINUX_PY"; REROUTED=1
        fi
    fi

    # macOS last resort: python.org. Its build lacks SQLite extension support, so
    # rebuild_and_verify_venv will REJECT it — kept only in case a future build
    # enables the flag. The real macOS fix is Homebrew (handled above).
    if [ "$REROUTED" -ne 1 ] && [[ "$OSTYPE" == darwin* ]]; then
        info "Trying python.org Python 3.12..."
        PY_PKG="/tmp/molebie-python-3.12.pkg"
        if curl -fsSL "https://www.python.org/ftp/python/3.12.13/python-3.12.13-macos11.pkg" -o "$PY_PKG" </dev/null 2>/dev/null \
           && sudo installer -pkg "$PY_PKG" -target / </dev/null 2>/dev/null; then
            for pypath in /usr/local/bin/python3.12 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12; do
                [ -x "$pypath" ] && rebuild_and_verify_venv "$pypath" && { PYTHON_CMD="$pypath"; REROUTED=1; break; }
            done
        fi
        rm -f "$PY_PKG"
    fi

    if [ "$REROUTED" -ne 1 ]; then
        echo ""
        if [[ "$OSTYPE" == darwin* ]]; then
            fail "Could not find or install a Python that supports SQLite extensions.\n\n  On macOS, install Homebrew Python (python.org builds will NOT work):\n    1. Install Homebrew:  https://brew.sh\n    2. brew install python@3.12\n    3. rm -rf .venv && ./install.sh"
        else
            fail "Could not find or install a Python that supports SQLite extensions.\n  Install Python 3.10+ with loadable SQLite extension support, then re-run ./install.sh"
        fi
    fi

    PY_VERSION=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    ok "Switched to Python $PY_VERSION ($PYTHON_CMD) — wheels + SQLite extensions verified"
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
