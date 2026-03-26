#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Molebie AI — Setup Script
# ══════════════════════════════════════════════════════════════
# Checks prerequisites, installs dependencies, generates .env.local
# with a JWT secret, and creates the data directory.
#
# No Docker or Supabase required for the database.
# Docker is only needed for optional services (SearXNG, Kokoro TTS).
#
# Usage:  bash setup.sh
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
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Molebie AI — Setup${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo ""

# ──────────────────────────────────────────────────────────────
# Phase 1: Check prerequisites
# ──────────────────────────────────────────────────────────────
info "Checking prerequisites..."
MISSING=0

# Node.js
if command -v node &>/dev/null; then
    NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -ge 18 ] 2>/dev/null; then
        ok "Node.js $(node -v)"
    else
        echo -e "${RED}[MISSING]${NC} Node.js 18+ required (found $(node -v)) — brew install node"
        MISSING=1
    fi
else
    echo -e "${RED}[MISSING]${NC} Node.js — brew install node"
    MISSING=1
fi

# Python 3
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.minor}')")
    if [ "$PY_VER" -ge 10 ] 2>/dev/null; then
        ok "Python 3.$(python3 -c 'import sys; print(sys.version_info.minor)')"
    else
        echo -e "${RED}[MISSING]${NC} Python 3.10+ required — brew install python"
        MISSING=1
    fi
else
    echo -e "${RED}[MISSING]${NC} Python 3 — brew install python"
    MISSING=1
fi

# Docker (optional — for SearXNG, Kokoro TTS)
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    ok "Docker found (optional services available: web search, TTS)"
else
    warn "Docker not found — optional services (web search, TTS) won't be available"
fi

# ffmpeg (optional — for voice features)
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg (voice features enabled)"
else
    warn "ffmpeg not found — voice features won't work. Install later: brew install ffmpeg"
fi

if [ "$MISSING" -ne 0 ]; then
    echo ""
    fail "Missing prerequisites. Install them and run setup.sh again."
fi

echo ""

# ──────────────────────────────────────────────────────────────
# Phase 2: Choose deployment mode
# ──────────────────────────────────────────────────────────────
DEPLOY_MODE="single"
GPU_IP="localhost"
SERVER_IP="localhost"

echo -e "${BOLD}How will you run this?${NC}"
echo "  1) Single machine — everything on this Mac [default]"
echo "  2) Two machines — GPU on a separate Mac (Tailscale/LAN)"
echo ""
read -rp "Choose [1]: " MODE_CHOICE
MODE_CHOICE="${MODE_CHOICE:-1}"

if [ "$MODE_CHOICE" = "2" ]; then
    DEPLOY_MODE="two-machine"
    echo ""
    read -rp "  IP of GPU machine (runs MLX inference): " GPU_IP
    read -rp "  IP of THIS machine (runs gateway/webapp): " SERVER_IP
    if [ -z "$GPU_IP" ] || [ -z "$SERVER_IP" ]; then
        fail "Both IPs are required for two-machine setup."
    fi
    ok "Two-machine mode: GPU=$GPU_IP, Server=$SERVER_IP"
else
    ok "Single-machine mode: all services on localhost"
fi

echo ""

# ──────────────────────────────────────────────────────────────
# Phase 3: Generate .env.local with JWT secret
# ──────────────────────────────────────────────────────────────
if [ -f .env.local ]; then
    warn ".env.local already exists — skipping generation. Delete it to regenerate."
else
    info "Generating .env.local from template..."
    cp .env.example .env.local

    # Generate a random JWT secret
    JWT_SECRET_VAL=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i '' "s|JWT_SECRET=CHANGE_ME_TO_A_RANDOM_SECRET|JWT_SECRET=${JWT_SECRET_VAL}|" .env.local

    if [ "$DEPLOY_MODE" = "two-machine" ]; then
        sed -i '' "s|INFERENCE_THINKING_URL=http://localhost:8080|INFERENCE_THINKING_URL=http://${GPU_IP}:8080|" .env.local
        sed -i '' "s|INFERENCE_INSTANT_URL=http://localhost:8081|INFERENCE_INSTANT_URL=http://${GPU_IP}:8081|" .env.local
        sed -i '' "s|NEXT_PUBLIC_GATEWAY_URL=http://localhost:8000|NEXT_PUBLIC_GATEWAY_URL=http://${SERVER_IP}:8000|" .env.local
        sed -i '' "s|CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000|CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://${SERVER_IP}:3000|" .env.local
    fi

    ok ".env.local created with JWT secret"
fi

echo ""

# ──────────────────────────────────────────────────────────────
# Phase 4: Install dependencies
# ──────────────────────────────────────────────────────────────
info "Installing Gateway dependencies (Python)..."
pip3 install -r gateway/requirements.txt --quiet 2>&1 | tail -1
ok "Gateway dependencies installed"

info "Installing Webapp dependencies (Node.js)..."
cd webapp && npm install --silent 2>&1 | tail -1
cd "$SCRIPT_DIR"
ok "Webapp dependencies installed"

echo ""

# ──────────────────────────────────────────────────────────────
# Phase 5: Create data directory
# ──────────────────────────────────────────────────────────────
info "Creating data directory..."
mkdir -p data
ok "Data directory created (SQLite database will be initialized on first start)"

echo ""

# ──────────────────────────────────────────────────────────────
# Done!
# ──────────────────────────────────────────────────────────────
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo ""

if [ "$DEPLOY_MODE" = "two-machine" ]; then
    echo "  1. On the GPU machine ($GPU_IP):"
    echo "     make mlx-thinking        # Start thinking LLM"
    echo "     make mlx-instant         # Start instant LLM (optional)"
    echo ""
    echo "  2. On this machine ($SERVER_IP):"
    echo "     make dev-gateway         # Terminal 1"
    echo "     make dev-webapp          # Terminal 2"
    echo ""
    echo "  3. Optional: docker compose up -d  (web search + TTS)"
    echo ""
    echo "  4. Open http://${SERVER_IP}:3000 and set your password"
else
    echo "  1. Start MLX inference (in a new terminal):"
    echo "     make mlx-thinking        # Required — Qwen 3.5 9B"
    echo ""
    echo "  2. Start the app:"
    echo "     make dev-gateway         # Terminal 1"
    echo "     make dev-webapp          # Terminal 2"
    echo ""
    echo "  3. Optional: docker compose up -d  (web search + TTS)"
    echo ""
    echo "  4. Open http://localhost:3000 and set your password"
fi

echo ""
