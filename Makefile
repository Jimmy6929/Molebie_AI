# Local AI Assistant - Makefile
# Run `make help` to see available commands

.PHONY: help setup dev dev-gateway dev-webapp dev-supabase test test-gateway lint format clean install mlx-thinking mlx-instant mlx-install mlx-vlm-install autopull-install autopull-uninstall autopull-status autopull-logs autopull-diagnose

# Default target
help:
	@echo ""
	@echo "Local AI Assistant - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "  make setup          First-time setup (checks prereqs, installs deps, configures Supabase)"
	@echo ""
	@echo "  make install        Install all dependencies"
	@echo "  make dev            Start all services (supabase, gateway, webapp)"
	@echo "  make dev-gateway    Start only the gateway API"
	@echo "  make dev-webapp     Start only the webapp"
	@echo "  make dev-supabase   Start only Supabase local"
	@echo ""
	@echo "  make test           Run all tests"
	@echo "  make test-gateway   Run gateway tests only"
	@echo ""
	@echo "  make lint           Run linters on gateway code"
	@echo "  make format         Format gateway code with black"
	@echo ""
	@echo "  make mlx-install      Install mlx-lm on GPU machine"
	@echo "  make mlx-vlm-install  Install mlx-vlm on GPU machine"
	@echo "  make mlx-thinking     Start MLX-VLM thinking server (Qwen3.5-9B, :8080)"
	@echo "  make mlx-instant      Start MLX-VLM instant server (:8081)"
	@echo ""
	@echo "  make clean          Remove build artifacts and caches"
	@echo "  make stop           Stop all running services"
	@echo ""
	@echo "  make autopull-install    Install auto-pull service (MacBook Pro 2016)"
	@echo "  make autopull-uninstall  Remove auto-pull service"
	@echo "  make autopull-status     Check auto-pull service status"
	@echo "  make autopull-logs       Tail auto-pull log file"
	@echo "  make autopull-diagnose   Diagnose auto-pull issues"
	@echo ""

# ──────────────────────────────────────────────────────────────
# INSTALLATION
# ──────────────────────────────────────────────────────────────

install: install-gateway install-webapp
	@echo "✅ All dependencies installed"

install-gateway:
	@echo "📦 Installing gateway dependencies..."
	cd gateway && pip install -r requirements.txt

install-gateway-dev:
	@echo "📦 Installing gateway dev dependencies..."
	cd gateway && pip install -e ".[dev]"

install-webapp:
	@echo "📦 Installing webapp dependencies..."
	cd webapp && npm install

# ──────────────────────────────────────────────────────────────
# DEVELOPMENT SERVERS
# ──────────────────────────────────────────────────────────────

setup:
	@bash setup.sh

dev:
	@echo "🚀 Starting all services..."
	@echo "   First-time? Run: make setup"
	@echo ""
	@echo "   Run each command in a separate terminal:"
	@echo ""
	@echo "   Terminal 1: make mlx-thinking     (MLX inference)"
	@echo "   Terminal 2: make dev-gateway      (API server)"
	@echo "   Terminal 3: make dev-webapp       (frontend)"
	@echo "   Background: docker compose up -d  (search + TTS)"
	@echo ""
	@echo "   Or use: make dev-all (requires tmux)"

dev-all:
	@command -v tmux >/dev/null 2>&1 || { echo "❌ tmux is required for dev-all. Install with: brew install tmux"; exit 1; }
	tmux new-session -d -s localai 'make dev-supabase' \; \
		split-window -h 'sleep 5 && make dev-gateway' \; \
		split-window -v 'sleep 8 && make dev-webapp' \; \
		attach

dev-supabase:
	@echo "🗄️  Starting Supabase local..."
	cd supabase && supabase start

dev-gateway:
	@echo "⚡ Starting Gateway API on http://localhost:8000..."
	cd gateway && KMP_DUPLICATE_LIB_OK=TRUE python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-webapp:
	@echo "🌐 Starting Webapp on http://localhost:3000..."
	cd webapp && npm run dev

stop:
	@echo "🛑 Stopping services..."
	-cd supabase && supabase stop
	-pkill -f "uvicorn app.main:app" 2>/dev/null || true
	@echo "✅ Services stopped"

# ──────────────────────────────────────────────────────────────
# MLX INFERENCE SERVERS (Apple Silicon GPU)
# ──────────────────────────────────────────────────────────────

mlx-install:
	@echo "📦 Installing mlx-lm (text-only models)..."
	pip install -U mlx-lm

mlx-vlm-install:
	@echo "📦 Installing mlx-vlm (vision-language models)..."
	pip install -U mlx-vlm

mlx-thinking:
	@echo "🧠 Starting MLX-VLM thinking server (Qwen3.5-9B) on :8080..."
	python3 scripts/mlx_server.py --host 0.0.0.0 --port 8080

mlx-instant:
	@echo "⚡ Starting MLX-VLM instant server on :8081..."
	python3 scripts/mlx_server.py --host 0.0.0.0 --port 8081

# ──────────────────────────────────────────────────────────────
# TESTING
# ──────────────────────────────────────────────────────────────

test: test-gateway
	@echo "✅ All tests passed"

test-gateway:
	@echo "🧪 Running gateway tests..."
	cd gateway && pytest tests/ -v

test-gateway-cov:
	@echo "🧪 Running gateway tests with coverage..."
	cd gateway && pytest tests/ --cov=app --cov-report=html --cov-report=term

# ──────────────────────────────────────────────────────────────
# LINTING & FORMATTING
# ──────────────────────────────────────────────────────────────

lint:
	@echo "🔍 Linting gateway code..."
	cd gateway && ruff check app/ tests/
	cd gateway && mypy app/

format:
	@echo "✨ Formatting gateway code..."
	cd gateway && black app/ tests/
	cd gateway && ruff check --fix app/ tests/

# ──────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────

db-reset:
	@echo "🗑️  Resetting database..."
	cd supabase && supabase db reset

db-migrate:
	@echo "📝 Running migrations..."
	cd supabase && supabase db push

db-studio:
	@echo "🎨 Opening Supabase Studio..."
	@echo "   Visit: http://localhost:54323"

# ──────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────

clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf webapp/.next 2>/dev/null || true
	@echo "✅ Clean complete"

# ──────────────────────────────────────────────────────────────
# AUTO-PULL (home server — polls git and auto-updates)
# ──────────────────────────────────────────────────────────────

autopull-install:
	@echo "Installing auto-pull service..."
	chmod +x scripts/auto-pull.sh
	mkdir -p logs
	-launchctl unload ~/Library/LaunchAgents/com.jimmy.localai.autopull.plist 2>/dev/null
	sed -e 's|__REPO_DIR__|$(CURDIR)|g' -e 's|__HOME_DIR__|$(HOME)|g' \
		scripts/com.jimmy.localai.autopull.plist > ~/Library/LaunchAgents/com.jimmy.localai.autopull.plist
	launchctl load ~/Library/LaunchAgents/com.jimmy.localai.autopull.plist
	@echo "✅ Auto-pull service installed and running (polls every 60s)"

autopull-uninstall:
	@echo "Removing auto-pull service..."
	launchctl unload ~/Library/LaunchAgents/com.jimmy.localai.autopull.plist 2>/dev/null || true
	rm -f ~/Library/LaunchAgents/com.jimmy.localai.autopull.plist
	@echo "✅ Auto-pull service removed"

autopull-status:
	@launchctl list | grep localai.autopull || echo "Auto-pull service is not running"
	@echo "--- Last heartbeat ---"
	@grep "HEARTBEAT" logs/auto-pull.log 2>/dev/null | tail -1 || echo "No heartbeat found"
	@echo "--- Recent log entries ---"
	@tail -20 logs/auto-pull.log 2>/dev/null || echo "No log file yet"

autopull-logs:
	@tail -f logs/auto-pull.log

autopull-diagnose:
	@echo "=== Auto-Pull Diagnostics ==="
	@echo ""
	@echo "1. Service loaded?"
	@launchctl list | grep localai.autopull && echo "   OK" || echo "   FAIL: service not loaded — run: make autopull-install"
	@echo ""
	@echo "2. Script executable?"
	@test -x scripts/auto-pull.sh && echo "   OK" || echo "   FAIL: run: chmod +x scripts/auto-pull.sh"
	@echo ""
	@echo "3. Uncommitted changes?"
	@test -z "$$(git status --porcelain)" && echo "   OK: working tree clean" || echo "   WARN: uncommitted changes — auto-pull will skip"
	@echo ""
	@echo "4. Plist installed?"
	@test -f ~/Library/LaunchAgents/com.jimmy.localai.autopull.plist && echo "   OK" || echo "   FAIL: plist not in ~/Library/LaunchAgents/ — run: make autopull-install"
	@echo ""
	@echo "5. Git fetch works?"
	@git fetch --dry-run origin main 2>&1 && echo "   OK" || echo "   FAIL: credentials may need refreshing — run: git fetch origin main"
	@echo ""
	@echo "6. Recent heartbeats (last 5)?"
	@grep "HEARTBEAT" logs/auto-pull.log 2>/dev/null | tail -5 || echo "   NONE: no heartbeats found — service may not be running"
	@echo ""
	@echo "7. Recent errors?"
	@grep "ERROR" logs/auto-pull.log 2>/dev/null | tail -5 || echo "   NONE: no errors found"
	@echo ""
	@echo "8. Stderr log?"
	@tail -10 logs/auto-pull-stderr.log 2>/dev/null || echo "   (empty or missing — that's fine)"
