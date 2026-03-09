# Local AI Assistant - Makefile
# Run `make help` to see available commands

.PHONY: help dev dev-gateway dev-webapp dev-supabase test test-gateway lint format clean install mlx-thinking mlx-install mlx-vlm-install

# Default target
help:
	@echo ""
	@echo "Local AI Assistant - Development Commands"
	@echo "=========================================="
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
	@echo ""
	@echo "  make clean          Remove build artifacts and caches"
	@echo "  make stop           Stop all running services"
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

dev:
	@echo "🚀 Starting all services..."
	@echo "   Run each command in a separate terminal:"
	@echo ""
	@echo "   Terminal 1: make dev-supabase"
	@echo "   Terminal 2: make dev-gateway"
	@echo "   Terminal 3: make dev-webapp"
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
	cd gateway && python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-webapp:
	@echo "🌐 Starting Webapp on http://localhost:3000..."
	cd webapp && npm run dev

stop:
	@echo "🛑 Stopping services..."
	-cd supabase && supabase stop
	-pkill -f "uvicorn app.main:app" 2>/dev/null || true
	@echo "✅ Services stopped"

# ──────────────────────────────────────────────────────────────
# MLX INFERENCE SERVERS (run on GPU machine — M2 Pro)
# ──────────────────────────────────────────────────────────────

mlx-install:
	@echo "📦 Installing mlx-lm (text-only models)..."
	pip install -U mlx-lm

mlx-vlm-install:
	@echo "📦 Installing mlx-vlm (vision-language models)..."
	pip install -U mlx-vlm

mlx-thinking:
	@echo "🧠 Starting MLX-VLM thinking server (Qwen3.5-9B) on :8080..."
	mlx_vlm.server --model mlx-community/Qwen3.5-9B-4bit --host 0.0.0.0 --port 8080

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
