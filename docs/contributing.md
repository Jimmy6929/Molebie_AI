# Developer Reference

This is the day-to-day command reference for Molebie AI development. For the contribution process, code of conduct, and PR guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Initial Setup

```bash
git clone https://github.com/Jimmy6929/Molebie_AI.git
cd Molebie_AI
./install.sh
```

The installer creates a Python venv, installs the CLI, and launches the interactive setup wizard.

## Project Layout

```
gateway/      # FastAPI backend (Python)
webapp/       # Next.js frontend (TypeScript)
cli/          # molebie-ai CLI (Python + Typer)
docs/         # Public documentation
scripts/      # Deployment scripts (auto-pull, MLX wrapper)
data/         # Runtime data — SQLite DB, uploaded files (gitignored)
```

## Running Services

### Production-style (CLI)

```bash
molebie-ai run           # Start everything — auto-configures on first run
molebie-ai install       # Re-run the interactive wizard
molebie-ai doctor        # Diagnose dependency / config issues
molebie-ai status        # Show running services and config
molebie-ai stop          # Stop all services
```

### Development (hot-reload)

```bash
make dev-gateway         # Gateway API with --reload (port 8000)
make dev-webapp          # Next.js dev server (port 3000)
make dev-all             # Start all services in tmux panes
make stop                # Stop all services started by make dev-*
```

### Inference servers (MLX, Apple Silicon)

```bash
make mlx-install         # Install mlx-lm
make mlx-vlm-install     # Install mlx-vlm (vision)
make mlx-thinking        # Start thinking-tier server (port 8080)
make mlx-instant         # Start instant-tier server (port 8081)
```

## Testing

```bash
make test                       # Run all gateway tests
make test-gateway               # Same as above (explicit)
make test-gateway-cov           # With coverage report
cd webapp && npx tsc --noEmit   # TypeScript type check
```

Tests live in `gateway/tests/`. The test pattern is shown in `test_web_search_routing.py`. Use `pytest` fixtures from `conftest.py` for environment setup.

## Linting & Formatting

```bash
make lint                # ruff check (gateway code)
make format              # black + ruff --fix
```

Ruff config is in `gateway/pyproject.toml` under `[tool.ruff.lint]`.

## Database

```bash
make db-reset            # Delete and recreate the SQLite database
```

The database lives at `data/molebie.db`. Schema is initialized on first gateway start by `gateway/app/schema.py`.

## Configuration

Local environment goes in `.env.local` at the repo root (gitignored). Manage it via:

```bash
molebie-ai config env                # List all variables
molebie-ai config get KEY            # Show one value
molebie-ai config set KEY=VALUE      # Update a value
molebie-ai config init               # Regenerate from .env.example
```

See [configuration.md](configuration.md) for the full variable reference.

## Cleanup

```bash
make clean               # Remove __pycache__, .pytest_cache, .next, build artifacts
```

## Auto-Pull (macOS Home Server)

For running Molebie AI as a home server with automatic updates:

```bash
make autopull-install    # Install LaunchAgent that polls git every 5 min
make autopull-status     # Check if the agent is running
make autopull-logs       # Tail the auto-pull log
make autopull-uninstall  # Remove the LaunchAgent
```

## Useful File Locations

| What | Where |
|------|-------|
| System prompts | `gateway/prompts/system.txt`, `gateway/prompts/system_voice.txt` |
| Gateway routes | `gateway/app/routes/` |
| Gateway services | `gateway/app/services/` |
| Webapp pages | `webapp/src/app/` |
| Webapp lib | `webapp/src/lib/` |
| CLI commands | `cli/commands/` |
| Logs | `data/logs/gateway.log` |
| SQLite DB | `data/molebie.db` |
