# Contributing

## Development Setup

```bash
git clone https://github.com/Jimmy6929/Molebie_AI.git
cd Molebie_AI
./install.sh
```

## Running Services (Development)

```bash
# CLI (recommended for running)
molebie-ai run           # Start all services (auto-configures on first run)
molebie-ai install       # Interactive setup wizard (optional)
molebie-ai doctor        # Diagnose issues
molebie-ai status        # Check what's running

# Make targets (for development — hot-reload enabled)
make dev-gateway         # Start Gateway API with hot-reload (:8000)
make dev-webapp          # Start Web App (:3000)
make dev-all             # Start all via tmux
make stop                # Stop all services
```

## Testing

```bash
make test                    # All tests
make test-gateway            # Gateway tests only
make test-gateway-cov        # With coverage report
cd webapp && npx tsc --noEmit  # TypeScript check
```

## Linting & Formatting

```bash
make lint                # Lint gateway code (ruff + mypy)
make format              # Format gateway code (black + ruff --fix)
```

## Other Useful Commands

```bash
make cli                 # Install the CLI (pip install -e .)
make db-reset            # Reset SQLite database
make clean               # Remove build artifacts
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push and open a Pull Request
