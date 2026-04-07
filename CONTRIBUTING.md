# Contributing to Molebie AI

Thanks for your interest in contributing! Molebie AI is a self-hosted AI assistant focused on privacy, performance, and developer experience. We welcome contributions of all kinds — bug reports, feature ideas, documentation, and code.

## Ways to Contribute

- **Report bugs** — Open an [issue](https://github.com/Jimmy6929/Molebie_AI/issues) with reproduction steps, environment details, and what you expected vs. what happened.
- **Suggest features** — Open an issue describing the use case and why it matters. Discussion before code is encouraged for non-trivial changes.
- **Improve docs** — Typo fixes, clearer explanations, or new guides in `docs/` are all welcome.
- **Submit code** — Bug fixes, new features, performance improvements, or refactors. See the PR process below.
- **Test on your platform** — Try the installer on your OS and report what works/breaks.

## Development Setup

```bash
git clone https://github.com/Jimmy6929/Molebie_AI.git
cd Molebie_AI
./install.sh
```

For the full development command reference (running services, hot-reload, database management), see [docs/contributing.md](docs/contributing.md).

## Coding Standards

### Python (gateway/, cli/)
- Format with `black` (88-char line length)
- Lint with `ruff` — config in `gateway/pyproject.toml`
- Run before committing: `make format && make lint`
- Type hints encouraged for public APIs

### TypeScript (webapp/)
- Run `npx tsc --noEmit` before committing to catch type errors
- Follow existing patterns in `webapp/src/`

### General
- Keep changes focused — one logical change per PR
- Don't mix refactoring with feature work
- Update docs when behavior changes

## Testing

All PRs must pass CI. Before pushing:

```bash
make test            # Run gateway tests
make lint            # Lint with ruff
cd webapp && npx tsc --noEmit   # TypeScript check
```

If you're adding new functionality, add tests in `gateway/tests/`. Look at `test_web_search_routing.py` for the testing pattern.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** — keep commits focused and atomic
3. **Test locally** — run `make test` and `make lint`
4. **Push** your branch and open a Pull Request against `main`
5. **Describe** the change in the PR body:
   - What does it do?
   - Why is this needed?
   - How was it tested?
   - Screenshots/recordings for UI changes
6. **Respond to feedback** — CI must pass and reviewers may request changes

### PR Checklist

- [ ] Code follows the project's style (`make format && make lint` passes)
- [ ] Tests pass locally (`make test`)
- [ ] New functionality has tests
- [ ] Documentation updated if behavior changed
- [ ] Commit messages are descriptive
- [ ] PR description explains the *why*, not just the *what*

## Commit Message Conventions

Use clear, present-tense commit messages. The first line should be a short summary (≤72 chars), followed by an optional body explaining the *why*.

Good examples:
```
fix: prevent duplicate session creation on rapid clicks
feat(rag): add cross-encoder reranking for retrieved chunks
docs: add multi-machine setup guide
```

Prefixes are optional but helpful: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`.

## Reporting Security Issues

Please **do not** open public issues for security vulnerabilities. Instead, report them privately by opening a [GitHub security advisory](https://github.com/Jimmy6929/Molebie_AI/security/advisories/new).

## Code of Conduct

Be respectful, constructive, and patient. We're all here to build something useful together. Harassment, personal attacks, or discriminatory behavior will not be tolerated.

## Questions?

Open an issue with the `question` label, or start a discussion. We're happy to help newcomers get oriented.
