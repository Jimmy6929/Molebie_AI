<p align="center">
  <img src="assets/logo.jpg" alt="Molebie AI" width="480">
</p>

<p align="center">
  <a href="https://github.com/Jimmy6929/Molebie_AI/actions/workflows/ci.yml"><img src="https://github.com/Jimmy6929/Molebie_AI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform">
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
</p>

<p align="center">
  A self-hosted AI assistant with voice conversation, vision, RAG document memory, and web search.<br>
  <strong>Private. Fast. Yours.</strong>
</p>

<p align="center">
  <img src="assets/chat-demo.gif" alt="Chat demo" width="800">
</p>

## Features

- **Three Inference Modes** — Instant (fast), Thinking (chain-of-thought), Think Harder (extended reasoning)
- **Voice Conversation** — Wake-word activation, speech-to-text (Whisper), text-to-speech (Kokoro), speaker verification
- **Image Understanding** — Attach images via file picker, paste, or drag-and-drop
- **RAG Document Memory** — Upload PDFs, DOCX, TXT, MD for persistent knowledge with hybrid vector + BM25 search
- **Web Search** — Self-hosted SearXNG with LLM-powered intent classification and source citation
- **Full Data Ownership** — All data stored locally in SQLite with file-based storage
- **Multi-User** — User isolation from day one, no cloud dependency
- **Any Backend** — Works with MLX, Ollama, vLLM, llama.cpp, or OpenAI API

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/Jimmy6929/Molebie_AI/main/install.sh | bash
```

Or from source:

```bash
git clone https://github.com/Jimmy6929/Molebie_AI.git
cd Molebie_AI
./install.sh
```

<p align="center">
  <img src="assets/install-demo.gif" alt="Installation demo" width="500">
</p>

## Quick Start

```bash
molebie-ai run
```

Auto-detects your system, picks models for your RAM, generates config, and starts all services. Open **http://localhost:3000** and start chatting.

For more control, run `molebie-ai install` to use the interactive setup wizard.

## CLI

| Command | Description |
|---------|-------------|
| `molebie-ai run` | Start all services — auto-configures on first run |
| `molebie-ai install` | Interactive setup wizard |
| `molebie-ai doctor` | Diagnose problems — checks deps, config, and services |
| `molebie-ai status` | Show current config and running services |
| `molebie-ai config env` | List all environment variables |
| `molebie-ai config set KEY=VALUE` | Update a config value |
| `molebie-ai model list` | Show available models and download status |
| `molebie-ai model add 9b` | Download a model |
| `molebie-ai feature add voice` | Enable a feature (also: `search`, `rag`) |

See [Configuration](docs/configuration.md) for the full command reference.

## Documentation

- [Architecture](docs/architecture.md) — System diagram, project structure, multi-machine setup
- [API Reference](docs/api.md) — Endpoints, request/response formats, database schema
- [Configuration](docs/configuration.md) — CLI commands, env vars, models, features, inference modes
- [Contributing](docs/contributing.md) — Dev setup, testing, linting, PR process
- [Troubleshooting](docs/troubleshooting.md) — Common issues and diagnostics

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, Tailwind CSS v4, TypeScript |
| Backend | FastAPI, Uvicorn, Pydantic |
| Auth & DB | SQLite, sqlite-vec, JWT + bcrypt, local file storage |
| Inference | Any OpenAI-compatible server (MLX, Ollama, vLLM, llama.cpp) |
| Embeddings | sentence-transformers |
| Reranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| STT | faster-whisper |
| TTS | Kokoro FastAPI (Docker) |
| Web Search | SearXNG (Docker) |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [Qwen](https://github.com/QwenLM/Qwen3) — Alibaba's open-source LLM family
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Next.js](https://nextjs.org/) — React framework
- [SearXNG](https://github.com/searxng/searxng) — Privacy-respecting metasearch engine
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI) — Fast local text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Fast Whisper inference
