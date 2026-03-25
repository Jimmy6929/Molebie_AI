# Molebie AI

A self-hosted AI assistant with voice conversation, vision, RAG document memory, and web search. Private. Fast. Yours.

## Features

- **Three Inference Modes** — Instant (fast), Thinking (chain-of-thought), Think Harder (extended reasoning)
- **Voice Conversation** — Wake-word activation, speech-to-text (Whisper), text-to-speech (Kokoro), speaker verification
- **Image Understanding** — Attach images via file picker, paste, or drag-and-drop
- **RAG Document Memory** — Upload PDFs, DOCX, TXT, MD for persistent knowledge with hybrid vector + BM25 search, cross-encoder reranking
- **Web Search** — Self-hosted SearXNG with LLM-powered intent classification and source citation
- **Full Data Ownership** — All data stored in your own Supabase instance
- **Multi-User** — Row-level security from day one
- **Any Backend** — Works with MLX, Ollama, vLLM, llama.cpp, or OpenAI API

## Quick Start

```bash
git clone git@github.com:Jimmy6929/Molebie_AI.git
cd molebie-ai
./install.sh
```

That's it. The installer handles everything:

1. **Checks your system** — OS, chip, RAM, disk space
2. **Installs missing tools** — Docker, Node.js, Supabase CLI (asks first)
3. **Detects the best backend** — recommends MLX on Apple Silicon, Ollama elsewhere
4. **Lets you choose models** — Light (8+ GB RAM) or Balanced (16+ GB RAM)
5. **Installs the backend** — installs `mlx-vlm` and downloads models, or pulls Ollama models
6. **Configures features** — web search (SearXNG), RAG document memory, voice
7. **Sets up services** — starts Docker containers, Supabase, and generates all config files
8. **Offers to start** — launches everything with one confirmation

After setup, start the system any time with:

```bash
./bin/molebie-ai run
```

Open **http://localhost:3000**, create an account, and start chatting.

### CLI Commands

| Command | Description |
|---------|-------------|
| `molebie-ai install` | Interactive setup wizard — installs backend, downloads models, configures everything |
| `molebie-ai run` | Start all configured services (Supabase, Gateway, Webapp, inference) |
| `molebie-ai doctor` | Diagnose environment problems — checks dependencies, config, and service health |
| `molebie-ai status` | Show current configuration and which services are running |
| `molebie-ai config show` | Display saved configuration (JSON) |
| `molebie-ai feature list` | Show optional features and their status |
| `molebie-ai feature add voice` | Enable a feature (also: `search`, `rag`) |
| `molebie-ai feature remove voice` | Disable a feature |

### Alternative: Manual Setup

If you prefer to set up manually or need finer control:

```bash
bash setup.sh
```

`setup.sh` checks prerequisites, installs dependencies, starts Supabase, and generates `.env.local`. You then start services individually:

```bash
# Start an inference backend (pick one)
make mlx-thinking          # MLX on Apple Silicon (downloads ~5GB model)
# ollama serve             # Ollama (cross-platform)

# Start services in separate terminals
make dev-gateway           # Gateway API (:8000)
make dev-webapp            # Web App (:3000)
docker compose up -d       # Optional: web search + TTS
```

### Supported Inference Backends

| Backend | Command | Notes |
|---------|---------|-------|
| **MLX** (Apple Silicon) | `make mlx-thinking` | Best for Mac. Auto-installed by CLI |
| **Ollama** | `ollama serve` | Easiest cross-platform. Auto-configured by CLI |
| **vLLM** | `vllm serve <model>` | Production GPU servers |
| **llama.cpp** | `llama-server -m <model>` | Lightweight, any hardware |
| **OpenAI API** | Set `INFERENCE_API_KEY` in `.env.local` | Cloud fallback |

## System Requirements

- **OS**: macOS, Linux, or Windows (WSL2)
- **RAM**: 8GB minimum, 16GB+ recommended for local inference
- **GPU**: Recommended for local LLM inference (Apple Silicon, NVIDIA)
- **Disk**: ~2-10GB per model depending on quantization
- **Docker**: Required for Supabase, optional services (SearXNG, Kokoro TTS)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Web App (Next.js 16)                       │   │
│  │  - Chat UI (Streaming)    - Deep Think Toggle           │   │
│  │  - Session History        - Voice Conversation          │   │
│  │  - Image Upload/Paste     - Document Brain (RAG)        │   │
│  │  - Auth via Supabase      - Web Search Sources          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS + JWT
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL PLANE                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Gateway API (FastAPI)                      │   │
│  │  - JWT Validation      - Session Management             │   │
│  │  - Request Routing     - SSE Streaming                  │   │
│  │  - Web Search (SearXNG)- Voice (STT + TTS + Speaker)    │   │
│  │  - RAG Pipeline        - Image/Vision Handling          │   │
│  │  - Logging & Audit     - Cost Controls                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │             │              │              │
          ▼             ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ THINKING LLM │ │ INSTANT LLM  │ │ SearXNG  │ │ Kokoro TTS   │
│ (any model)  │ │ (any model)  │ │ :8888    │ │ :8880        │
│ :8080        │ │ :8081        │ │ (Docker)  │ │ (Docker)     │
└──────────────┘ └──────────────┘ └──────────┘ └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Supabase                                   │   │
│  │  - Auth (Users, JWT)     - Storage (Documents + Images)  │   │
│  │  - Postgres (Data)       - pgvector (RAG Embeddings)    │   │
│  │  - RLS (Multi-tenant)    - BM25 Full-Text Search        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
molebie-ai/
├── pyproject.toml                   # CLI package (pip install -e .)
├── .env.example                     # Environment config template
├── Makefile                         # Development commands
├── setup.sh                         # Alternative manual setup script
├── docker-compose.yml               # SearXNG + Kokoro TTS
│
├── cli/                             # CLI (Python + Typer)
│   ├── main.py                      # Typer app entry point
│   ├── commands/                    # install, run, doctor, status, config, feature
│   ├── services/                    # Backend setup, feature setup, system detection
│   ├── models/                      # Config schema (Pydantic)
│   └── ui/                          # Console output + interactive prompts (Rich)
│
├── gateway/                         # FastAPI Backend (Python)
│   ├── app/
│   │   ├── main.py                  # Application entry point
│   │   ├── config.py                # Configuration & settings
│   │   ├── middleware/auth.py       # JWT auth middleware
│   │   ├── models/                  # Pydantic request/response models
│   │   ├── routes/
│   │   │   ├── health.py            # Health check endpoints
│   │   │   ├── chat.py              # Chat, voice, TTS, vision endpoints
│   │   │   └── documents.py         # Document upload & RAG management
│   │   └── services/                # Business logic (inference, search, RAG, voice)
│   ├── prompts/                     # Customizable system prompts
│   │   ├── system.txt               # Main assistant personality
│   │   └── system_voice.txt         # Voice conversation personality
│   ├── tests/                       # Pytest test suite
│   └── requirements.txt
│
├── webapp/                          # Next.js Frontend (TypeScript)
│   └── src/
│       ├── app/chat/                # Chat UI, sidebar, voice settings
│       └── lib/                     # API client, voice hooks, Supabase
│
├── supabase/                        # Database (Postgres + pgvector)
│   ├── config.toml
│   └── migrations/                  # SQL migrations
│
├── searxng/                         # SearXNG search config
│   └── settings.yml
│
└── scripts/                         # MLX server wrapper, auto-pull daemon
```

## Configuration

### Customizing the Assistant Personality

Edit the prompt files in `gateway/prompts/`:
- **`system.txt`** — Main chat personality and behavior
- **`system_voice.txt`** — Voice conversation mode (shorter, optimized for speech)

Template variables: `{current_date}` (auto-injected)

### Environment Variables

All config lives in `.env.local` at the project root. Copy the template:

```bash
cp .env.example .env.local
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ASSISTANT_NAME` | Assistant name (for logs/metadata) | `Assistant` |
| `PROMPT_DIR` | Directory for system prompts | `prompts` |
| `INFERENCE_API_KEY` | API key for commercial backends | — |
| `INFERENCE_THINKING_URL` | Thinking tier endpoint | `http://localhost:8080` |
| `INFERENCE_INSTANT_URL` | Instant tier endpoint | `http://localhost:8081` |
| `WEB_SEARCH_ENABLED` | Enable web search | `true` |
| `RAG_ENABLED` | Enable document RAG | `true` |
| `ROUTING_DEFAULT_MODE` | Default inference mode | `thinking` |

See `.env.example` for the complete list with descriptions.

### Optional Features

Toggle features via the CLI or environment variables:

```bash
molebie-ai feature list          # See what's enabled
molebie-ai feature add voice     # Enable voice
molebie-ai feature remove search # Disable search
```

| Feature | CLI Toggle | Env Variable | Setup |
|---------|-----------|-------------|-------|
| Web Search | `molebie-ai feature add search` | `WEB_SEARCH_ENABLED` | SearXNG Docker container |
| Text-to-Speech | `molebie-ai feature add voice` | — | Kokoro TTS Docker container + ffmpeg |
| RAG Documents | `molebie-ai feature add rag` | `RAG_ENABLED` | Embedding model auto-downloads on first use |
| Image Vision | Always available | — | Requires a vision-capable model |

### Custom Wake Words

Edit `WAKE_PHRASES` in `webapp/src/lib/voice.ts`:

```typescript
const WAKE_PHRASES = ["hey assistant", "hello assistant", "hi assistant"];
```

## Inference Modes

| Mode | Use Case | Thinking Budget |
|------|----------|-----------------|
| **Instant** | Quick answers, voice mode | None (fast) |
| **Thinking** | Reasoning, coding, analysis | 2,048 tokens |
| **Think Harder** | Complex problems | 8,192 tokens |

- Thinking blocks are collapsible in the UI
- Automatic fallback: if thinking tier is down, requests fall back to instant

### Cost Controls

| Variable | Description | Default |
|----------|-------------|---------|
| `THINKING_DAILY_REQUEST_LIMIT` | Max thinking requests/user/day | `100` |
| `THINKING_MAX_CONCURRENT` | Max parallel thinking requests | `2` |

## Multi-Machine Setup

Split across two machines — a GPU machine for inference and a server for everything else:

```
Browser → Webapp (:3000) → Gateway (:8000) ──┬──→ Thinking LLM (:8080)  ← GPU machine
              │                                └──→ Instant LLM  (:8081)  ← GPU machine
              └──→ Supabase (:54321)                               ← Server
```

Run `molebie-ai install` and choose "Two machines" when prompted. The installer will:
- Ask for your GPU machine IP and server IP
- Configure all endpoints in `.env.local` automatically
- Update Supabase auth redirects for cross-machine access
- Print exact commands to run on the GPU machine

On the **server machine** (runs gateway, webapp, database):
```bash
molebie-ai install    # Choose "Two machines", enter IPs
molebie-ai run        # Starts Supabase, Gateway, Webapp, Docker services
```

On the **GPU machine** (runs inference only):
```bash
python -m pip install -U mlx-vlm
make mlx-thinking     # Port 8080
make mlx-instant      # Port 8081 (optional)
```

The CLI verifies the remote GPU is reachable before starting local services.

## API Reference

### Health Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Basic health check |
| `/health/auth` | GET | Yes | Validates JWT, returns user info |
| `/health/inference` | GET | No | Instant + thinking tier status |

### Chat Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat` | POST | Yes | Send message, receive full response |
| `/chat/stream` | POST | Yes | Send message, receive SSE stream |
| `/chat/sessions` | GET | Yes | List chat sessions |
| `/chat/sessions/create` | POST | Yes | Create empty session |
| `/chat/sessions/{id}/messages` | GET | Yes | Get session messages |
| `/chat/sessions/{id}` | PATCH | Yes | Rename session |
| `/chat/sessions/{id}` | DELETE | Yes | Delete session |

### Document Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/documents/upload` | POST | Yes | Upload document for RAG |
| `/documents` | GET | Yes | List documents |
| `/documents/{id}` | DELETE | Yes | Delete document |
| `/documents/sessions/{id}/attach` | POST | Yes | Attach doc to session |

### Voice Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat/transcribe` | POST | Yes | Speech-to-text (Whisper) |
| `/chat/tts` | POST | Yes | Text-to-speech (Kokoro, returns MP3) |
| `/chat/voice-enroll` | POST | Yes | Enroll voice sample |
| `/chat/voice-profile` | GET/DELETE | Yes | Manage voice profile |

### Chat Request/Response

```json
// POST /chat or /chat/stream
{
  "session_id": "uuid | null",
  "message": "string",
  "mode": "instant | thinking | thinking_harder",
  "conversation_mode": false,
  "image": "data:image/jpeg;base64,... | null"
}
```

## Database Schema

| Table | Description |
|-------|-------------|
| `profiles` | User profiles (auto-created on signup) |
| `chat_sessions` | Conversations with pinning support |
| `chat_messages` | Messages with `reasoning_content` and `mode_used` |
| `message_images` | Image attachments (metadata; bytes in Supabase Storage) |
| `documents` | Uploaded document metadata |
| `document_chunks` | Chunks with pgvector embeddings + tsvector for BM25 |
| `session_documents` | Per-session document attachments |

All tables have Row-Level Security enabled — users can only access their own data.

## Development

### Commands

```bash
# CLI (recommended)
molebie-ai install       # Guided setup wizard
molebie-ai run           # Start all services
molebie-ai doctor        # Diagnose issues
molebie-ai status        # Check what's running

# Make targets (for individual service control)
make dev-gateway         # Start Gateway API (:8000)
make dev-webapp          # Start Web App (:3000)
make dev-all             # Start all via tmux
make test                # Run tests
make lint                # Lint gateway code
make format              # Format gateway code
make db-reset            # Reset database
make clean               # Remove build artifacts
make stop                # Stop all services
make cli                 # Install the CLI (pip install -e .)
```

### Testing

```bash
make test                    # All tests
make test-gateway            # Gateway tests only
make test-gateway-cov        # With coverage report
cd webapp && npx tsc --noEmit  # TypeScript check
```

### Troubleshooting

Run `molebie-ai doctor` for a full diagnostic — it checks dependencies, config files, and service health with suggested fixes.

| Problem | Solution |
|---------|----------|
| Something seems wrong | `molebie-ai doctor` — checks everything and suggests fixes |
| `uvicorn: command not found` | Use `python3 -m uvicorn` (Makefile does this) |
| Address already in use | `lsof -i :<port>` then `kill` the process |
| Supabase 401 errors | Run `supabase status` and check keys match `.env.local` |
| Voice transcription fails | `brew install ffmpeg` |
| OMP error on macOS | `make dev-gateway` sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically |
| Config looks wrong | `molebie-ai config show` to inspect, `molebie-ai install` to reconfigure |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, Tailwind CSS v4, TypeScript |
| Backend | FastAPI, Uvicorn, Pydantic |
| Auth & DB | Supabase (Auth, Postgres, pgvector, Storage) |
| Inference | Any OpenAI-compatible server (MLX, Ollama, vLLM, llama.cpp) |
| Embeddings | sentence-transformers |
| Reranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| STT | faster-whisper |
| TTS | Kokoro FastAPI (Docker) |
| Web Search | SearXNG (Docker) |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push and open a Pull Request

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [Qwen](https://github.com/QwenLM/Qwen3) — Alibaba's open-source LLM family
- [Supabase](https://supabase.com/) — Backend as a Service
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Next.js](https://nextjs.org/) — React framework
- [SearXNG](https://github.com/searxng/searxng) — Privacy-respecting metasearch engine
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI) — Fast local text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Fast Whisper inference

---

Built for privacy-conscious developers who want to own their data and run everything locally.
