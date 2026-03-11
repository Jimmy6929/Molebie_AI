# Local AI Assistant

A private, self-hosted AI assistant system with multi-user support, three-tier inference, voice conversation, web search, and full data ownership.

## Overview

Local AI Assistant is a production-ready chat application that provides a ChatGPT-like experience while running entirely on your own infrastructure. It features:

- **Three Inference Modes**: Instant (fast), Thinking (chain-of-thought), and Think Harder (extended reasoning)
- **SSE Streaming**: Real-time token-by-token response streaming with thinking block support
- **Voice Conversation ("Alfred")**: Wake-word activation, speech-to-text (Whisper), text-to-speech (Kokoro), and speaker verification
- **Web Search**: Self-hosted SearXNG integration — the AI can search the web and cite sources
- **Full Data Ownership**: All conversations and documents stored in your Supabase instance
- **Multi-User Ready**: Row-level security (RLS) enabled from day one
- **No Third-Party LLM Costs**: Use your own GPU infrastructure for inference

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Web App (Next.js 16)                       │   │
│  │  - Chat UI (Streaming)    - Deep Think Toggle           │   │
│  │  - Session History        - Voice / Alfred Mode         │   │
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
│  │  - Logging & Audit     - Cost Controls                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │             │              │              │
          │             │              │              │
          ▼             ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ THINKING LLM │ │ INSTANT LLM  │ │ SearXNG  │ │ Kokoro TTS   │
│ Qwen 3.5 9B  │ │ Qwen 3.5 4B  │ │ :8888    │ │ :8880        │
│ :8080        │ │ :8081        │ │ (Docker)  │ │ (Docker)     │
└──────────────┘ └──────────────┘ └──────────┘ └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Supabase                                   │   │
│  │  - Auth (Users, JWT)     - Storage (Documents)          │   │
│  │  - Postgres (Data)       - pgvector (Embeddings)        │   │
│  │  - RLS (Multi-tenant)                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Local_AI_Project/
├── .env.local                       # Shared environment config (all services)
├── Makefile                         # Development commands (make dev, make test)
├── docker-compose.searxng.yml       # SearXNG web search (Docker)
├── README.md
│
├── gateway/                         # FastAPI Backend (Python)
│   ├── app/
│   │   ├── main.py                  # Application entry point
│   │   ├── config.py                # Configuration & settings (all tiers, TTS, search)
│   │   ├── middleware/
│   │   │   └── auth.py              # JWT auth middleware
│   │   ├── models/
│   │   │   └── chat.py              # Pydantic models (ChatRequest, TTSRequest, etc.)
│   │   ├── routes/
│   │   │   ├── health.py            # Health endpoints (/health, /health/auth, /health/inference)
│   │   │   └── chat.py              # Chat, voice, TTS, streaming endpoints
│   │   └── services/
│   │       ├── inference.py          # LLM inference (instant/thinking/thinking_harder)
│   │       ├── database.py           # Supabase REST client (sessions, messages, profiles)
│   │       ├── web_search.py         # SearXNG web search integration
│   │       ├── transcription.py      # Speech-to-text (faster-whisper, tiny model)
│   │       └── speaker.py            # Voice enrollment & speaker verification (MFCC)
│   ├── tests/                        # Pytest test suite
│   ├── pyproject.toml
│   └── requirements.txt
│
├── webapp/                           # Next.js Frontend (TypeScript)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx              # Home page (auth redirect)
│   │   │   ├── layout.tsx            # Root layout (Geist Mono, dark theme)
│   │   │   ├── globals.css           # Global styles (Tailwind v4)
│   │   │   ├── chat/
│   │   │   │   ├── page.tsx          # Main chat interface (streaming, modes, voice)
│   │   │   │   ├── sidebar.tsx       # Session sidebar (list, rename, delete)
│   │   │   │   ├── MessageBubble.tsx # Message rendering (markdown, think blocks, sources)
│   │   │   │   └── VoiceSettings.tsx # Alfred voice panel (voice, speed, enrollment)
│   │   │   └── login/
│   │   │       └── page.tsx          # Login / signup page
│   │   └── lib/
│   │       ├── gateway.ts            # Gateway API client (chat, streaming, TTS, voice)
│   │       ├── supabase.ts           # Supabase client (auth)
│   │       ├── thinkParser.ts        # <think> tag parser for reasoning blocks
│   │       └── voice.ts              # Voice hooks (STT, TTS, silence detection, wake word)
│   ├── package.json
│   └── tsconfig.json
│
├── searxng/                          # SearXNG configuration
│   └── settings.yml                  # Search engines (Google, DuckDuckGo, Brave, Wikipedia)
│
├── supabase/                         # Database Configuration
│   ├── config.toml                   # Supabase local config
│   ├── migrations/
│   │   ├── 20260222000000_initial_schema.sql
│   │   ├── 20260310000000_add_reasoning_content.sql
│   │   └── 20260310100000_add_thinking_harder_mode.sql
│   └── snippets/                     # SQL reference snippets (00-07)
│
└── docs/                             # Project documentation (private, not in git)
```

## Features

### Chat & Streaming

- **SSE streaming** via `POST /chat/stream` — tokens arrive in real time
- **Non-streaming** fallback via `POST /chat` for simpler integrations
- **Markdown rendering** with syntax highlighting (react-markdown + react-syntax-highlighter)
- **Session management** — create, rename, archive, delete conversations
- **Context window** — last 20 messages per session sent to the LLM

### Three Inference Modes

| Mode | Model | Budget | Use Case |
|------|-------|--------|----------|
| **Instant** | Qwen 3.5 4B (4-bit) | No thinking | Quick answers, voice conversation |
| **Thinking** | Qwen 3.5 9B (4-bit) | 2,048 tokens | Reasoning, coding, analysis |
| **Think Harder** | Qwen 3.5 9B (4-bit) | 8,192 tokens | Complex problems, extended reasoning |

- Thinking blocks are collapsible in the UI with a "Show reasoning" toggle
- Fallback: if the thinking tier is down, requests can fall back to instant

### Voice Conversation ("Alfred" Mode)

A full voice loop with wake-word activation:

1. **Speech-to-Text**: Browser records audio → `POST /chat/transcribe` → faster-whisper (tiny model, ~75 MB)
2. **AI Response**: Transcribed text sent to chat → streamed response
3. **Text-to-Speech**: Response spoken aloud via Kokoro TTS (`POST /chat/tts`)
4. **Speaker Verification**: Optional MFCC-based voice enrollment (3 samples) — only recognized voices can use Alfred mode
5. **Wake Word**: Say "Hey Alfred" to start listening
6. **Silence Detection**: Auto-stops recording after silence

Voice profiles are stored locally at `~/.local-ai/voice-profiles/`.

### Web Search (SearXNG)

- Self-hosted SearXNG instance (Docker) — no API keys needed
- Gateway auto-detects when a question needs web results (skips greetings/trivial messages)
- Search results injected into the LLM system prompt with source URLs
- Frontend shows cited sources below the AI response
- Engines: Google, DuckDuckGo, Brave, Wikipedia

### Authentication & Security

- Supabase Auth (email/password) with JWT tokens
- Gateway validates JWT on every request (HS256)
- Row-Level Security on all database tables
- CORS restricted to allowed origins
- GPU endpoints only accessible via Tailscale VPN

## Quick Start

### Prerequisites

- **macOS/Linux** development machine
- **Docker Desktop** (for Supabase, SearXNG, Kokoro TTS)
- **Node.js 18+** (for Web App)
- **Python 3.10+** (for Gateway API)
- **ffmpeg** (for voice/speaker features: `brew install ffmpeg`)
- **Git** (for version control)
- **Supabase CLI** (`brew install supabase/tap/supabase`)

### 1. Clone the Repository

```bash
git clone git@github.com:Jimmy6929/local_AI.git
cd local_AI
```

### 2. Configure Environment

```bash
# Copy the template and fill in your Supabase keys + Tailscale IPs
cp .env.example .env.local
```

All services share a single `.env.local` at the project root. Gateway and webapp both read from `../.env.local` automatically — no per-folder env files needed. The `.env.local` must be **copied manually** between machines (it's gitignored).

### 3. Install Dependencies

```bash
make install          # Installs both gateway + webapp deps
# Or individually:
cd gateway && pip3 install -r requirements.txt
cd webapp && npm install
```

## Running the Project — Terminal-by-Terminal

You need to start **5–7 services across 2 machines**. Order matters — start from the bottom of the stack up.

### Machine 1: M2 Pro (GPU Machine) — 1–2 terminals

**Terminal 1 — Thinking LLM (required)**

```bash
mlx_vlm.server --host 0.0.0.0 --port 8080 \
  --model mlx-community/Qwen3.5-9B-4bit \
  --enable-thinking \
  --thinking-budget 2048 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>"
```

**Terminal 2 — Instant LLM (optional, for voice/instant mode)**

```bash
mlx_lm.server --host 0.0.0.0 --port 8081 --model mlx-community/Qwen3.5-4B-Instruct-4bit
```

### Machine 2: MacBook 2019 (Home Server) — 4–5 terminals

**Terminal 1 — Supabase (start first, requires Docker Desktop)**

```bash
cd ~/Documents/App-project/Local_AI_Project/supabase
supabase start
```

Wait for Supabase to finish starting before proceeding.

**Terminal 2 — Gateway API**

```bash
cd ~/Documents/App-project/Local_AI_Project/gateway
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Web App**

```bash
cd ~/Documents/App-project/Local_AI_Project/webapp
npm run dev
```

**Terminal 4 — SearXNG Web Search (optional, for web search)**

```bash
cd ~/Documents/App-project/Local_AI_Project
docker compose -f docker-compose.searxng.yml up -d
```

Verify at `http://localhost:8888`. Runs in the background (`-d`), persists across restarts.

**Terminal 5 — Kokoro TTS (optional, for voice responses)**

```bash
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

Verify at `http://localhost:8880/docs`. First pull downloads ~1 GB; subsequent starts are instant. Default voice: `bm_george` (British male). Change voice in the Voice Settings panel in the UI.

### Startup Summary

| # | Terminal | Machine | Command | Port |
|---|----------|---------|---------|------|
| 1 | Thinking LLM | M2 Pro | `mlx_vlm.server ...` | 8080 |
| 2 | Instant LLM | M2 Pro | `mlx_lm.server ...` | 8081 |
| 3 | Supabase | 2019 MacBook | `supabase start` | 54321-54323 |
| 4 | Gateway API | 2019 MacBook | `python3 -m uvicorn ...` | 8000 |
| 5 | Web App | 2019 MacBook | `npm run dev` | 3000 |
| 6 | SearXNG | 2019 MacBook | `docker compose ... up -d` | 8888 |
| 7 | Kokoro TTS | 2019 MacBook | `docker run ...` | 8880 |

> Terminals 2, 6, and 7 are optional. The core experience (chat with thinking mode) only needs terminals 1, 3, 4, and 5.

### Verify Everything Is Running

```bash
# From M2 Pro — check LLM servers
curl http://localhost:8080/health                    # Thinking LLM
curl http://localhost:8081/v1/models                  # Instant LLM

# From Home Server — check services
curl http://127.0.0.1:8000/health                    # Gateway
curl http://127.0.0.1:8000/health/inference           # Gateway → LLM connection
curl http://localhost:8888/search?q=test&format=json   # SearXNG
curl http://localhost:8880/docs                        # Kokoro TTS
open http://localhost:3000                             # Web App
```

### Quick Commands (Makefile)

```bash
make help           # Show all available commands
make install        # Install all dependencies (gateway + webapp)
make dev            # Instructions for starting all services
make dev-all        # Start all via tmux (requires tmux)
make dev-gateway    # Start Gateway API only
make dev-webapp     # Start Web App only
make dev-supabase   # Start Supabase only
make mlx-thinking   # Start MLX-VLM thinking server on GPU machine
make test           # Run all tests
make test-gateway   # Run gateway tests with pytest
make lint           # Lint gateway code
make format         # Format gateway code with black
make db-reset       # Reset database and rerun migrations
make clean          # Remove build artifacts
make stop           # Stop all services
```

### Stopping & Checking Services

Check if a service is running:

```bash
lsof -i :8000    # Gateway
lsof -i :3000    # Web App
lsof -i :54321   # Supabase
lsof -i :8888    # SearXNG
lsof -i :8880    # Kokoro TTS
```

Stop a service:

```bash
kill $(lsof -t -i :8000)                              # Stop Gateway
kill $(lsof -t -i :3000)                              # Stop Web App
supabase stop                                          # Stop Supabase
docker compose -f docker-compose.searxng.yml down      # Stop SearXNG
docker stop $(docker ps -q --filter ancestor=ghcr.io/remsky/kokoro-fastapi-cpu:latest)  # Stop Kokoro TTS
# Or just press Ctrl+C in the terminal running the service
```

> **Tip:** If you get `Address already in use` when starting a service, it's already running.
> Run `curl http://127.0.0.1:8000/health` to confirm, or kill it and restart.

## API Reference

### Health Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Basic health check |
| `/health/auth` | GET | Yes | Validates JWT and returns user info |
| `/health/inference` | GET | No | Checks instant + thinking tier status and routing config |

### Chat Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat` | POST | Yes | Send message, receive full AI response |
| `/chat/stream` | POST | Yes | Send message, receive SSE-streamed response |
| `/chat/sessions` | GET | Yes | List user's chat sessions |
| `/chat/sessions/{id}/messages` | GET | Yes | Get messages in a session |
| `/chat/sessions/{id}` | PATCH | Yes | Rename a session |
| `/chat/sessions/{id}` | DELETE | Yes | Delete a session and its messages |

### Voice Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat/transcribe` | POST | Yes | Transcribe audio to text (faster-whisper) |
| `/chat/tts` | POST | Yes | Text-to-speech via Kokoro TTS (returns MP3) |
| `/chat/voice-enroll` | POST | Yes | Enroll a voice sample for speaker verification |
| `/chat/voice-profile` | GET | Yes | Get voice enrollment status |
| `/chat/voice-profile` | DELETE | Yes | Delete voice profile |

### POST /chat (and /chat/stream) Request

```json
{
  "session_id": "uuid | null",
  "message": "string",
  "mode": "instant | thinking | thinking_harder",
  "conversation_mode": false
}
```

### POST /chat Response

```json
{
  "session_id": "uuid",
  "message": {
    "id": "uuid",
    "role": "assistant",
    "content": "string",
    "reasoning_content": "string | null",
    "mode_used": "instant | thinking | thinking_harder",
    "created_at": "datetime"
  },
  "session_title": "string | null",
  "inference": {
    "mode_used": "thinking",
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "fallback_used": false,
    "latency_ms": 1200,
    "tokens_used": 150,
    "finish_reason": "stop"
  }
}
```

### POST /chat/stream SSE Events

The streaming endpoint sends Server-Sent Events (SSE):

```
data: {"delta": {"content": "Hello"}}
data: {"delta": {"reasoning_content": "Let me think..."}}
data: {"search_done": true, "sources": [...]}
data: [DONE]
```

### POST /chat/tts Request

```json
{
  "text": "Hello, how are you?",
  "voice": "bm_george",
  "speed": 1.0
}
```

Returns `audio/mpeg` binary (MP3).

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `profiles` | User profile information (auto-created on signup via trigger) |
| `chat_sessions` | Chat conversation sessions |
| `chat_messages` | Individual messages with `reasoning_content` and `mode_used` |
| `documents` | Uploaded document metadata (future RAG) |
| `document_chunks` | Document chunks with embeddings (future RAG) |

### Entity Relationship

```
profiles (1) ─────< chat_sessions (1) ─────< chat_messages
    │
    └──────────────< documents (1) ────────< document_chunks
```

### Key Schema Details

- `chat_messages.reasoning_content` — stores the `<think>` block content separately
- `chat_messages.mode_used` — constrained to `instant`, `thinking`, `thinking_harder`
- `auth.users` trigger auto-creates a `profiles` row on signup
- `chat_messages` trigger auto-updates `chat_sessions.updated_at`
- pgvector extension with IVFFlat index for future RAG similarity search

### Row-Level Security (RLS)

All tables have RLS enabled. Users can only access their own data:

- `profiles`: Users can view/update only their own profile
- `chat_sessions`: Users can CRUD only their own sessions
- `chat_messages`: Users can CRUD only messages in their sessions
- `documents`: Users can CRUD only their own documents

### Migrations

| Migration | Description |
|-----------|-------------|
| `20260222000000_initial_schema.sql` | Initial schema: profiles, sessions, messages, documents, chunks, RLS, triggers, storage |
| `20260310000000_add_reasoning_content.sql` | Adds `reasoning_content` column to `chat_messages` |
| `20260310100000_add_thinking_harder_mode.sql` | Adds `thinking_harder` to `mode_used` check constraint |

## Two-Machine Setup

This project runs across **two machines** connected via Tailscale VPN:

| Machine | Role | Tailscale IP |
|---------|------|-------------|
| **MacBook Pro 2019 (i7)** | Home Server: Gateway, Webapp, Supabase, SearXNG, Kokoro TTS | `100.99.189.104` |
| **MacBook Pro M2 Pro (16GB)** | GPU Machine: MLX inference servers (Thinking + Instant) | `100.104.193.59` |

```
Browser → Webapp (:3000) → Gateway (:8000) ──┬──→ MLX Thinking (:8080)  ← M2 Pro
              │                                ├──→ MLX Instant  (:8081)  ← M2 Pro
              │                                ├──→ SearXNG      (:8888)  ← Home Server
              │                                └──→ Kokoro TTS   (:8880)  ← Home Server
              └──→ Supabase  (:54321)                               ← Home Server
```

## Inference Modes

### Model Configuration

| Tier | Model | Server | Port | API Path | Thinking | Use Case |
|------|-------|--------|------|----------|----------|----------|
| **Instant** | `Qwen3.5-4B-Instruct-4bit` | `mlx_lm.server` | 8081 | `/v1/chat/completions` | No | Voice, quick answers |
| **Thinking** | `Qwen3.5-9B-4bit` | `mlx_vlm.server` | 8080 | `/chat/completions` | Yes (2K budget) | Reasoning, coding |
| **Think Harder** | `Qwen3.5-9B-4bit` | `mlx_vlm.server` | 8080 | `/chat/completions` | Yes (8K budget) | Complex problems |

> **Why mlx_vlm for the 9B?** Qwen 3.5 is a VLM (Vision-Language Model) that includes an image/video encoder.
> It requires `mlx-vlm` (which supports vision models). It works great for text chat and
> *also* has vision capabilities for future use.

> **Why mlx_lm for the 4B?** The 4B Instruct model is text-only, so it uses the lighter `mlx_lm.server`.

### Cost Controls

- `THINKING_DAILY_REQUEST_LIMIT` — max thinking requests per user per day (default: 100)
- `THINKING_MAX_CONCURRENT` — max parallel thinking requests (default: 2)
- `ROUTING_THINKING_FALLBACK_TO_INSTANT` — fall back to instant if thinking tier is down

### MLX Setup (GPU Machine — M2 Pro)

#### One-Time Installation

```bash
pip install -U "mlx-vlm[torch]"    # Qwen 3.5 9B — needs PyTorch for vision processor
pip install -U mlx-lm               # Qwen 3.5 4B — text-only, lighter
```

> **Thinking mode requires mlx-vlm >= March 5 2026** (PR #789). If you installed
> before that date, run `pip install -U "mlx-vlm[torch]"` again to pick up
> `--enable-thinking` and `--thinking-budget` server flags.

#### Starting the LLM Servers (Every Session)

```bash
# Terminal 1 — Thinking tier (Qwen 3.5 9B)
mlx_vlm.server --host 0.0.0.0 --port 8080 \
  --model mlx-community/Qwen3.5-9B-4bit \
  --enable-thinking \
  --thinking-budget 2048 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>"

# Terminal 2 — Instant tier (Qwen 3.5 4B) [optional]
mlx_lm.server --host 0.0.0.0 --port 8081 --model mlx-community/Qwen3.5-4B-Instruct-4bit
```

| Flag | Purpose |
|------|---------|
| `--model` | Explicitly load the model instead of relying on cache |
| `--enable-thinking` | Passes `enable_thinking=True` into Qwen's chat template |
| `--thinking-budget 2048` | Server-side hard cap — forcibly closes `</think>` after 2048 tokens |
| `--thinking-start-token` / `--thinking-end-token` | Tells the server which tokens delimit the reasoning block |

> The gateway also sends per-request `thinking_budget` in the API body (2048 for Think, 8192 for Think Harder).
> The server-side `--thinking-budget` acts as a safety net default.

## Security

### Authentication Flow

1. User logs in via Supabase Auth (Web App)
2. Supabase issues JWT token
3. Web App includes JWT in all Gateway requests
4. Gateway validates JWT on every request
5. User ID extracted from JWT for database queries

### Security Features

- **JWT Validation**: All API endpoints require valid JWT
- **Row-Level Security**: Database enforces data isolation
- **Private Inference**: GPU endpoints not publicly accessible
- **Speaker Verification**: Voice mode can require voice enrollment
- **CORS Configuration**: Restricted to allowed origins

## Environment Variables

All environment variables live in a **single `.env.local` at the project root**. Both the gateway and webapp read from this file automatically.

```bash
cp .env.example .env.local
```

### Gateway Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `SUPABASE_URL` | Supabase API URL | `http://127.0.0.1:54321` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | — |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | — |
| `JWT_SECRET` | JWT signing secret | — |
| `INFERENCE_INSTANT_URL` | Instant tier endpoint URL | — |
| `INFERENCE_INSTANT_MODEL` | Instant tier model name | — |
| `INFERENCE_INSTANT_MAX_TOKENS` | Max tokens for instant mode | `2048` |
| `INFERENCE_INSTANT_TEMPERATURE` | Temperature for instant mode | `0.7` |
| `INFERENCE_INSTANT_ENABLE_THINKING` | Enable CoT for instant | `false` |
| `INFERENCE_THINKING_URL` | Thinking tier endpoint URL | — |
| `INFERENCE_THINKING_MODEL` | Thinking tier model name | — |
| `INFERENCE_THINKING_MAX_TOKENS` | Max tokens for thinking mode | `24576` |
| `INFERENCE_THINKING_TEMPERATURE` | Temperature for thinking mode | `0.6` |
| `INFERENCE_THINKING_ENABLE_THINKING` | Enable CoT for thinking | `true` |
| `INFERENCE_THINKING_BUDGET` | Thinking token budget (Think) | `2048` |
| `INFERENCE_THINKING_HARDER_MAX_TOKENS` | Max tokens for Think Harder | `28672` |
| `INFERENCE_THINKING_HARDER_BUDGET` | Thinking token budget (Think Harder) | `8192` |
| `KOKORO_TTS_URL` | Kokoro TTS endpoint | `http://localhost:8880` |
| `KOKORO_TTS_DEFAULT_VOICE` | Default TTS voice | `bm_george` |
| `SEARXNG_URL` | SearXNG search endpoint | `http://localhost:8888` |
| `WEB_SEARCH_ENABLED` | Enable web search | `true` |
| `WEB_SEARCH_TIMEOUT` | Search timeout (seconds) | `5.0` |
| `WEB_SEARCH_MAX_RESULTS` | Max search results | `5` |
| `ROUTING_DEFAULT_MODE` | Default inference mode | `thinking` |
| `ROUTING_THINKING_FALLBACK_TO_INSTANT` | Fall back if thinking is down | `true` |
| `THINKING_DAILY_REQUEST_LIMIT` | Max thinking requests/user/day | `100` |
| `THINKING_MAX_CONCURRENT` | Max parallel thinking requests | `2` |

### Web App Variables

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase API URL (Tailscale IP for cross-machine access) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anonymous key |
| `NEXT_PUBLIC_GATEWAY_URL` | Gateway API URL (Tailscale IP for cross-machine access) |

## Complete Command Reference

### M2 Pro (GPU Machine)

```bash
# ── One-Time Setup ──────────────────────────────────
pip install -U "mlx-vlm[torch]"          # Qwen 3.5 9B — needs PyTorch
pip install -U mlx-lm                     # Qwen 3.5 4B — text-only

# ── Start LLM Servers (every session) ──────────────
# Terminal 1 — Thinking LLM
mlx_vlm.server --host 0.0.0.0 --port 8080 \
  --model mlx-community/Qwen3.5-9B-4bit \
  --enable-thinking \
  --thinking-budget 2048 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>"

# Terminal 2 — Instant LLM (optional)
mlx_lm.server --host 0.0.0.0 --port 8081 --model mlx-community/Qwen3.5-4B-Instruct-4bit

# ── Health Checks ──────────────────────────────────
curl http://localhost:8080/health         # Thinking LLM status
curl http://localhost:8080/models         # List loaded VLM models
curl http://localhost:8081/v1/models      # List loaded instant models
```

### MacBook 2019 (Home Server)

```bash
# ── One-Time Setup ──────────────────────────────────
cd ~/Documents/App-project/Local_AI_Project
cd gateway && pip3 install -r requirements.txt       # Gateway Python deps
cd ../webapp && npm install                           # Webapp Node deps
brew install supabase/tap/supabase                   # Supabase CLI
brew install ffmpeg                                   # Required for voice features

# ── Start Services (every session, in order) ───────
# 1. Make sure Docker Desktop is open first!
cd ~/Documents/App-project/Local_AI_Project/supabase && supabase start

# 2. Gateway (new terminal)
cd ~/Documents/App-project/Local_AI_Project/gateway
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Webapp (new terminal)
cd ~/Documents/App-project/Local_AI_Project/webapp && npm run dev

# 4. SearXNG (new terminal, optional — runs in background)
cd ~/Documents/App-project/Local_AI_Project
docker compose -f docker-compose.searxng.yml up -d

# 5. Kokoro TTS (new terminal, optional)
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest

# ── Stop Services ──────────────────────────────────
supabase stop                                         # Stop Supabase + Docker containers
kill $(lsof -t -i :8000)                              # Stop Gateway
kill $(lsof -t -i :3000)                              # Stop Webapp
docker compose -f docker-compose.searxng.yml down     # Stop SearXNG
# Or just Ctrl+C in each terminal

# ── Health Checks ──────────────────────────────────
curl http://127.0.0.1:8000/health                    # Gateway
curl http://127.0.0.1:8000/health/inference           # Gateway → LLM connection
curl http://127.0.0.1:54321/auth/v1/health            # Supabase Auth
curl http://localhost:8888/search?q=test&format=json   # SearXNG
curl http://localhost:8880/docs                        # Kokoro TTS
supabase status                                       # All Supabase info + keys

# ── Database ───────────────────────────────────────
supabase db reset                         # Reset DB and rerun migrations
supabase migration new <name>             # Create new migration
supabase migration list                   # List migrations

# ── Testing & Linting ─────────────────────────────
cd gateway && python3 -m pytest tests/ -v                    # Run tests
cd gateway && python3 -m pytest tests/ --cov=app             # Tests + coverage
cd webapp && npx tsc --noEmit                                # TypeScript check
cd webapp && npm run lint                                    # Lint webapp

# ── Makefile Shortcuts (from project root) ─────────
make help                # Show all commands
make install             # Install gateway + webapp deps
make dev-gateway         # Start gateway
make dev-webapp          # Start webapp
make dev-supabase        # Start supabase
make dev-all             # Start all via tmux
make mlx-thinking        # Start thinking LLM on GPU machine
make test                # Run all tests
make lint                # Lint gateway
make format              # Format gateway code
make db-reset            # Reset database
make clean               # Remove caches
```

### Syncing Files Between Machines

```bash
# .env.local is gitignored — must be copied manually
# After editing .env.local on one machine, copy to the other.
# Everything else syncs via git:
git add -A && git commit -m "description" && git push   # On machine A
git pull                                                  # On machine B
```

### Troubleshooting

```bash
# "uvicorn: command not found" on 2019 MacBook
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # Use python3 -m prefix

# "pip: command not found" on 2019 MacBook
pip3 install -r requirements.txt          # Use pip3 instead

# Qwen3.5-9B fails with "Torchvision not found"
pip install -U "mlx-vlm[torch]"           # Install PyTorch + Torchvision

# Thinking gets stuck in loops / socket.send() errors
pip install -U "mlx-vlm[torch]"           # Upgrade — needs PR #789 (Mar 5 2026+)
# Then restart the server WITH --enable-thinking and --thinking-budget flags

# "Address already in use" on any port
lsof -i :<port>                           # Check what's using the port
kill $(lsof -t -i :<port>)                # Kill it

# Gateway shows old config after .env.local change
# Restart the gateway (Ctrl+C, then start again)

# Supabase 401 errors
supabase status                           # Check keys match .env.local

# SearXNG not returning results
docker compose -f docker-compose.searxng.yml logs   # Check logs
docker compose -f docker-compose.searxng.yml restart # Restart

# Kokoro TTS not working
curl http://localhost:8880/docs            # Check if running
docker logs $(docker ps -q --filter ancestor=ghcr.io/remsky/kokoro-fastapi-cpu:latest)

# Voice transcription fails
brew install ffmpeg                        # Required for audio conversion
# Whisper tiny model auto-downloads on first use (~75 MB)
```

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Next.js (App Router) | 16.1.6 |
| **Frontend** | React | 19.2.3 |
| **Frontend** | Tailwind CSS | v4 |
| **Frontend** | TypeScript | 5.x |
| **Backend** | FastAPI | 0.115.0 |
| **Backend** | Uvicorn | 0.30.0 |
| **Backend** | Pydantic | 2.9.0 |
| **Auth & DB** | Supabase (Auth, Postgres, Storage) | — |
| **Inference** | MLX / mlx-vlm / mlx-lm | Latest |
| **Thinking LLM** | Qwen 3.5 9B (4-bit) | — |
| **Instant LLM** | Qwen 3.5 4B Instruct (4-bit) | — |
| **STT** | faster-whisper (tiny model) | 1.1.1 |
| **TTS** | Kokoro FastAPI (Docker) | Latest |
| **Web Search** | SearXNG (Docker) | Latest |
| **Speaker ID** | MFCC embeddings (numpy) | — |
| **Networking** | Tailscale VPN | — |
| **Markdown** | react-markdown + remark-gfm | — |

## Roadmap

### Phase 1: Chat MVP ✅
- [x] Local Supabase setup
- [x] Database schema with RLS
- [x] Gateway API (FastAPI)
- [x] Chat endpoints
- [x] Web App (Next.js)
- [x] Connect to GPU inference

### Phase 2: Thinking Inference ✅
- [x] Thinking Mode (Qwen 3.5 9B via mlx_vlm, enable_thinking=True)
- [x] Think Harder mode (8K thinking budget)
- [x] Mode toggle in UI
- [x] Collapsible reasoning blocks in UI
- [x] SSE streaming with reasoning_content support
- [x] Instant tier (Qwen 3.5 4B via mlx_lm)

### Phase 3: Voice & Search ✅ (Current)
- [x] Speech-to-text (faster-whisper)
- [x] Text-to-speech (Kokoro TTS via Docker)
- [x] Voice conversation mode ("Alfred")
- [x] Wake word detection ("Hey Alfred")
- [x] Silence detection (auto-stop recording)
- [x] Speaker verification (MFCC voice enrollment)
- [x] Web search (SearXNG, self-hosted)
- [x] Search results cited in responses with source URLs

### Phase 4: RAG (Document Memory)
- [ ] File upload functionality
- [ ] Document processing pipeline
- [ ] Embedding generation
- [ ] Context retrieval

### Phase 5: Tools Framework
- [ ] Custom tool support
- [ ] Note saving
- [ ] Calendar integration

### Phase 6: Production Launch
- [ ] Hosted Supabase migration
- [ ] Public web app deployment
- [ ] Rate limiting
- [ ] Monitoring & alerting

## Testing

### Gateway API Tests

```bash
cd gateway
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chat_full.py -v

# Or use Makefile (from project root)
make test-gateway
```

### Web App Type Check

```bash
cd webapp

# TypeScript type checking
npx tsc --noEmit

# Lint
npm run lint
```

## Documentation

Detailed documentation is maintained locally in the `/docs` folder (not included in git for privacy).

| Section | Description |
|---------|-----------|
| `00_README/` | Project overview, glossary, getting started |
| `01_PRODUCT/` | MVP scope, vision, success metrics |
| `02_ARCHITECTURE/` | System design, data flow, environments |
| `03_INFRA/` | Deployment, GPU setup, cost controls |
| `04_DATABASE_SUPABASE/` | Schema, migrations, RLS policies |
| `05_GATEWAY_API/` | API contracts, routing, tools |
| `06_WEB_APP/` | UI components, auth UX, chat UX |
| `07_SECURITY/` | Threat model, secrets, hardening |
| `08_ROADMAP/` | Development phases |
| `09_DECISIONS/` | Architecture Decision Records (ADRs) |

Gateway-specific docs (GPU setup guides) are in `gateway/docs/`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

All Rights Reserved © 2026

This is a private project. Unauthorized copying, modification, distribution, or use of this software is strictly prohibited.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — MLX server for text LLMs
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — MLX server for vision-language models
- [Qwen](https://github.com/QwenLM/Qwen3) — Alibaba's open-source LLM family
- [Supabase](https://supabase.com/) — Backend as a Service
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Next.js](https://nextjs.org/) — React framework for production
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first CSS framework
- [Tailscale](https://tailscale.com/) — Zero-config VPN for the two-machine setup
- [SearXNG](https://github.com/searxng/searxng) — Privacy-respecting metasearch engine
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI) — Fast local text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Fast Whisper inference with CTranslate2

---

Built for privacy-conscious AI enthusiasts.
