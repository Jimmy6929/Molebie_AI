# Local AI Assistant

A private, self-hosted AI assistant system with multi-user support, three-tier inference, voice conversation, web search, RAG document memory, image understanding, and full data ownership.

## Overview

Local AI Assistant is a production-ready chat application that provides a ChatGPT-like experience while running entirely on your own infrastructure. It features:

- **Three Inference Modes**: Instant (fast), Thinking (chain-of-thought), and Think Harder (extended reasoning)
- **SSE Streaming**: Real-time token-by-token response streaming with thinking block support
- **Image Understanding (Vision)**: Attach images via file picker, paste, or drag-and-drop — the AI analyzes them using Qwen 3.5's built-in vision encoder
- **RAG Document Memory ("Brain")**: Upload documents (PDF, DOCX, TXT, MD) for persistent knowledge — hybrid vector + BM25 search, cross-encoder reranking, contextual retrieval
- **Voice Conversation ("Chat")**: Wake-word activation, speech-to-text (Whisper), text-to-speech (Kokoro), and speaker verification
- **Web Search**: Self-hosted SearXNG integration with LLM-powered intent classification — the AI decides when to search, fetches full-page content, and cites sources
- **Full Data Ownership**: All conversations, documents, and images stored in your Supabase instance
- **Multi-User Ready**: Row-level security (RLS) enabled from day one
- **No Third-Party LLM Costs**: Use your own GPU infrastructure for inference

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Web App (Next.js 16)                       │   │
│  │  - Chat UI (Streaming)    - Deep Think Toggle           │   │
│  │  - Session History        - Voice / Chat Mode         │   │
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
│  │  - Auth (Users, JWT)     - Storage (Documents + Images)  │   │
│  │  - Postgres (Data)       - pgvector (RAG Embeddings)    │   │
│  │  - RLS (Multi-tenant)    - BM25 Full-Text Search        │   │
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
│   │   │   ├── chat.py              # Chat, voice, TTS, vision, streaming endpoints
│   │   │   └── documents.py         # Document upload, RAG management, session attachments
│   │   └── services/
│   │       ├── inference.py          # LLM inference (instant/thinking/thinking_harder, multimodal)
│   │       ├── database.py           # Supabase REST client (sessions, messages, images, profiles)
│   │       ├── web_search.py         # SearXNG web search with LLM intent classification
│   │       ├── rag.py                # RAG pipeline (hybrid search, RRF fusion)
│   │       ├── reranker.py           # Cross-encoder reranking (ms-marco-MiniLM)
│   │       ├── context_generator.py  # Contextual retrieval (LLM chunk prefixes)
│   │       ├── embedding.py          # Embedding service (sentence-transformers)
│   │       ├── document_processor.py # Document parsing (PDF, DOCX, TXT, MD) + chunking
│   │       ├── rag_eval.py           # RAG metrics (hit rate, MRR, latency)
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
│   │   │   │   └── VoiceSettings.tsx # Chat voice panel (voice, speed, enrollment)
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

### Voice Conversation ("Chat" Mode)

A full voice loop with wake-word activation:

1. **Speech-to-Text**: Browser records audio → `POST /chat/transcribe` → faster-whisper (tiny model, ~75 MB)
2. **AI Response**: Transcribed text sent to chat → streamed response
3. **Text-to-Speech**: Response spoken aloud via Kokoro TTS (`POST /chat/tts`)
4. **Speaker Verification**: Optional MFCC-based voice enrollment (3 samples) — only recognized voices can use Chat mode
5. **Wake Word**: Say "Hey Chat" to start listening
6. **Silence Detection**: Auto-stops recording after silence

Voice profiles are stored locally at `~/.local-ai/voice-profiles/`.

### Image Understanding (Vision)

Attach an image to any message — the AI analyzes it using Qwen 3.5's built-in vision encoder.

- **Three input methods**: File picker button, clipboard paste (Ctrl+V), drag-and-drop
- **Auto-compression**: Images resized to max 1024px before sending (prevents OOM on 16GB)
- **Auto-routing**: Image messages force the thinking tier (vision-capable model)
- **Storage**: Images stored in Supabase Storage (`chat-images` bucket), metadata in `message_images` table
- **History**: Images display inline in chat history; older images shown as `[Image was attached]` in model context
- **Supported types**: JPEG, PNG, GIF, WebP (max 5MB)

### RAG Document Memory ("Brain")

Upload documents for persistent knowledge retrieval across all conversations.

- **Document upload**: PDF, DOCX, TXT, MD (up to 50MB) via `POST /documents/upload`
- **Processing pipeline**: Extract text → chunk with overlap (1024 chars, 128 overlap) → generate embeddings
- **Hybrid search**: 70% vector similarity (pgvector cosine) + 30% BM25 full-text (PostgreSQL tsvector) with RRF fusion
- **Cross-encoder reranking**: ms-marco-MiniLM-L6-v2 reranks top-20 → top-5 for precision
- **Contextual retrieval**: LLM generates context prefixes per chunk during upload for better embedding quality
- **Embedding model**: Orange/orange-nomic-v1.5-1536 (1536 dimensions)
- **Session attachments**: "Attach to chat" uploads inject full document text into the current session only (no chunking/embedding)
- **Quality metrics**: Hit rate, MRR, per-query latency logging
- **Evidence summary**: Model receives structured metadata about retrieval quality for confidence calibration

### Web Search (SearXNG)

- Self-hosted SearXNG instance (Docker) — no API keys needed
- **Three-tier intent classification**:
  - Auto-skip: greetings, code questions, creative writing (~20 patterns)
  - Auto-search: temporal queries, news, weather, stock prices (~15 patterns)
  - LLM-classify: ambiguous queries decided by instant-tier LLM (3-token budget)
- **Full-page content fetching**: Top 3 results fetched via trafilatura (2000 chars each)
- **Source trust classification**: Official (.gov, docs) > Reference > Forum > News > Web
- **Duplicate detection**: Jaccard similarity filtering (60% threshold)
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

- **Apple Silicon Mac** (M1/M2/M3/M4, 16GB+ RAM recommended)
- **Docker Desktop** (for Supabase, SearXNG, Kokoro TTS)
- **Node.js 18+** (`brew install node`)
- **Python 3.10+** (comes with macOS or `brew install python`)
- **Supabase CLI** (`brew install supabase/tap/supabase`)
- **ffmpeg** (for voice features: `brew install ffmpeg`)

### 1. Clone and Setup

```bash
git clone https://github.com/Jimmy6929/local_AI.git
cd local_AI
bash setup.sh
```

`setup.sh` checks prerequisites, installs dependencies, starts Supabase, and generates `.env.local` with the correct keys. It will ask if you're running on a single machine or two machines.

### 2. Start MLX Inference

```bash
# Terminal 1 — Thinking LLM (required)
make mlx-thinking

# Terminal 2 — Instant LLM (optional, for voice/fast mode)
make mlx-instant
```

First run downloads the models (~5GB for 9B, ~2.5GB for 4B). Subsequent starts are instant.

> **Note:** `make mlx-thinking` / `make mlx-instant` use `scripts/mlx_server.py`, a wrapper that suppresses asyncio socket warnings when users stop generation mid-stream.

### 3. Start Services

```bash
# Terminal 3 — Gateway + Webapp (requires tmux)
make dev-all

# Or start individually:
# Terminal 3: make dev-gateway
# Terminal 4: make dev-webapp
```

### 4. Start Optional Services

```bash
# Web search + Text-to-speech (background Docker containers)
docker compose up -d
```

### 5. Open the App

Visit **http://localhost:3000**, create an account, and start chatting.

## Services

| # | Service | Command | Port | Required? |
|---|---------|---------|------|-----------|
| 1 | Thinking LLM | `make mlx-thinking` | 8080 | Yes |
| 2 | Instant LLM | `make mlx-instant` | 8081 | Optional (voice/fast mode) |
| 3 | Supabase | `make dev-supabase` | 54321 | Yes (auto-started by setup.sh) |
| 4 | Gateway API | `make dev-gateway` | 8000 | Yes |
| 5 | Web App | `make dev-webapp` | 3000 | Yes |
| 6 | SearXNG + Kokoro | `docker compose up -d` | 8888, 8880 | Optional (search + TTS) |

> The core experience (chat with thinking mode) only needs services 1, 3, 4, and 5.

### Verify Everything Is Running

```bash
curl http://localhost:8080/health                      # Thinking LLM
curl http://localhost:8081/v1/models                    # Instant LLM
curl http://127.0.0.1:8000/health                      # Gateway
curl http://127.0.0.1:8000/health/inference             # Gateway → LLM connection
curl http://localhost:8888/search?q=test&format=json    # SearXNG
curl http://localhost:8880/docs                         # Kokoro TTS
open http://localhost:3000                              # Web App
```

### Stopping Services

```bash
make stop                     # Stop Supabase + Gateway
docker compose down           # Stop SearXNG + Kokoro TTS
# Or just press Ctrl+C in each terminal
```

### Quick Commands (Makefile)

```bash
make setup          # First-time setup (prereqs, deps, Supabase keys)
make help           # Show all available commands
make install        # Install all dependencies (gateway + webapp)
make dev-all        # Start gateway + webapp via tmux
make dev-gateway    # Start Gateway API only
make dev-webapp     # Start Web App only
make dev-supabase   # Start Supabase only
make mlx-thinking   # Start thinking LLM server
make mlx-instant    # Start instant LLM server
make test           # Run all tests
make lint           # Lint gateway code
make format         # Format gateway code
make db-reset       # Reset database and rerun migrations
make clean          # Remove build artifacts
make stop           # Stop all services
```

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
| `/chat` | POST | Yes | Send message (with optional image), receive full AI response |
| `/chat/stream` | POST | Yes | Send message (with optional image), receive SSE-streamed response |
| `/chat/sessions` | GET | Yes | List user's chat sessions |
| `/chat/sessions/create` | POST | Yes | Create an empty session (for attaching docs before first message) |
| `/chat/sessions/{id}/messages` | GET | Yes | Get messages in a session (includes `image_id` for messages with images) |
| `/chat/sessions/{id}` | PATCH | Yes | Rename a session |
| `/chat/sessions/{id}` | DELETE | Yes | Delete a session and its messages |
| `/chat/images/{id}` | GET | Yes | Serve a chat image from Supabase Storage |
| `/chat/sessions/{id}/images` | GET | Yes | Get image metadata for all messages in a session |

### Document Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/documents/upload` | POST | Yes | Upload a document for RAG (PDF, DOCX, TXT, MD) |
| `/documents` | GET | Yes | List user's uploaded documents |
| `/documents/{id}` | DELETE | Yes | Delete a document and its chunks |
| `/documents/sessions/{id}/attach` | POST | Yes | Attach a document to a chat session |
| `/documents/sessions/{id}/attachments` | GET | Yes | List attachments for a session |
| `/documents/sessions/{id}/attachments/{att_id}` | DELETE | Yes | Remove a session attachment |

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
  "conversation_mode": false,
  "image": "data:image/jpeg;base64,... | null"
}
```

> When `image` is provided, the request is automatically routed to the thinking tier (vision-capable model) regardless of `mode`.

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
| `message_images` | Image attachments for chat messages (metadata; bytes in Supabase Storage) |
| `documents` | Uploaded document metadata for RAG |
| `document_chunks` | Document chunks with pgvector embeddings + tsvector for BM25 |
| `session_documents` | Per-session document attachments (full text, no embedding) |

### Entity Relationship

```
profiles (1) ─────< chat_sessions (1) ─────< chat_messages (1) ────< message_images
    │                     │
    │                     └──────────────< session_documents
    └──────────────< documents (1) ────────< document_chunks
```

### Storage Buckets

| Bucket | Purpose |
|--------|---------|
| `documents` | Uploaded RAG documents (PDF, DOCX, TXT, MD) |
| `chat-images` | Image attachments from chat messages (JPEG, PNG, GIF, WebP) |

### Key Schema Details

- `chat_messages.reasoning_content` — stores the `<think>` block content separately
- `chat_messages.mode_used` — constrained to `instant`, `thinking`, `thinking_harder`
- `message_images.storage_path` — path in Supabase Storage `chat-images` bucket
- `document_chunks.embedding` — pgvector 1536-dimensional vectors (HNSW index)
- `document_chunks.content` — tsvector GIN index for BM25 full-text search
- `auth.users` trigger auto-creates a `profiles` row on signup
- `chat_messages` trigger auto-updates `chat_sessions.updated_at`

### Row-Level Security (RLS)

All tables have RLS enabled. Users can only access their own data:

- `profiles`: Users can view/update only their own profile
- `chat_sessions`: Users can CRUD only their own sessions
- `chat_messages`: Users can CRUD only messages in their sessions
- `message_images`: Users can view/insert/delete only their own images
- `documents`: Users can CRUD only their own documents
- `document_chunks`: Users can view/insert/delete only their own chunks
- `session_documents`: Users can CRUD only their own session attachments
- `storage.objects`: Users can only access files in their own folder (`user_id/...`)

### Migrations

| Migration | Description |
|-----------|-------------|
| `20260222000000_initial_schema.sql` | Initial schema: profiles, sessions, messages, documents, chunks, RLS, triggers, storage |
| `20260310000000_add_reasoning_content.sql` | Adds `reasoning_content` column to `chat_messages` |
| `20260310100000_add_thinking_harder_mode.sql` | Adds `thinking_harder` to `mode_used` check constraint |
| `20260312000000_embedding_384_dim.sql` | Initial embedding dimension setup |
| `20260319192337_fix_embedding_index.sql` | Fix pgvector embedding index |
| `20260319192338_session_documents.sql` | Session document attachments table |
| `20260320000000_embedding_1536_dim.sql` | Upgrade to 1536-dimensional embeddings |
| `20260321000000_hnsw_tuning.sql` | HNSW index tuning for better recall |
| `20260321100000_hybrid_search.sql` | Add tsvector column + GIN index for BM25 search |
| `20260321200000_contextual_retrieval.sql` | Contextual retrieval metadata column |
| `20260321300000_rag_metrics.sql` | RAG evaluation metrics support |
| `20260322000000_message_images.sql` | Image attachments table + `chat-images` storage bucket |

## Advanced: Multi-Machine Setup

You can split the system across **two machines** — a GPU machine for inference and a server machine for everything else. Connect them via [Tailscale](https://tailscale.com/) (recommended) or a local network.

| Machine | Role |
|---------|------|
| **Server** (any Mac/Linux) | Gateway, Webapp, Supabase, SearXNG, Kokoro TTS |
| **GPU** (Apple Silicon Mac) | MLX inference servers (Thinking + Instant) |

```
Browser → Webapp (:3000) → Gateway (:8000) ──┬──→ MLX Thinking (:8080)  ← GPU machine
              │                                ├──→ MLX Instant  (:8081)  ← GPU machine
              │                                ├──→ SearXNG      (:8888)  ← Server
              │                                └──→ Kokoro TTS   (:8880)  ← Server
              └──→ Supabase  (:54321)                               ← Server
```

Run `bash setup.sh` and choose **"Two machines"** — it will ask for the IPs and configure everything automatically. Or manually set these in `.env.local`:

```bash
# GPU machine IP (where MLX inference runs)
INFERENCE_THINKING_URL=http://<GPU_IP>:8080
INFERENCE_INSTANT_URL=http://<GPU_IP>:8081

# Server machine IP (where the browser connects)
NEXT_PUBLIC_SUPABASE_URL=http://<SERVER_IP>:54321
NEXT_PUBLIC_GATEWAY_URL=http://<SERVER_IP>:8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://<SERVER_IP>:3000
```

Also add your server IP to `supabase/config.toml`:
```toml
additional_redirect_urls = ["http://127.0.0.1:3000", "http://localhost:3000", "http://<SERVER_IP>:3000"]
```

### Syncing Between Machines

```bash
# .env.local is gitignored — copy it manually to the GPU machine
# Everything else syncs via git
```

## Inference Modes

### Model Configuration

| Tier | Model | Server | Port | API Path | Thinking | Use Case |
|------|-------|--------|------|----------|----------|----------|
| **Instant** | `Qwen3.5-4B-Instruct-4bit` | `mlx_lm.server` | 8081 | `/v1/chat/completions` | No | Voice, quick answers |
| **Thinking** | `Qwen3.5-9B-4bit` | `mlx_vlm.server` | 8080 | `/chat/completions` | Yes (2K budget) | Reasoning, coding |
| **Think Harder** | `Qwen3.5-9B-4bit` | `mlx_vlm.server` | 8080 | `/chat/completions` | Yes (8K budget) | Complex problems |

> **Why mlx_vlm for the 9B?** Qwen 3.5 is a VLM (Vision-Language Model) that includes an image/video encoder.
> It requires `mlx-vlm` (which supports vision models). It handles text chat, chain-of-thought reasoning,
> *and* image understanding — all in one model. Images are auto-compressed to max 1024px to stay within 16GB memory.

> **Why mlx_lm for the 4B?** The 4B Instruct model is text-only, so it uses the lighter `mlx_lm.server`.

### Cost Controls

- `THINKING_DAILY_REQUEST_LIMIT` — max thinking requests per user per day (default: 100)
- `THINKING_MAX_CONCURRENT` — max parallel thinking requests (default: 2)
- `ROUTING_THINKING_FALLBACK_TO_INSTANT` — fall back to instant if thinking tier is down

### MLX Setup (Apple Silicon)

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
python3 scripts/mlx_server.py --host 0.0.0.0 --port 8080 \
  --model mlx-community/Qwen3.5-9B-4bit \
  --enable-thinking \
  --thinking-budget 2048 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>"

# Terminal 2 — Instant tier (Qwen 3.5 4B) [optional]
python3 scripts/mlx_server.py --host 0.0.0.0 --port 8081 --model mlx-community/Qwen3.5-4B-Instruct-4bit
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
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase API URL (localhost or server IP for multi-machine) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anonymous key |
| `NEXT_PUBLIC_GATEWAY_URL` | Gateway API URL (localhost or server IP for multi-machine) |
| `CORS_ORIGINS` | Comma-separated allowed origins for CORS |

## Complete Command Reference

```bash
# ── First-Time Setup ────────────────────────────────
bash setup.sh                             # Interactive setup (prereqs, deps, Supabase keys)

# ── MLX One-Time Install ────────────────────────────
pip install -U "mlx-vlm[torch]"           # Qwen 3.5 9B — needs PyTorch for vision
pip install -U mlx-lm                     # Qwen 3.5 4B — text-only, lighter

# ── Start Services (every session) ──────────────────
make mlx-thinking                         # Thinking LLM (:8080)
make mlx-instant                          # Instant LLM (:8081, optional)
make dev-supabase                         # Supabase (:54321)
make dev-gateway                          # Gateway API (:8000)
make dev-webapp                           # Web App (:3000)
docker compose up -d                      # SearXNG + Kokoro TTS (optional)

# Or use tmux for gateway + webapp:
make dev-all

# ── Stop Services ───────────────────────────────────
make stop                                 # Stop Supabase + Gateway
docker compose down                       # Stop SearXNG + Kokoro TTS
# Or just Ctrl+C in each terminal

# ── Health Checks ───────────────────────────────────
curl http://localhost:8080/health          # Thinking LLM
curl http://localhost:8081/v1/models       # Instant LLM
curl http://127.0.0.1:8000/health          # Gateway
curl http://127.0.0.1:8000/health/inference # Gateway → LLM
curl http://localhost:8888/search?q=test&format=json  # SearXNG
curl http://localhost:8880/docs            # Kokoro TTS

# ── Database ────────────────────────────────────────
supabase db reset                         # Reset DB and rerun migrations
supabase migration new <name>             # Create new migration
supabase status                           # Show Supabase info + keys

# ── Testing & Linting ──────────────────────────────
make test                                 # Run all tests
make lint                                 # Lint gateway code
make format                               # Format gateway code
cd webapp && npx tsc --noEmit             # TypeScript check
cd webapp && npm run lint                 # Lint webapp
```

### Troubleshooting

```bash
# "uvicorn: command not found"
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # Use python3 -m prefix

# "pip: command not found"
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
docker compose logs searxng                          # Check logs
docker compose restart searxng                       # Restart

# Kokoro TTS not working
curl http://localhost:8880/docs            # Check if running
docker compose logs kokoro-tts                       # Check Kokoro logs

# Voice transcription fails
brew install ffmpeg                        # Required for audio conversion
# Whisper tiny model auto-downloads on first use (~75 MB)

# OMP Error: "libiomp5.dylib already initialized" (macOS, Whisper/embedding)
# Use make dev-gateway (sets KMP_DUPLICATE_LIB_OK), or add KMP_DUPLICATE_LIB_OK=TRUE to .env.local

# Embedding model (RAG)
# all-MiniLM-L6-v2 ~80MB, fast load. Set EMBEDDING_LOCAL_ONLY=true after first run.
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
| **Thinking LLM** | Qwen 3.5 9B (4-bit, VLM — text + vision) | — |
| **Instant LLM** | Qwen 3.5 4B Instruct (4-bit, text-only) | — |
| **Embeddings** | sentence-transformers (Orange/orange-nomic-v1.5-1536) | — |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L6-v2 | — |
| **STT** | faster-whisper (tiny model) | 1.1.1 |
| **TTS** | Kokoro FastAPI (Docker) | Latest |
| **Web Search** | SearXNG (Docker) | Latest |
| **Speaker ID** | MFCC embeddings (numpy) | — |
| **Networking** | Tailscale VPN (optional, for multi-machine) | — |
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

### Phase 3: Voice & Search ✅
- [x] Speech-to-text (faster-whisper)
- [x] Text-to-speech (Kokoro TTS via Docker)
- [x] Voice conversation mode ("Chat")
- [x] Wake word detection ("Hey Chat")
- [x] Silence detection (auto-stop recording)
- [x] Speaker verification (MFCC voice enrollment)
- [x] Web search (SearXNG, self-hosted)
- [x] LLM-powered search intent classification (three-tier routing)
- [x] Full-page content fetching (trafilatura)
- [x] Source trust classification and duplicate detection
- [x] Search results cited in responses with source URLs

### Phase 4: RAG & Vision ✅ (Current)
- [x] Document upload (PDF, DOCX, TXT, MD) with text extraction
- [x] Chunking with overlap (1024 chars, 128 overlap, markdown-aware)
- [x] Embedding generation (Orange/orange-nomic-v1.5-1536)
- [x] Hybrid search (vector + BM25 with RRF fusion)
- [x] Cross-encoder reranking (ms-marco-MiniLM-L6-v2)
- [x] Contextual retrieval (LLM-generated chunk context prefixes)
- [x] Session document attachments ("Attach to Chat")
- [x] RAG quality metrics (hit rate, MRR, latency)
- [x] Evidence summary for model confidence calibration
- [x] Image understanding (vision) — file picker, paste, drag-and-drop
- [x] Image compression (max 1024px) for memory safety
- [x] Image storage in Supabase Storage with RLS
- [x] Image display in chat history

### Phase 5: UX Polish ✅
- [x] Copy button on code blocks
- [x] Regenerate message button
- [x] Session search/filter and pinning
- [x] Inline source citations ([1], [2] in text)
- [x] Math/LaTeX rendering (KaTeX)
- [x] Conversation export (Markdown)

### Phase 6: Distribution ✅
- [x] Root `.env.example` with comprehensive defaults
- [x] `setup.sh` one-command installer (single + two-machine)
- [x] Unified `docker-compose.yml` (SearXNG + Kokoro TTS)
- [x] Configurable CORS (no hardcoded IPs)
- [x] Generalized README for any Apple Silicon user

### Future
- [ ] Custom tool/plugin support
- [ ] `curl | bash` remote installer
- [ ] OAuth sign-in (GitHub, Google)
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

MIT License © 2026

See [LICENSE](LICENSE) for details.

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

## Other Platforms

This project is optimized for **Apple Silicon** (M1/M2/M3/M4) using MLX for inference. The gateway and webapp work on any OS, but you'll need to swap the inference backend:

- **NVIDIA GPU (Linux/Windows)**: Use [vLLM](https://github.com/vllm-project/vllm) or [llama.cpp](https://github.com/ggerganov/llama.cpp) instead of MLX. Point `INFERENCE_THINKING_URL` / `INFERENCE_INSTANT_URL` at your server.
- **Ollama**: Serve Qwen models via [Ollama](https://ollama.com/) and point the inference URLs at it.

The rest of the stack (Supabase, SearXNG, Kokoro TTS) runs in Docker and works on any platform.

---

Built for privacy-conscious AI enthusiasts who want to own their data and run everything locally.
