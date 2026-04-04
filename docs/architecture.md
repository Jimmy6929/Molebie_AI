# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Web App (Next.js 16)                       │   │
│  │  - Chat UI (Streaming)    - Deep Think Toggle           │   │
│  │  - Session History        - Voice Conversation          │   │
│  │  - Image Upload/Paste     - Document Brain (RAG)        │   │
│  │  - Auth via JWT           - Web Search Sources          │   │
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
│  │              SQLite + Local Storage                     │   │
│  │  - Auth (JWT + bcrypt)   - Storage (Local filesystem)   │   │
│  │  - SQLite (Data)         - sqlite-vec (RAG Embeddings)  │   │
│  │  - User isolation        - FTS5 Full-Text Search        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
molebie-ai/
├── pyproject.toml                   # CLI package (pip install -e .)
├── .env.example                     # Environment config template
├── Makefile                         # Development commands
├── docker-compose.yml               # SearXNG + Kokoro TTS
│
├── cli/                             # CLI (Python + Typer)
│   ├── main.py                      # Typer app entry point
│   ├── commands/                    # install, run, doctor, status, config, feature, model
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
│       └── lib/                     # API client, voice hooks, auth
│
├── data/                            # Runtime data (SQLite DB, documents, images)
│
├── searxng/                         # SearXNG search config
│   └── settings.yml
│
└── scripts/                         # MLX server wrapper, auto-pull daemon
```

## Multi-Machine Setup

Split across two machines — a GPU machine for inference and a server for everything else:

```
Browser → Webapp (:3000) → Gateway (:8000) ──┬──→ Thinking LLM (:8080)  ← GPU machine
              │                                └──→ Instant LLM  (:8081)  ← GPU machine
              └──→ SQLite (data/molebie.db)                         ← Server
```

Run `molebie-ai install` and choose "Two machines" when prompted. The installer will:
- Ask for your GPU machine IP and server IP
- Configure all endpoints in `.env.local` automatically
- Configure auth endpoints for cross-machine access
- Print exact commands to run on the GPU machine

On the **server machine** (runs gateway, webapp, database):
```bash
molebie-ai install    # Choose "Two machines", enter IPs
molebie-ai run        # Starts Gateway, Webapp, Docker services
```

On the **GPU machine** (runs inference only):
```bash
python -m pip install -U mlx-vlm
make mlx-thinking     # Port 8080
make mlx-instant      # Port 8081 (optional)
```

The CLI verifies the remote GPU is reachable before starting local services.
