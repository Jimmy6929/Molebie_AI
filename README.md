# Local AI Assistant

A private, self-hosted AI assistant system with multi-user support, two-tier inference, and full data ownership.

## Overview

Local AI Assistant is a production-ready chat application that provides a ChatGPT-like experience while running entirely on your own infrastructure. It features:

- **Thinking Inference**: Deep reasoning with Qwen 3.5 9B (instant tier available for future use)
- **Full Data Ownership**: All conversations and documents stored in your Supabase instance
- **Multi-User Ready**: Row-level security (RLS) enabled from day one
- **No Third-Party LLM Costs**: Use your own GPU infrastructure for inference

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Web App (Next.js 16)                       │   │
│  │  - Chat UI          - Deep Think Toggle                 │   │
│  │  - Session History  - File Upload                       │   │
│  │  - Auth via Supabase                                    │   │
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
│  │  - Request Routing     - RAG Retrieval (Future)         │   │
│  │  - Tool Execution      - Logging & Audit                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                    │                    │
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     THINKING INFERENCE                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ MLX-VLM Server :8080                                      │  │
│  │ Qwen 3.5 9B · enable_thinking=True                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
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
├── .env.local               # Shared environment config (all services)
├── Makefile                 # Development commands (make dev, make test)
├── README.md
│
├── gateway/                 # FastAPI Backend (Python)
│   ├── app/
│   │   ├── main.py         # Application entry point
│   │   ├── config.py       # Configuration & settings
│   │   ├── middleware/     # JWT auth middleware
│   │   ├── models/         # Pydantic models
│   │   ├── routes/         # API endpoints (chat, health)
│   │   └── services/       # Database & inference services
│   ├── docs/               # Gateway-specific documentation
│   │   ├── GPU_SETUP_GUIDE.md
│   │   └── GPU_SETUP_GUIDE_THINKING.md
│   ├── tests/              # Pytest test suite
│   │   ├── conftest.py     # Shared fixtures
│   │   └── test_*.py       # Test files
│   ├── pyproject.toml      # Python project config
│   └── requirements.txt    # Python dependencies
│
├── webapp/                  # Next.js Frontend (TypeScript)
│   ├── src/
│   │   ├── app/            # Next.js App Router pages
│   │   │   ├── page.tsx    # Home page
│   │   │   ├── chat/       # Chat interface
│   │   │   └── login/      # Authentication page
│   │   └── lib/            # Utility libraries
│   │       ├── gateway.ts  # Gateway API client
│   │       └── supabase.ts # Supabase client
│   ├── package.json
│   └── tsconfig.json
│
├── supabase/                # Database Configuration
│   ├── config.toml         # Supabase local config
│   ├── migrations/         # SQL migrations
│   │   └── 20260222000000_initial_schema.sql
│   └── snippets/           # SQL reference snippets (00-07)
│
└── docs/                    # Project documentation (private, not in git)
```

## Quick Start

### Prerequisites

- **macOS/Linux** development machine
- **Docker Desktop** (for local Supabase)
- **Node.js 18+** (for Web App)
- **Python 3.10+** (for Gateway API)
- **Git** (for version control)
- **Supabase CLI** (`brew install supabase/tap/supabase`)

### Quick Commands (Makefile)

```bash
make help           # Show all available commands
make install        # Install all dependencies (gateway + webapp)
make dev            # Instructions for starting all services
make dev-gateway    # Start Gateway API only
make dev-webapp     # Start Web App only
make dev-supabase   # Start Supabase only
make test           # Run all tests
make test-gateway   # Run gateway tests with pytest
make lint           # Lint gateway code
make format         # Format gateway code with black
make clean          # Remove build artifacts
make stop           # Stop all services
```

### Startup Checklist (Daily Use)

You need to start **4 services across 2 machines**. Order matters.

#### On M2 Pro (GPU Machine) — 1 terminal

```bash
# Terminal 1 — Thinking LLM (Qwen 3.5 9B)
mlx_vlm.server --model mlx-community/Qwen3.5-9B-4bit --host 0.0.0.0 --port 8080
```

#### On MacBook 2019 (Home Server) — 3 terminals

```bash
# Terminal 1 — Supabase (make sure Docker Desktop is running first)
cd ~/Documents/App-project/Local_AI_Project/supabase
supabase start

# Terminal 2 — Gateway API (wait for Supabase to finish)
cd ~/Documents/App-project/Local_AI_Project/gateway
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3 — Web App
cd ~/Documents/App-project/Local_AI_Project/webapp
npm run dev
```

#### Verify Everything Is Running

```bash
# From M2 Pro — check local LLM server
curl http://localhost:8080/health                    # Thinking LLM

# From M2 Pro — check Home Server services via Tailscale
curl http://100.99.189.104:8000/health               # Gateway
curl http://100.99.189.104:8000/health/inference      # Gateway → LLM connection
open http://100.99.189.104:3000                       # Web App
```

> **Note:** All services share a single `.env.local` at the project root.
> Gateway and webapp both read from `../.env.local` automatically — no per-folder env files needed.
> **The `.env.local` must be copied manually** between machines (it's gitignored).

| Service | Machine | URL | What it does |
|---------|---------|-----|--------------|
| **Thinking LLM** | M2 Pro | http://localhost:8080 | Qwen 3.5 9B via mlx_vlm |
| **Supabase** | 2019 MacBook | http://127.0.0.1:54321 | Auth + Database (Docker) |
| **Supabase Studio** | 2019 MacBook | http://127.0.0.1:54323 | DB admin dashboard |
| **Gateway API** | 2019 MacBook | http://127.0.0.1:8000 | Routes chat to DB + LLM |
| **Web App** | 2019 MacBook | http://localhost:3000 | Next.js frontend |

### Stopping & Checking Services

Check if a service is running:

```bash
lsof -i :8000    # Gateway
lsof -i :3000    # Web App
lsof -i :54321   # Supabase
```

Stop a service:

```bash
# Stop Gateway
kill $(lsof -t -i :8000)

# Stop Web App
kill $(lsof -t -i :3000)

# Stop Supabase
supabase stop

# Or if the service is running in a terminal, just press Ctrl+C
```

> **Tip:** If you get `Address already in use` when starting a service, it's already running.
> Run `curl http://127.0.0.1:8000/health` to confirm, or kill it with `kill $(lsof -t -i :8000)` and restart.

### 1. Clone the Repository

```bash
git clone git@github.com:Jimmy6929/local_AI.git
cd local_AI
```

### 2. Start Local Supabase

```bash
# Install Supabase CLI (if not installed)
brew install supabase/tap/supabase

# Start local Supabase services
supabase start

# Note the credentials printed (or check LOCAL-SUPABASE-INFO.txt)
```

Local Supabase endpoints:
| Service | URL |
|---------|-----|
| Studio (Dashboard) | http://127.0.0.1:54323 |
| API | http://127.0.0.1:54321 |
| Database | postgresql://postgres:postgres@127.0.0.1:54322/postgres |

### 3. Set Up the Gateway API

```bash
cd gateway

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server (reads .env.local from project root automatically)
uvicorn app.main:app --reload --port 8000

# Or use Makefile
make dev-gateway
```

The Gateway API will be available at http://localhost:8000

### 4. Set Up the Web App

```bash
cd webapp

# Install dependencies
npm install

# Start development server (reads .env.local from project root automatically)
npm run dev

# Or use Makefile
make dev-webapp
```

The Web App will be available at http://localhost:3000

## API Reference

### Health Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Basic health check |
| `/health/auth` | GET | Yes | Validates JWT and returns user info |
| `/health/inference` | GET | No | Checks inference endpoint status |

### Chat Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat` | POST | Yes | Send message and receive AI response |
| `/chat/sessions` | GET | Yes | List user's chat sessions |
| `/chat/sessions/{id}` | GET | Yes | Get session details |
| `/chat/sessions/{id}/messages` | GET | Yes | Get messages in a session |
| `/chat/sessions/{id}` | PATCH | Yes | Update session (rename, archive) |
| `/chat/sessions/{id}` | DELETE | Yes | Delete a session |

### POST /chat Request

```json
{
  "session_id": "uuid | null",
  "message": "string",
  "mode": "instant | think"
}
```

### POST /chat Response

```json
{
  "session_id": "uuid",
  "message_id": "uuid",
  "content": "string",
  "mode_used": "instant | think",
  "tokens_used": 150,
  "latency_ms": 1200
}
```

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `profiles` | User profile information (auto-created on signup) |
| `chat_sessions` | Chat conversation sessions |
| `chat_messages` | Individual messages within sessions |
| `documents` | Uploaded document metadata (future RAG) |
| `document_chunks` | Document chunks with embeddings (future RAG) |

### Entity Relationship

```
profiles (1) ─────< chat_sessions (1) ─────< chat_messages
    │
    └──────────────< documents (1) ────────< document_chunks
```

### Row-Level Security (RLS)

All tables have RLS enabled. Users can only access their own data:

- `profiles`: Users can view/update only their own profile
- `chat_sessions`: Users can CRUD only their own sessions
- `chat_messages`: Users can CRUD only messages in their sessions
- `documents`: Users can CRUD only their own documents

## Two-Machine Setup

This project runs across **two machines** connected via Tailscale VPN:

| Machine | Role | Tailscale IP |
|---------|------|-------------|
| **MacBook Pro 2019 (i7)** | Home Server: Gateway, Webapp, Docker, Supabase | `100.99.189.104` |
| **MacBook Pro M2 Pro (16GB)** | GPU Machine: MLX inference servers (both LLMs) | `100.104.193.59` |

```
Browser → Webapp (:3000) → Gateway (:8000) → MLX-VLM server on M2 Pro (:8080)
              Home Server                          GPU Machine
```

## Inference Mode

The thinking tier runs Qwen 3.5 9B via `mlx_vlm.server` with chain-of-thought reasoning enabled:

| Tier | Model | Server | Port | API Path | Use Case |
|------|-------|--------|------|----------|----------|
| **Thinking** | `mlx-community/Qwen3.5-9B-4bit` | `mlx_vlm.server` | 8080 | `/chat/completions` | Reasoning, coding, analysis |
| **Instant** | *(not configured)* | — | — | — | Reserved for future use |

> **Why mlx_vlm?** Qwen 3.5 is a VLM (Vision-Language Model) that includes an image/video encoder.
> It requires `mlx-vlm` (which supports vision models). It works great for text chat and
> *also* has vision capabilities for future use.

### MLX Setup (GPU Machine — M2 Pro)

#### One-Time Installation

```bash
# Install mlx-vlm with PyTorch (for Qwen 3.5 9B)
pip install -U "mlx-vlm[torch]"
```

> **Important:** Qwen 3.5 requires PyTorch + Torchvision for its video/image processor.
> Use `pip install -U "mlx-vlm[torch]"` to get everything in one command.

#### Starting the LLM Server (Every Session)

Open **one terminal** on the M2 Pro:

```bash
# Thinking tier (Qwen 3.5 9B VLM)
mlx_vlm.server --model mlx-community/Qwen3.5-9B-4bit --host 0.0.0.0 --port 8080
```

#### Verify LLM Server Is Running

```bash
curl http://localhost:8080/health
curl http://localhost:8080/models
```

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
- **CORS Configuration**: Restricted to allowed origins

## Documentation

Detailed documentation is maintained locally in the `/docs` folder (not included in git for privacy).

Documentation structure:

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

## Complete Command Reference

### M2 Pro (GPU Machine)

```bash
# ── One-Time Setup ──────────────────────────────────
pip install -U "mlx-vlm[torch]"          # Qwen 3.5 9B — needs PyTorch

# ── Start LLM Server (every session) ───────────────
mlx_vlm.server --model mlx-community/Qwen3.5-9B-4bit --host 0.0.0.0 --port 8080   # Thinking (Qwen 3.5 9B)

# ── Health Checks ──────────────────────────────────
curl http://localhost:8080/health         # Thinking LLM status
curl http://localhost:8080/models         # List loaded VLM models
```

### MacBook 2019 (Home Server)

```bash
# ── One-Time Setup ──────────────────────────────────
cd ~/Documents/App-project/Local_AI_Project
cd gateway && pip3 install -r requirements.txt       # Gateway Python deps
cd ../webapp && npm install                           # Webapp Node deps
brew install supabase/tap/supabase                   # Supabase CLI

# ── Start Services (every session, in order) ───────
# 1. Make sure Docker Desktop is open first!
cd ~/Documents/App-project/Local_AI_Project/supabase && supabase start

# 2. Gateway (in a new terminal)
cd ~/Documents/App-project/Local_AI_Project/gateway
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Webapp (in a new terminal)
cd ~/Documents/App-project/Local_AI_Project/webapp && npm run dev

# ── Stop Services ──────────────────────────────────
supabase stop                             # Stop Supabase + Docker containers
kill $(lsof -t -i :8000)                  # Stop Gateway
kill $(lsof -t -i :3000)                  # Stop Webapp
# Or just Ctrl+C in each terminal

# ── Health Checks ──────────────────────────────────
curl http://127.0.0.1:8000/health                    # Gateway
curl http://127.0.0.1:8000/health/inference           # Gateway → LLM connection
curl http://127.0.0.1:54321/auth/v1/health            # Supabase Auth
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
make test                # Run all tests
make lint                # Lint gateway
make format              # Format gateway code
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

# "Address already in use" on any port
lsof -i :<port>                           # Check what's using the port
kill $(lsof -t -i :<port>)                # Kill it

# Gateway shows old config after .env.local change
# Restart the gateway (Ctrl+C, then start again)

# Supabase 401 errors
supabase status                           # Check keys match .env.local
```

## Roadmap

### Phase 1: Chat MVP ✅
- [x] Local Supabase setup
- [x] Database schema with RLS
- [x] Gateway API (FastAPI)
- [x] Chat endpoints
- [x] Web App (Next.js)
- [x] Connect to GPU inference

### Phase 2: Thinking Inference ✅ (Current)
- [x] Thinking Mode (Qwen 3.5 9B via mlx_vlm, enable_thinking=True)
- [x] Mode toggle in UI
- [x] Mode indicator on responses
- [x] Instant tier reserved for future use

### Phase 3: RAG (Document Memory)
- [ ] File upload functionality
- [ ] Document processing pipeline
- [ ] Embedding generation
- [ ] Context retrieval

### Phase 4: Tools Framework
- [ ] Web search integration
- [ ] Note saving
- [ ] Custom tool support

### Phase 5: Production Launch
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
pytest tests/

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

## Development

### Environment Variables

All environment variables live in a **single `.env.local` at the project root**. Both the gateway and webapp read from this file automatically.

```bash
# First-time setup: copy the template and fill in your values
cp .env.example .env.local
```

#### Gateway Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `SUPABASE_URL` | Supabase API URL | `http://127.0.0.1:54321` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | - |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | - |
| `JWT_SECRET` | JWT signing secret | - |
| `INFERENCE_INSTANT_URL` | Instant mode endpoint | - |
| `INFERENCE_THINKING_URL` | Thinking mode endpoint | - |
| `INFERENCE_MODEL_NAME` | Model name for inference | `default` |
| `INFERENCE_MAX_TOKENS` | Max tokens per response | `2048` |
| `INFERENCE_TEMPERATURE` | Model temperature | `0.7` |
| `INFERENCE_TIMEOUT` | Request timeout (seconds) | `120.0` |

#### Web App Variables

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase API URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anonymous key |
| `NEXT_PUBLIC_GATEWAY_URL` | Gateway API URL |

### Database Migrations

```bash
# Create new migration
supabase migration new <migration_name>

# Apply migrations (reset database)
supabase db reset

# View migration history
supabase migration list
```

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

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - MLX server for text LLMs
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - MLX server for vision-language models
- [Qwen](https://github.com/QwenLM/Qwen3) - Alibaba's open-source LLM family
- [Supabase](https://supabase.com/) - Backend as a Service
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Next.js](https://nextjs.org/) - React framework for production
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [Tailscale](https://tailscale.com/) - Zero-config VPN for the two-machine setup

---

Built for privacy-conscious AI enthusiasts.
