---
name: Full System Architecture Map
overview: Generate a comprehensive visual architecture and flow map of Molebie AI -- a self-hosted, ChatGPT-style chat application with private LLM inference, RAG, web search, voice, and vision support, covering all services, data flows, authentication, database schema, and physical deployment.
todos: []
isProject: false
---

# Full System Architecture Map

Below is the complete infrastructure and architecture of Molebie AI, broken down into multiple diagrams covering every layer.

---

## 1. High-Level Architecture (All Services)

This is the bird's-eye view of how every component connects:

```mermaid
graph TB
    subgraph browser [Browser]
        User["User (Browser)"]
    end

    subgraph machine1 ["Machine 1 — Server (100.99.189.104)"]
        WebApp["Next.js 16 Web App\n:3000"]
        Gateway["FastAPI Gateway\n:8000"]
        subgraph dockerSvc [Docker Services]
            SearXNG["SearXNG\nWeb Search\n:8888"]
            KokoroTTS["Kokoro TTS\n:8880"]
        end
        subgraph supabaseSvc [Supabase Docker Stack]
            SupaAuth["Supabase Auth"]
            SupaREST["PostgREST API\n:54321"]
            Postgres["PostgreSQL 17\n:54322\npgvector + pg_trgm"]
            Studio["Supabase Studio\n:54323"]
            Storage["Supabase Storage\n(documents, chat-images)"]
        end
    end

    subgraph machine2 ["Machine 2 — GPU Node (100.104.193.59)"]
        ThinkingLLM["Inference — Thinking Tier\n(MLX / Ollama / vLLM / OpenAI)\n:8080"]
        InstantLLM["Inference — Instant Tier\n(MLX / Ollama / vLLM / OpenAI)\n:8081"]
    end

    User -->|"HTTPS :3000"| WebApp
    WebApp -->|"Supabase Auth SDK"| SupaAuth
    WebApp -->|"REST + JWT\n:8000"| Gateway
    Gateway -->|"REST + JWT (RLS)\n:54321"| SupaREST
    SupaREST --> Postgres
    SupaAuth --> Postgres
    Storage --> Postgres
    Gateway -->|"HTTP /v1/chat/completions\nvia Tailscale :8080"| ThinkingLLM
    Gateway -->|"HTTP /v1/chat/completions\nvia Tailscale :8081"| InstantLLM
    Gateway -->|"HTTP :8888"| SearXNG
    Gateway -->|"HTTP :8880"| KokoroTTS
```



**Key points:**

- Two physical machines connected via **Tailscale VPN** (also supports single-machine deployment)
- Machine 1 runs the web app, gateway, Supabase stack, SearXNG, and Kokoro TTS (all via Docker)
- Machine 2 runs GPU inference servers (supports MLX, Ollama, vLLM, llama.cpp, or OpenAI-compatible APIs)
- All inter-service communication is HTTP; no message queues or gRPC
- Gateway is the central orchestrator — routes to inference, RAG, web search, TTS, and database
- `molebie-ai` CLI manages setup, configuration, and service lifecycle

---

## 2. Request Flow — Chat Completion (Streaming)

The main user-facing flow when sending a chat message, now including web search and RAG context injection:

```mermaid
sequenceDiagram
    participant U as Browser
    participant W as Next.js :3000
    participant G as Gateway :8000
    participant DB as Supabase/Postgres
    participant S as SearXNG :8888
    participant RAG as RAG Pipeline
    participant LLM as Inference Tier

    U->>W: Type message, click Send
    W->>W: Get JWT from Supabase Auth session
    W->>G: POST /chat/stream {message, mode, session_id, images[]}<br/>Authorization: Bearer JWT

    G->>G: Decode JWT, extract user_id
    G->>DB: GET or CREATE chat_session (via REST + RLS)
    DB-->>G: session_id

    G->>DB: INSERT chat_message (role=user) + store images
    G->>DB: SELECT last 20 messages for context

    par Context Enrichment
        G->>S: Intent classification → search if needed
        S-->>G: Web results + snippets
    and
        G->>RAG: Hybrid search (vector + BM25 → RRF → rerank)
        RAG-->>G: Relevant document chunks
    and
        G->>DB: GET session_documents (attached files)
        DB-->>G: Document content
    end

    G->>G: Build system prompt with evidence summary<br/>(web sources + RAG chunks + attached docs)

    G->>LLM: POST /v1/chat/completions<br/>{model, messages[], stream:true,<br/>enable_thinking, thinking_budget, images[]}

    loop SSE Streaming
        LLM-->>G: data: {delta.content, delta.reasoning_content}
        G-->>W: data: {delta.content, delta.reasoning_content}
        W-->>U: Render markdown + KaTeX incrementally
    end

    LLM-->>G: data: [DONE]
    G->>G: Strip think tags, extract reasoning_content
    G->>DB: INSERT chat_message (role=assistant,<br/>content, reasoning_content, mode_used)
    G-->>W: data: [DONE]
    W->>W: Parse think tags, show reasoning toggle
    W-->>U: Final rendered response with source citations
```



---

## 3. Authentication Flow

```mermaid
sequenceDiagram
    participant U as Browser
    participant W as Next.js :3000
    participant SA as Supabase Auth
    participant G as Gateway :8000

    U->>W: Navigate to /login
    U->>W: Enter email + password
    W->>SA: signInWithPassword(email, password)
    SA-->>W: JWT access_token + refresh_token
    W->>W: Store tokens in cookie (via @supabase/ssr)

    U->>W: Navigate to /chat
    W->>W: Read JWT from cookie
    W->>G: GET /chat/sessions<br/>Authorization: Bearer JWT
    G->>G: Decode JWT (HS256)<br/>Extract sub (user_id), email, role
    G->>G: Attach user_id to request context
    G-->>W: 200 OK + session list
```



**Auth details:**

- Supabase Auth issues JWTs (HS256, shared secret)
- Gateway decodes JWT locally (no round-trip to Supabase Auth for validation)
- All Supabase DB queries pass the user JWT for Row-Level Security enforcement
- In dev mode, signature verification is skipped for convenience

---

## 4. Inference Mode Routing

```mermaid
flowchart TD
    Req["Incoming Chat Request"]
    Backend{"Backend type?"}
    Mode{"mode parameter?"}

    Req --> Backend
    Backend -->|"MLX"| MLX["API prefix: empty string\nmlx_vlm.server / mlx_lm.server"]
    Backend -->|"Ollama / vLLM / llama.cpp"| OAI["API prefix: /v1\nOpenAI-compatible endpoint"]
    Backend -->|"OpenAI API"| Cloud["HTTPS + Bearer token\nAPI prefix: /v1"]

    MLX --> Mode
    OAI --> Mode
    Cloud --> Mode

    Mode -->|"instant"| Instant["Instant Tier\n:8081\nFast, no CoT"]
    Mode -->|"thinking"| Thinking["Thinking Tier\n:8080\nbudget: 2048 tokens"]
    Mode -->|"thinking_harder"| Harder["Thinking Tier\n:8080\nbudget: 8192 tokens\nmax_tokens: 28672"]

    Thinking -->|"fails?"| Fallback{"Fallback enabled?"}
    Harder -->|"fails?"| Fallback
    Fallback -->|"yes"| Instant
    Fallback -->|"no"| Error["Return Error"]

    Instant --> Resp["Return Response"]
    Thinking --> Resp
    Harder --> Resp
```



**Cost controls:**

- `THINKING_DAILY_REQUEST_LIMIT` caps heavy inference per user per day (default: 100)
- `THINKING_MAX_CONCURRENT` limits parallel thinking requests (default: 2)
- Fallback to instant tier is configurable via `ROUTING_THINKING_FALLBACK_TO_INSTANT`

**Supported inference backends:**

- **MLX** (Apple Silicon) — `mlx_vlm.server` or `mlx_lm.server`, API prefix `""`
- **Ollama** — HTTP, API prefix `/v1`
- **vLLM** — HTTP, API prefix `/v1`
- **llama.cpp** — HTTP, API prefix `/v1`
- **OpenAI API** — HTTPS with Bearer token, API prefix `/v1`

---

## 5. Web Search Pipeline

```mermaid
flowchart TD
    Msg["User Message"]
    Classify{"LLM Intent Classification\n(needs web search?)"}

    Msg --> Classify
    Classify -->|"no"| Skip["Skip search"]
    Classify -->|"yes"| Query["Generate search query"]
    Query --> SearX["SearXNG :8888\n(self-hosted metasearch)"]
    SearX --> Results["Top 6 results\n(title, URL, snippet)"]
    Results --> Fetch["Fetch full content\n(top 3 pages)"]
    Fetch --> Inject["Inject as evidence\ninto system prompt"]
    Inject --> Citations["Source citations\nrendered in UI"]
```

**Web search details:**

- Powered by **SearXNG** — self-hosted, privacy-respecting, no API keys needed
- **LLM intent classification** decides whether a query needs web results (configurable via `WEB_SEARCH_LLM_CLASSIFY`)
- Fetches full page content for top 3 results (up to 2000 chars each)
- Results injected as evidence blocks in the system prompt with quality labels

---

## 6. RAG Pipeline (Document Retrieval)

```mermaid
flowchart TD
    Upload["User uploads PDF/DOCX/TXT/MD"]
    Extract["DocumentProcessor\nextract text"]
    Chunk["Split into chunks\n(1024 chars, 128 overlap)"]
    Embed["Generate embeddings\n(sentence-transformers, 1536-dim)"]
    BM25["Generate tsvector\n(full-text search)"]
    Store["Store in document_chunks\n(pgvector + GIN index)"]

    Upload --> Extract --> Chunk --> Embed --> Store
    Chunk --> BM25 --> Store

    Query["User asks a question"]
    VecSearch["Vector similarity search\n(pgvector HNSW)"]
    TextSearch["BM25 full-text search\n(tsvector + GIN)"]
    RRF["Reciprocal Rank Fusion\n(weight: 0.7 vector / 0.3 text)"]
    Rerank["Cross-encoder reranking\n(ms-marco-MiniLM-L6-v2)"]
    Context["Top 5 chunks → system prompt"]

    Query --> VecSearch
    Query --> TextSearch
    VecSearch --> RRF
    TextSearch --> RRF
    RRF --> Rerank --> Context
```

**RAG details:**

- **Hybrid search**: vector similarity (pgvector HNSW) + BM25 full-text (tsvector + GIN) fused via RRF
- **Cross-encoder reranking** for final relevance scoring
- **Contextual retrieval**: LLM generates context prefixes for each chunk at ingest time
- **Session document attachments**: files can be attached to specific sessions and injected directly into the system prompt
- **Embedding model**: configurable (default: `sentence-transformers/all-MiniLM-L6-v2`, 1536-dim via Orange/orange-nomic)
- **RAG metrics**: performance logging with quality tracking

---

## 7. Voice Pipeline

```mermaid
sequenceDiagram
    participant U as Browser
    participant W as Next.js :3000
    participant G as Gateway :8000
    participant STT as Whisper (faster-whisper)
    participant TTS as Kokoro TTS :8880
    participant LLM as Inference Tier

    Note over U,W: Voice Conversation Mode
    U->>W: Hold mic / wake-word detected
    W->>W: Record audio (Web Audio API)
    W->>G: POST /chat/transcribe (audio file)
    G->>STT: Transcribe audio
    STT-->>G: Transcribed text
    G-->>W: {text: "..."}

    W->>G: POST /chat/stream {message: transcribed text}
    G->>LLM: Generate response (streaming)
    LLM-->>G: Response text
    G-->>W: SSE response

    W->>G: POST /chat/tts {text, voice, speed}
    G->>TTS: Synthesize speech
    TTS-->>G: Audio (WAV)
    G-->>W: Audio response
    W->>U: Play audio response
```

**Voice details:**

- **STT**: `faster-whisper` (local Whisper inference) via `POST /chat/transcribe`
- **TTS**: Kokoro FastAPI (Docker, CPU) with 12 voice options (British/American, male/female)
- **Speaker verification**: enroll voice profile, verify subsequent speakers match
- **Wake-word detection**: browser-side voice activity detection
- **Voice settings**: configurable voice, speed (0.5x–2.0x), auto-read toggle

---

## 8. Database Schema (ER Diagram)

```mermaid
erDiagram
    auth_users {
        uuid id PK
        text email
        jsonb raw_user_meta_data
    }

    profiles {
        uuid id PK,FK
        text email
        text name
        text avatar_url
        timestamptz created_at
        timestamptz updated_at
    }

    chat_sessions {
        uuid id PK
        uuid user_id FK
        text title
        boolean is_archived
        boolean is_pinned
        timestamptz created_at
        timestamptz updated_at
    }

    chat_messages {
        uuid id PK
        uuid session_id FK
        uuid user_id FK
        text role
        text content
        text reasoning_content
        text mode_used
        integer tokens_used
        timestamptz created_at
    }

    message_images {
        uuid id PK
        uuid message_id FK
        uuid user_id FK
        text storage_path
        text filename
        text mime_type
        integer file_size
        timestamptz created_at
    }

    documents {
        uuid id PK
        uuid user_id FK
        text filename
        text storage_path
        text file_type
        integer file_size
        text status
        timestamptz created_at
        timestamptz processed_at
    }

    document_chunks {
        uuid id PK
        uuid document_id FK
        uuid user_id FK
        text content
        vector embedding
        tsvector content_tsv
        integer chunk_index
        timestamptz created_at
    }

    session_documents {
        uuid id PK
        uuid session_id FK
        uuid user_id FK
        text filename
        text content
        integer file_size
        timestamptz created_at
    }

    auth_users ||--|| profiles : "trigger creates"
    profiles ||--o{ chat_sessions : "owns"
    chat_sessions ||--o{ chat_messages : "contains"
    profiles ||--o{ chat_messages : "authored by"
    chat_messages ||--o{ message_images : "has attachments"
    profiles ||--o{ message_images : "owns"
    profiles ||--o{ documents : "uploads"
    documents ||--o{ document_chunks : "split into"
    profiles ||--o{ document_chunks : "owns"
    chat_sessions ||--o{ session_documents : "has attached"
    profiles ||--o{ session_documents : "owns"
```



**Key schema features:**

- Row-Level Security on every table (users only see their own data)
- `auth.users` trigger auto-creates a `profiles` row on signup
- `chat_messages` trigger auto-updates `chat_sessions.updated_at`
- pgvector extension with **HNSW index** (M=16, ef_construction=64) for RAG similarity search
- **tsvector + GIN index** on `document_chunks` for BM25 full-text search
- **RRF hybrid search** function in Postgres for vector + BM25 fusion
- `mode_used` supports: `instant`, `thinking`, `thinking_harder`
- `message_images` stored in Supabase Storage (`chat-images` bucket), metadata in table
- `session_documents` holds full-text content injected directly into system prompts
- `chat_sessions.is_pinned` for session pinning/favoriting
- Storage buckets: `documents` (RAG uploads), `chat-images` (inline image attachments)

---

## 9. Gateway API Routes

```
/health
  GET /              — Basic health check
  GET /auth          — JWT validation + user info
  GET /inference     — Inference tier status

/chat
  POST /             — Send message (full response)
  POST /stream       — Send message (SSE streaming)
  GET  /sessions     — List user sessions
  POST /sessions/create  — Create new session
  GET  /sessions/{id}/messages — Get session messages
  PATCH /sessions/{id}   — Rename session
  PATCH /sessions/{id}/pin — Pin/unpin session
  DELETE /sessions/{id}  — Delete session
  POST /transcribe   — Whisper STT (audio → text)
  POST /tts          — Kokoro TTS (text → audio)
  POST /voice-enroll — Voice profile enrollment
  GET  /voice-profile — Get voice profile
  DELETE /voice-profile — Delete voice profile

/documents
  POST /upload       — Upload file for RAG processing
  GET  /             — List user documents
  DELETE /{id}       — Delete document + chunks
  POST /sessions/{id}/attach  — Attach document to session
  DELETE /sessions/{id}/attach — Remove attachment
```

---

## 10. Physical Deployment / Network Topology

```mermaid
graph LR
    subgraph tailnet [Tailscale VPN Mesh]
        subgraph server ["Server Machine\n100.99.189.104"]
            next["Next.js :3000"]
            fastapi["FastAPI :8000"]
            supa["Supabase Docker\n:54321 / :54322 / :54323"]
            searx["SearXNG :8888"]
            kokoro["Kokoro TTS :8880"]
        end
        subgraph gpu ["GPU Node\n100.104.193.59"]
            thinking["Thinking Tier :8080"]
            instant["Instant Tier :8081"]
        end
    end

    next --- fastapi
    fastapi --- supa
    fastapi --- searx
    fastapi --- kokoro
    fastapi ---|"Tailscale"| thinking
    fastapi ---|"Tailscale"| instant
```

**Deployment modes:**

- **Two-machine**: Gateway/webapp on server, inference on GPU node (Tailscale/LAN)
- **Single-machine**: Everything on localhost (configured via `molebie-ai install`)
- **Auto-pull daemon**: macOS LaunchAgent polls git and auto-updates on new commits

---

## 11. Frontend Page Structure

```mermaid
flowchart TD
    Root["/ (root page.tsx)"]
    Root -->|"authenticated?"| Chat["/chat\nMain Chat UI"]
    Root -->|"not authenticated"| Login["/login\nSign In / Sign Up"]

    Chat --> Sidebar["Session Sidebar\n(list, rename, pin, delete)"]
    Chat --> ChatArea["Chat Area\n(messages, streaming, images)"]
    Chat --> ModeSelect["Mode Selector\n(instant / thinking / thinking_harder)"]
    Chat --> VoiceMode["Voice Conversation Mode\n(STT + TTS + wake-word)"]
    Chat --> DocPanel["Document Panel\n(upload, attach, RAG brain)"]

    ChatArea --> Markdown["react-markdown\n+ syntax highlighting\n+ KaTeX math"]
    ChatArea --> ThinkBlock["Collapsible Reasoning\n(think tag parser)"]
    ChatArea --> Sources["Web Search Citations\n(source links)"]
    ChatArea --> ImageView["Image Attachments\n(paste/drag-drop/upload)"]
```



**Frontend stack:** Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4, Geist Mono font, dark glass UI theme with green accents. State is managed purely with React hooks (no external state library).

**Key frontend features:**

- Voice conversation mode with wake-word detection and speaker verification
- Document upload/attachment for RAG and per-session context
- Image upload via paste, drag-and-drop, or file picker (stored in Supabase Storage)
- Web search source citations with clickable links
- KaTeX math rendering in messages
- Session pinning/favoriting

---

## 12. CLI Tool (molebie-ai)

```mermaid
flowchart TD
    Install["molebie-ai install"]
    Run["molebie-ai run"]
    Doctor["molebie-ai doctor"]
    Status["molebie-ai status"]
    Config["molebie-ai config show/edit"]
    Feature["molebie-ai feature list/add/remove"]

    Install --> Prereqs["Check prerequisites\n(Docker, Node, Python, ffmpeg)"]
    Prereqs --> Backend["Select backend\n(MLX / Ollama / OpenAI)"]
    Backend --> Models["Select model profile\n(Light / Balanced / Custom)"]
    Models --> Features["Toggle features\n(voice, search, RAG)"]
    Features --> Deploy["Select deployment\n(single / two-machine)"]
    Deploy --> Setup["Run setup\n(Supabase, env gen, model downloads)"]

    Run --> Start["Start all configured services"]
    Doctor --> Diagnose["Check environment health"]
```

**CLI details:**

- **Framework**: Python + Typer + Rich
- **Entry point**: `molebie-ai` (installed via `pip install -e .`)
- **Config storage**: `.molebie/config.json`
- **Env generation**: Auto-generates `.env.local` from CLI config
- **Prerequisite checker**: Detects and offers to install missing dependencies
- **Service manager**: Starts/stops all services via subprocess

---

## Summary Table

| Service | Port | Framework | Purpose |
|---------|------|-----------|---------|
| **Web App** | 3000 | Next.js 16 | Chat UI, auth, voice, documents, images |
| **Gateway** | 8000 | FastAPI | Auth, routing, DB proxy, inference proxy, RAG, web search, TTS, SSE streaming |
| **Supabase** | 54321-54323 | Docker (Postgres 17) | Auth (JWT), PostgreSQL (RLS, pgvector, pg_trgm), Storage |
| **Thinking LLM** | 8080 | MLX / Ollama / vLLM / OpenAI | Deep reasoning with chain-of-thought |
| **Instant LLM** | 8081 | MLX / Ollama / vLLM / OpenAI | Fast responses, no CoT |
| **SearXNG** | 8888 | Docker | Self-hosted web search (no API keys) |
| **Kokoro TTS** | 8880 | Docker (FastAPI) | Text-to-speech (12 voices, CPU) |
| **Tailscale** | — | VPN mesh | Connects server + GPU node |
| **CLI** | — | Python (Typer) | Setup wizard, service management, diagnostics |

The gateway is the central orchestrator: it authenticates every request, manages sessions/messages in Supabase, routes to the appropriate inference tier, enriches context with web search and RAG results, handles voice transcription and synthesis, manages image attachments, builds evidence-augmented system prompts, handles streaming, extracts reasoning content, and applies cost controls.
