# Configuration

## CLI Commands

| Command | Description |
|---------|-------------|
| **Startup** | |
| `molebie-ai run` | Start all services — auto-configures on first run |
| `molebie-ai install` | Interactive setup wizard (optional — for full control) |
| **Diagnostics** | |
| `molebie-ai doctor` | Diagnose problems — checks dependencies, config, and service health |
| `molebie-ai doctor --fix` | Auto-generate missing `.env.local` and config |
| `molebie-ai status` | Show current configuration and which services are running |
| **Configuration** | |
| `molebie-ai config init` | Generate `.env.local` from template (auto-creates JWT secret) |
| `molebie-ai config show` | Display saved setup configuration (JSON) |
| `molebie-ai config env` | List all environment variables from `.env.local` (secrets masked) |
| `molebie-ai config get KEY` | Show the value of an environment variable |
| `molebie-ai config set KEY=VALUE` | Update an environment variable in `.env.local` |
| `molebie-ai config profile light` | Switch to light models (4B+4B) — less RAM |
| `molebie-ai config profile balanced` | Switch to balanced models (9B+4B) — better quality |
| **Models** | |
| `molebie-ai model list` | Show available models, download status, and active tier |
| `molebie-ai model add 9b` | Download/pull a model (aliases: `4b`, `9b`, or full name) |
| `molebie-ai model remove 9b` | Remove a downloaded model from disk |
| `molebie-ai model start` | Start inference server(s) (`--tier thinking`, `instant`, or `all`) |
| `molebie-ai model stop` | Stop inference server(s) (`--tier thinking`, `instant`, or `all`) |
| **Features** | |
| `molebie-ai feature list` | Show optional features and their status |
| `molebie-ai feature add voice` | Enable a feature and start its service (also: `search`, `rag`) |
| `molebie-ai feature remove voice` | Disable a feature and stop its service |

## Customizing the Assistant Personality

Edit the prompt files in `gateway/prompts/`:
- **`system.txt`** — Main chat personality and behavior
- **`system_voice.txt`** — Voice conversation mode (shorter, optimized for speech)

Template variables: `{current_date}` (auto-injected)

## Environment Variables

All config lives in `.env.local` at the project root. The CLI manages this for you:

```bash
molebie-ai config env                  # List all variables
molebie-ai config get KEY              # Show a specific value
molebie-ai config set KEY=VALUE        # Update a value
molebie-ai config init                 # Regenerate from template
```

Or copy the template manually: `cp .env.example .env.local`

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

## Model Management

Manage LLM models after installation — download new models, remove old ones, and start/stop inference servers independently:

```bash
molebie-ai model list             # See what's downloaded and active
molebie-ai model add 9b           # Download the 9B model
molebie-ai model add 4b           # Download the 4B model
molebie-ai model remove 9b        # Remove a model from disk
molebie-ai model start --tier thinking   # Start thinking server (:8080)
molebie-ai model stop --tier instant     # Stop instant server (:8081)
```

| Alias | MLX Model | Ollama Model |
|-------|-----------|-------------|
| `4b` | `mlx-community/Qwen3.5-4B-4bit` | `qwen3:4b` |
| `9b` | `mlx-community/Qwen3.5-9B-MLX-4bit` | `qwen3:8b` |

Full model names are also accepted. Aliases resolve automatically based on your configured backend.

## Optional Features

Toggle features via the CLI. Docker services are started/stopped automatically:

```bash
molebie-ai feature list          # See what's enabled
molebie-ai feature add voice     # Enable voice + starts Kokoro TTS container
molebie-ai feature remove search # Disable search + stops SearXNG container
```

| Feature | CLI Toggle | Env Variable | What happens |
|---------|-----------|-------------|-------------|
| Web Search | `feature add search` | `WEB_SEARCH_ENABLED` | Starts/stops SearXNG Docker container |
| Text-to-Speech | `feature add voice` | — | Starts/stops Kokoro TTS Docker container |
| RAG Documents | `feature add rag` | `RAG_ENABLED` | Embedding model downloads on first use |
| Image Vision | Always available | — | Requires a vision-capable model |

## Custom Wake Words

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

## Supported Inference Backends

| Backend | Command | Notes |
|---------|---------|-------|
| **MLX** (Apple Silicon) | `make mlx-thinking` | Best for Mac. Auto-installed by CLI |
| **Ollama** | `ollama serve` | Easiest cross-platform. Auto-configured by CLI |
| **vLLM** | `vllm serve <model>` | Production GPU servers |
| **llama.cpp** | `llama-server -m <model>` | Lightweight, any hardware |
| **OpenAI API** | Set `INFERENCE_API_KEY` in `.env.local` | Cloud fallback |

## Cost Controls

| Variable | Description | Default |
|----------|-------------|---------|
| `THINKING_DAILY_REQUEST_LIMIT` | Max thinking requests/user/day | `100` |
| `THINKING_MAX_CONCURRENT` | Max parallel thinking requests | `2` |
