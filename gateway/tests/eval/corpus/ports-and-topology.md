# Molebie ports and service topology

How the pieces of a Molebie deployment talk to each other, and on which
ports. Source of truth for the single- and two-machine layouts described
in `docs/architecture.md`.

## Default ports

| Service | Port | Notes |
|---|---|---|
| Webapp (Next.js) | 3000 | the browser-facing UI |
| Gateway (FastAPI) | 8000 | all API traffic; auth, chat, documents |
| Thinking LLM server | 8080 | Qwen thinking tier (MLX / OpenAI-compatible) |
| Instant LLM server | 8081 | Qwen instant tier (MLX / OpenAI-compatible) |
| SearXNG (optional) | 8888 | local web-search backend |
| Kokoro TTS (optional) | 8880 | local text-to-speech |

## Request flow

Browser → Webapp (:3000) → Gateway (:8000), which fans out to the
Thinking LLM (:8080) and the Instant LLM (:8081), reads and writes the
local SQLite database (`data/molebie.db`), and optionally calls SearXNG
(:8888) for web search and Kokoro (:8880) for speech.

## Two-machine split

In the "split across machines" layout, the GPU machine hosts the two LLM
servers (:8080 and :8081) while the server machine hosts the webapp
(:3000), the gateway (:8000), and the SQLite file. The installer
configures all endpoints in `.env.local` and the auth endpoints for
cross-machine access.
