# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-04-04

### Added
- Self-hosted AI assistant with chat, voice, vision, RAG, and web search
- Three inference modes: Instant, Thinking, Think Harder
- CLI (`molebie-ai`) for install, run, doctor, status, config, model, feature
- One-line installer (`curl ... | bash`) with auto-detection and setup wizard
- Multi-machine deployment support (split GPU inference from server)
- Cross-platform: macOS (ARM64/x86), Linux, Windows (WSL2)
- SQLite-based auth and storage (no cloud dependency)
- Works with MLX, Ollama, vLLM, llama.cpp, or OpenAI API
- RAG document memory with hybrid vector + BM25 search and cross-encoder reranking
- Web search via self-hosted SearXNG with LLM intent classification
- Voice conversation with wake-word, Whisper STT, Kokoro TTS, speaker verification
- Image understanding via vision-capable models
- Multi-user support with full data isolation
