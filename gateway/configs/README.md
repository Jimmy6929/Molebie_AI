# Inference Backend Configs

Ready-to-use configs for running Qwen3.5 4B/9B with the sampling presets,
KV-cache settings, and chat-template flags Molebie expects. Pick the file
that matches your backend; everything is set so the gateway's defaults
(`gateway/app/config.py`) work without further tuning.

| Backend                      | File                        | Notes |
|------------------------------|-----------------------------|-------|
| Ollama (9B)                  | `ollama-modelfile-9b`       | Run `ollama create molebie-9b -f ollama-modelfile-9b` |
| Ollama (4B)                  | `ollama-modelfile-4b`       | Same idea, smaller model |
| llama.cpp server             | `llamacpp-9b.sh`            | Requires build b5092+ for Qwen3.5 arch |
| vLLM                         | `vllm-9b.sh`                | OpenAI-compatible OOTB |
| MLX (current Molebie default)| (set via `.env.local`)      | `mlx_vlm.server` / `mlx_lm.server` already work |

## Critical traps to avoid

1. **Ollama default `num_ctx` is 2048.** This is the #1 bug in offline RAG
   setups — most documents will silently truncate. Always override.
2. **llama.cpp `-fa on` is required for KV-cache quantization.** Without
   it, `--cache-type-k q8_0` is silently ignored.
3. **llama.cpp build must be b5092+** for Qwen3.5 architecture support.
4. **Re-pull all GGUFs after 2026-03-05** — Unsloth's chat-template fix
   corrected tool-calling, prior builds drop tool calls intermittently.
5. **Don't fill 262K context.** Stay under 128K. Targeted retrieval to
   ~6–8K tokens beats stuffing every time.
6. **`presence_penalty` capped at 1.5.** Above that, Qwen3.5 starts mixing
   languages mid-response. The gateway already clamps; don't override
   above 1.5 in your backend either.
