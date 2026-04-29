#!/usr/bin/env bash
# Molebie — llama.cpp launch script for Qwen3.5-9B (thinking tier)
#
# Prereqs:
#   * llama.cpp build b5092 or later (Qwen3.5 architecture support)
#   * GGUF model — Unsloth's UD-Q4_K_XL is statistically indistinguishable
#     from Q8 per their Dynamic 2.0 benchmarks; smaller, faster, same quality
#   * Re-pull all GGUFs after 2026-03-05 to get Unsloth's chat-template fix
#     (older builds drop tool calls intermittently)
#
# Critical flags:
#   -fa on               KV-cache quantization is silently IGNORED without this
#   -c 131072            override default ctx (don't fill the 262K — LITM)
#   --no-context-shift   disables auto-shift, gives predictable behaviour
#   --jinja              use the model's chat template (required for Qwen)
#   chat-template-kwargs disable thinking by default; gateway overrides per-request
#
# After launch, point .env.local at it:
#   INFERENCE_THINKING_URL=http://localhost:8080
#   INFERENCE_THINKING_MODEL=Qwen3.5-9B
#   INFERENCE_THINKING_API_PREFIX=/v1
#
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL}"
PORT="${PORT:-8080}"

exec ./llama-server \
  -hf "$MODEL_REPO" \
  --port "$PORT" \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -c 131072 \
  -n 32768 \
  -ngl 99 \
  -fa on \
  -ub 2048 \
  -b 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0 \
  --no-context-shift
