#!/usr/bin/env bash
# Molebie — vLLM launch script for Qwen3.5-9B (thinking tier)
#
# Prereqs:
#   * vLLM 0.7+ for full Qwen3.5 support
#   * NVIDIA GPU with >= 24GB VRAM for 9B at bf16
#
# Critical flags:
#   --max-model-len             cap below the 262K ceiling — LITM hurts smaller models
#   --tool-call-parser hermes   required for Qwen3.5 tool calling via OpenAI-compatible API
#   --enable-auto-tool-choice   so tool selection works without explicit tool_choice
#   --gpu-memory-utilization    leave headroom for KV-cache growth
#
# Optional (Phase 5 task 5.7) — speculative decoding:
#   --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3.5-0.8B","num_speculative_tokens":5}'
#
# After launch, point .env.local at it:
#   INFERENCE_THINKING_URL=http://localhost:8000
#   INFERENCE_THINKING_MODEL=Qwen/Qwen3.5-9B-Instruct
#   INFERENCE_THINKING_API_PREFIX=/v1
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B-Instruct}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 131072 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill
