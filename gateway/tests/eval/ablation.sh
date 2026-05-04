#!/usr/bin/env bash
# Phase 3 ablation: run the baseline against four gateway boots and
# produce a comparison report.
#
#   1. baseline           — Phase 1 + 2 only (Judge / CoVe / SelfCheck off)
#   2. judge              — only Judge enabled
#   3. cove               — only CoVe enabled
#   4. all                — Judge + CoVe + SelfCheck enabled
#
# The point: see how each Phase 3 layer contributes pass-rate and latency
# *on top of* the Phase 1/2 stack. Numbers go into eval-results/.
#
# Prereqs:
#   * Local MLX/Ollama instance reachable per .env.local
#   * Single-user password set on first run; reuse the same one each run
#
# Usage:
#   PASSWORD=smoketest ./ablation.sh
#
# This script intentionally restarts the gateway between runs so each
# ablation flag is *real* (env vars are read at boot). Each run writes
# its full report to eval-results/<label>.json + a flat summary line to
# eval-results/summary.txt for quick eyeballing.
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(git -C . rev-parse --show-toplevel 2>/dev/null)"
[ -n "$ROOT_DIR" ] || ROOT_DIR="$(cd ../../.. && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

PASSWORD="${PASSWORD:-smoketest}"
PORT="${PORT:-8765}"
DATA_DIR="${DATA_DIR:-/tmp/molebie_ablation_data}"
GATEWAY="http://127.0.0.1:${PORT}"
RESULTS_DIR="${RESULTS_DIR:-eval-results}"
GATEWAY_LOG="${RESULTS_DIR}/gateway.log"
SKIP="${SKIP:-tool_call}"   # tool_call needs SearXNG + a stable corpus; skip by default

mkdir -p "$RESULTS_DIR"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"
: >"${RESULTS_DIR}/summary.txt"

# Common env: align EMBEDDING_DIM with whatever EMBEDDING_MODEL is in .env.local.
# If you've activated Phase 1.5 (Qwen3-Embedding-0.6B) leave EMBEDDING_DIM=1024;
# if you're still on Orange-1536 set EMBEDDING_DIM=1536. The dim-validation
# guard from F2 will fail loudly otherwise.
COMMON_ENV=(
  "DATA_DIR=${DATA_DIR}"
  "EMBEDDING_DIM=${EMBEDDING_DIM:-1536}"
  "EMBEDDING_LOCAL_ONLY=true"
)

# `env -i` clears the inherited env, which strips the user's INFERENCE_*
# / EMBEDDING_MODEL / RAG_* settings from .env.local. Forward those through
# explicitly so the gateway talks to the real MLX backends instead of mock.
# Per-run extras (passed to start_gateway) still win since they're appended last.
FORWARDED_ENV=()
ENV_LOCAL="${ROOT_DIR}/.env.local"
if [ -f "$ENV_LOCAL" ]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    case "$line" in
      INFERENCE_*|EMBEDDING_MODEL=*|RAG_*) FORWARDED_ENV+=("$line") ;;
    esac
  done < "$ENV_LOCAL"
fi

start_gateway() {
  local label="$1"; shift
  local extra=("$@")
  echo
  echo "═══ Boot: ${label} ═══"
  pkill -f "uvicorn app.main:app --host 127.0.0.1 --port ${PORT}" 2>/dev/null || true
  sleep 1
  env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="${ROOT_DIR}/gateway" \
    "${FORWARDED_ENV[@]}" \
    "${COMMON_ENV[@]}" "${extra[@]}" \
    "${VENV_PY}" -m uvicorn app.main:app \
      --host 127.0.0.1 --port "$PORT" --log-level warning \
      >"${GATEWAY_LOG}.${label}" 2>&1 &
  local pid=$!
  echo "  pid=$pid log=${GATEWAY_LOG}.${label}"
  for i in $(seq 1 20); do
    if curl -sf -m 1 "${GATEWAY}/health" >/dev/null 2>&1; then
      echo "  ready in ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "  TIMED OUT waiting for gateway"
  tail -20 "${GATEWAY_LOG}.${label}"
  return 1
}

run_baseline() {
  local label="$1"
  echo "─── Run: ${label} ───"
  "${VENV_PY}" run_baseline.py \
    --gateway "$GATEWAY" \
    --password "$PASSWORD" \
    --output "${RESULTS_DIR}/${label}.json" \
    --label "$label" \
    --skip "$SKIP" \
    || echo "  baseline run errored (continuing)"

  # Append a one-line summary
  "${VENV_PY}" - <<PY >>"${RESULTS_DIR}/summary.txt"
import json
d = json.load(open("${RESULTS_DIR}/${label}.json"))
s = d["summary"]
o = s["overall"]
lat = s["latency_seconds"]
fired = s["layer_activity"]["fired"]
flagged = s["layer_activity"]["flagged_when_fired"]
print(
    f"{d['label']:<14} pass={o['pass']}/{o['total']} ({o['pass_rate']:.1%}) "
    f"mean={lat['mean']}s p90={lat['p90']}s "
    f"cove(f={fired['cove']},+{flagged['cove']}) "
    f"judge(f={fired['judge']},+{flagged['judge']}) "
    f"sc(f={fired['selfcheck']},+{flagged['selfcheck']})"
)
PY
}

# ── Seed corpus (one-time, persists across all 4 configs) ─────────────────
# Without this, /tmp/molebie_ablation_data is empty → CoVe/Judge gates
# short-circuit on every query because no RAG chunks exist. The seeded
# docs cover the rag_grounded golden-set questions.
echo
echo "═══ Seed corpus ═══"
start_gateway "seed" \
  COVE_ENABLED=false JUDGE_ENABLED=false SELFCHECK_ENABLED=false
"${VENV_PY}" seed_corpus.py --gateway "$GATEWAY" --password "$PASSWORD" || {
  echo "  Seed failed; aborting"
  pkill -f "uvicorn app.main:app --host 127.0.0.1 --port ${PORT}" 2>/dev/null || true
  exit 1
}
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port ${PORT}" 2>/dev/null || true
sleep 2

# ── Run 1: baseline (no Phase 3) ──────────────────────────────────────────
start_gateway "baseline" \
  COVE_ENABLED=false JUDGE_ENABLED=false SELFCHECK_ENABLED=false
run_baseline "baseline"

# ── Run 2: judge-only ─────────────────────────────────────────────────────
start_gateway "judge" \
  COVE_ENABLED=false JUDGE_ENABLED=true SELFCHECK_ENABLED=false \
  JUDGE_MIN_RESPONSE_CHARS=100 JUDGE_THRESHOLD=0.3
run_baseline "judge"

# ── Run 3: cove-only ──────────────────────────────────────────────────────
start_gateway "cove" \
  COVE_ENABLED=true JUDGE_ENABLED=false SELFCHECK_ENABLED=false \
  COVE_MIN_RESPONSE_CHARS=150
run_baseline "cove"

# ── Run 4: all-on ─────────────────────────────────────────────────────────
start_gateway "all" \
  COVE_ENABLED=true JUDGE_ENABLED=true SELFCHECK_ENABLED=true \
  COVE_MIN_RESPONSE_CHARS=150 JUDGE_MIN_RESPONSE_CHARS=100 \
  JUDGE_THRESHOLD=0.3 SELFCHECK_MIN_RESPONSE_CHARS=100 SELFCHECK_SAMPLES=3
run_baseline "all"

# Cleanup
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port ${PORT}" 2>/dev/null || true
echo
echo "═══ Summary ═══"
cat "${RESULTS_DIR}/summary.txt"
