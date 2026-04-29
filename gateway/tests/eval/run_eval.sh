#!/usr/bin/env bash
# One-shot eval runner: Promptfoo (custom rules) + lm-eval-harness (baselines).
#
# Prereqs:
#   * promptfoo on PATH         (`npm install -g promptfoo`)
#   * lm-eval on PATH           (`pip install lm-eval`)
#   * Local inference server running (Ollama/vLLM/etc.)
#   * Optional: gateway running on :8000 with JWT in $BEARER
#
# Outputs:
#   * eval-results.json         (Promptfoo)
#   * lm-eval-results-*.json    (lm-eval, one per task)
set -euo pipefail

cd "$(dirname "$0")"

INFERENCE_HOST="${INFERENCE_HOST:-http://localhost:11434}"
INFERENCE_MODEL="${INFERENCE_MODEL:-qwen3.5:9b}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-truthfulqa_mc2}"   # add: mmlu_pro,gsm8k once baselined

echo "=== 1/2  Promptfoo (golden_set.jsonl) ============================"
if command -v promptfoo >/dev/null; then
  promptfoo eval -c promptfoo.yaml --no-cache --output eval-results.json || true
else
  echo "[skip] promptfoo not installed — run: npm install -g promptfoo"
fi

echo
echo "=== 2/2  lm-eval-harness ($LM_EVAL_TASKS) ========================="
if command -v lm_eval >/dev/null; then
  lm_eval \
    --model local-completions \
    --model_args "model=${INFERENCE_MODEL},base_url=${INFERENCE_HOST}/v1/completions" \
    --tasks "$LM_EVAL_TASKS" \
    --batch_size 1 \
    --output_path lm-eval-results
else
  echo "[skip] lm-eval not installed — run: pip install lm-eval"
fi

echo
echo "Done. See eval-results.json (promptfoo) and lm-eval-results/ (lm-eval)."
