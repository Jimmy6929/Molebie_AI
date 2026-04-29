# Hallucination eval (Phase 2 task 2.1)

This directory holds the regression suite for hallucination behaviour.
Two tools, run side by side:

- **Promptfoo** — bespoke pass/fail rules per query (great for must-abstain
  and citation-format checks). Local, no telemetry. `npm install -g promptfoo`.
- **lm-eval-harness** — standardised academic benchmarks (TruthfulQA,
  MMLU-Pro, etc.). `pip install lm-eval`.

## Files

| File | Purpose |
|------|---------|
| `golden_set.jsonl`   | Hand-curated regression queries + assertions. Expand this. |
| `promptfoo.yaml`     | Promptfoo config that loads `golden_set.jsonl` and runs against the local gateway / Ollama / vLLM. |
| `run_eval.sh`        | One-shot runner: Promptfoo + lm-eval baseline. |

## Workflow

1. **Pre-fill the golden set.** The shipped file has 4 example questions
   per category as placeholders. Replace them with questions about *your*
   notes / domain — that's the only way the eval signal is real.
2. **Boot a local inference server** (see `gateway/configs/`).
3. **Boot the gateway** so RAG is wired in. Promptfoo can hit either the
   raw inference endpoint or the gateway's `/chat` for a full-stack test.
4. **Run** `./run_eval.sh`. Failures print to stdout; results to
   `eval-results.json`.
5. **Set thresholds** once you have a baseline number. The build plan's
   defaults (TruthfulQA -2pp, faithfulness <0.85 fail, abstention <0.7
   fail) are a starting point — tune to your numbers.

## Categories in the golden set

| Category               | Count to aim for | What it tests |
|------------------------|------------------|---------------|
| `rag_grounded`         | 15+              | answer + citations land correctly |
| `adversarial_premise`  | 15+              | model refuses fake-package / fake-API questions |
| `must_abstain`         | 10+              | `"I don't have that in your notes."` fires |
| `tool_call`            | 10+              | (Phase 2 task 2.2) tool name + args match |

The shipped placeholders cover 4 questions per category — enough to wire
the framework but not enough to *trust* the numbers. Treat baseline runs
on the placeholder set as smoke tests, not signal.
