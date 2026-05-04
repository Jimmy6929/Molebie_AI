# Hallucination eval (Phase 2 task 2.1, expanded for Phase 4 baseline)

This directory holds the regression suite for hallucination behaviour
plus a layer-ablation runner that captures Phase 1–3's contribution
on top of vanilla inference. Two complementary tools, run side by side:

- **Promptfoo** — bespoke pass/fail rules per query (great for must-
  abstain and citation-format checks). Local, no telemetry.
  `npm install -g promptfoo`.
- **`run_baseline.py`** — Python runner that hits the gateway's
  `/chat` endpoint and captures latency + every Phase 3 layer's
  activity (CoVe verdicts, Judge verdicts, SelfCheck flags,
  citation-validity report). Outputs JSON. This is what feeds the
  Phase 4 fine-tuning before/after comparison.
- **lm-eval-harness** — standardised academic benchmarks (TruthfulQA,
  MMLU-Pro, etc.). `pip install lm-eval`.

## Files

| File | Purpose |
|------|---------|
| `golden_set.jsonl`   | 50 hand-curated regression queries across 5 categories. Add 50–100 more from your actual notes for stronger signal. |
| `promptfoo.yaml`     | Promptfoo config that loads `golden_set.jsonl` and runs against the local gateway / Ollama / vLLM. |
| `run_baseline.py`    | **The Phase 4 baseline runner.** Per-category pass rate AND per-layer activity (CoVe applied %, Judge fired %, SelfCheck flagged %). |
| `ablation.sh`        | Boots the gateway four times (baseline / judge-only / cove-only / all-on), runs the baseline against each, drops a one-line summary into `eval-results/summary.txt`. |
| `run_eval.sh`        | One-shot Promptfoo + lm-eval baseline (kept for academic-benchmark coverage). |
| `test_runner.py`     | Pytest unit tests for `run_baseline.py`'s pure-function helpers (assertions, layer extraction, summary aggregation). |

## Workflow

### Capture a one-shot baseline

```bash
# in one shell
make dev-gateway     # or your equivalent boot
# in another
cd gateway/tests/eval
.venv/bin/python run_baseline.py \
  --gateway http://localhost:8000 \
  --password <single-user-password> \
  --output eval-results/baseline-$(date +%F).json \
  --skip tool_call         # skip tool-calls until backend supports them
```

### Capture a layer ablation

```bash
PASSWORD=<single-user-password> ./ablation.sh
```

Boots the gateway four times with different layer-flag combos, runs
the same 50 questions against each, and writes:

```
eval-results/baseline.json
eval-results/judge.json
eval-results/cove.json
eval-results/all.json
eval-results/summary.txt        ← one-line-per-run comparison
eval-results/gateway.log.<label> ← per-boot log
```

The `summary.txt` is the eyeball view:

```
baseline    pass=42/50 (84.0%) mean=4.8s  p90=8.2s   cove(f=0,+0)  judge(f=0,+0)   sc(f=0,+0)
judge       pass=43/50 (86.0%) mean=5.4s  p90=9.1s   cove(f=0,+0)  judge(f=18,+3)  sc(f=0,+0)
cove        pass=46/50 (92.0%) mean=12.1s p90=22.7s  cove(f=12,+4) judge(f=0,+0)   sc(f=0,+0)
all         pass=47/50 (94.0%) mean=13.4s p90=24.5s  cove(f=12,+4) judge(f=18,+3)  sc(f=8,+5)
```

(Numbers above are *illustrative* — real values come from running
ablation.sh on your hardware against your golden set.)

### Set thresholds once you have a baseline

The build plan defaults are a starting point — tune to your numbers:

- TruthfulQA MC2 — fail on regression > 2pp from baseline
- Faithfulness (rag_grounded pass rate) — fail < 0.85
- Refusal rate (must_abstain pass rate) — fail < 0.7
- Mean latency increase from full-stack vs baseline — fail > 3×

## Categories in the golden set

| Category               | Count shipped | What it tests |
|------------------------|---------------|---------------|
| `rag_grounded`         | 15            | Answer + `[S#]` citation lands correctly. |
| `adversarial_premise`  | 15            | Refuses fake-package / fake-API / fake-paper / fake-CVE. |
| `must_abstain`         | 10            | Emits the exact `"I don't have that in your notes."` (or accepted refusal). |
| `tool_call`            | 8             | Correct tool dispatched with sensible arguments. |
| `rag_grounded_negative`| 2             | Disclaims when retrieval is weak instead of fabricating. |

The shipped 50 are realistic but **generic** — replace ~half with
questions sourced from your actual notes for the most-trustworthy
signal. Replace once and reuse for every Phase-4 before/after run.

## Layer-by-layer expectations

When you add a new question, also note which layer should *catch it*
if the model fabricates:

- **Citation validator (Phase 2.4)** — uncited specific facts (numbers,
  dates, URLs)
- **Judge (Phase 3.2)** — citations to off-topic chunks
- **CoVe (Phase 3.1)** — wrong number/date in an otherwise-relevant chunk
- **SelfCheck (Phase 3.3)** — fabrications in *no-RAG* responses
  (general-knowledge questions where the model doesn't have references
  and tends to invent inconsistently across samples)

If a question can only be caught by one specific layer, label that in
the JSONL `notes` field — makes ablation runs interpretable.
