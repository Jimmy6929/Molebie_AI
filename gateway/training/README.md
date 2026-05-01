# Phase 4 — Fine-tuning data pipeline + training wrappers

This directory ships everything *up to* training the LoRA. The actual
training has to run on your GPU. Two paths covered:

- **Apple Silicon (M3/M4 Max 64GB+):** `train_mlx.sh` → mlx-lm LoRA
- **Linux + CUDA (RTX 4090 24GB+):** `train_unsloth.sh` → Unsloth bf16 LoRA

## ⚠ The most important rule from the build plan

> **Fine-tuning teaches style, not facts.** Training on new factual
> knowledge linearly increases hallucination rate (Gekhman, EMNLP 2024).
> Keep facts in RAG, keep behavior in the LoRA.

Every script here exists to enforce that. The data pipeline:

1. Probes the *current* model behavior on a fixed question set.
2. Heuristically labels each response as grounded / abstention /
   fabrication.
3. Replaces fabricated responses with the canonical abstention text.
4. Builds SFT pairs and ORPO triples for behavior-only training.
5. Mixes in 25–50% reasoning-replay data (FineTome-100k +
   OpenMathReasoning) to prevent catastrophic forgetting.
6. After training, runs the eval baseline before/after and *enforces*
   the 2pp regression rule — discard the LoRA if any metric drops.

## Files

| File | Purpose |
|------|---------|
| `probe.py`             | Probe runner. Reuses `tests/eval/run_baseline.py`'s chat helper. |
| `probe_set.jsonl`      | 75 seed questions across 15 categories (fake APIs, fake papers, obscure facts, well-known facts, tool-use, etc.). |
| `label.py`             | Heuristic labelling + canonical-abstention rewriter. |
| `build_dataset.py`     | Probe records → SFT pairs + ORPO triples + replay mix manifest. |
| `hyperparams.yaml`     | Build-plan hyperparameters (LoRA r=16, lr=2e-4, 1 epoch, bf16, etc.) + eval gates. |
| `train_mlx.sh`         | Apple Silicon wrapper around mlx-lm canonical lora.py. |
| `train_unsloth.sh`     | Linux+CUDA wrapper around Unsloth + trl.SFTTrainer / ORPOTrainer. |
| `eval_lora.py`         | Diff two `run_baseline.py` reports + apply the build-plan gates. |
| `tests/`               | 28 pytest unit tests for label/build/eval pipeline. |

## End-to-end workflow

### 0. Prereqs (one-time)

```bash
# Apple Silicon
pip install -U mlx-lm pyyaml
huggingface-cli download Qwen/Qwen3.5-4B-Instruct          # 4B is the realistic
                                                            # path on M3/M4 Max

# Linux + CUDA
pip install -U "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install -U trl peft accelerate datasets pyyaml bitsandbytes
huggingface-cli download Qwen/Qwen3.5-9B-Instruct          # 9B fits 24GB bf16
```

Capture your **pre-tuning baseline** with all Phase 3 layers OFF:

```bash
# in one shell — boot gateway with COVE/JUDGE/SELFCHECK all false
make dev-gateway

# in another
cd gateway/tests/eval
python run_baseline.py \
    --gateway http://localhost:8000 \
    --password $PASSWORD \
    --output before-tuning.json \
    --label "phase3-all-off-baseline"
```

### 1. Probe — collect current model traces

Boot the gateway with **all Phase 3 layers OFF** (we want the base
behaviour the LoRA will improve, not the post-processed behaviour):

```bash
COVE_ENABLED=false JUDGE_ENABLED=false SELFCHECK_ENABLED=false make dev-gateway
```

Then probe:

```bash
cd gateway/training
python probe.py \
    --gateway http://localhost:8000 \
    --password $PASSWORD \
    --output probes/seed-$(date +%F).jsonl
```

Output: a JSONL with one line per query containing the response, the
inference metadata, and a heuristic first-pass label. The runner prints
a label histogram on completion — if `likely_fabrication` is < 20% of
the set, your probe set isn't eliciting enough hallucinations to be
useful training data. Add harder questions to `probe_set.jsonl`.

### 2. Label — relabel fabrications

```bash
python label.py \
    --input probes/seed-$(date +%F).jsonl \
    --output probes/seed-$(date +%F).labelled.jsonl
```

Default policy is `--fabrication-strict` (any uncited specific fact
without refusal = fabrication). For fewer-but-stronger samples:

```bash
python label.py --require-layer-flag ...
```

This requires Phase 3 layers (CoVe / Judge / SelfCheck) to also have
flagged the response. To use this you need probes captured with layers
ON — re-run `probe.py` after flipping them on.

### 3. Build dataset

```bash
python build_dataset.py \
    --labelled probes/seed-$(date +%F).labelled.jsonl \
    --out-dir datasets/v1/
```

Produces:

- `datasets/v1/sft.jsonl` — chat-format SFT pairs
- `datasets/v1/orpo.jsonl` — `(prompt, chosen, rejected)` triples
- `datasets/v1/replay_required.txt` — exact replay mix to construct

### 4. Construct replay mix (one-time)

The build plan requires ≥75% reasoning-heavy content overall to
preserve thinking-mode behavior. From `replay_required.txt`:

```bash
huggingface-cli download mlabonne/FineTome-100k
huggingface-cli download nvidia/OpenMathReasoning
# Sample N records from each per replay_required.txt and write them
# in the same chat-template format as datasets/v1/sft.jsonl. Concatenate.
```

(This step is intentionally manual — the right replay split is
domain-specific, and we don't want to bundle 100k records in the repo.)

### 5. Stage 1: R-Tuning SFT

**Apple Silicon:**

```bash
./train_mlx.sh sft datasets/v1/sft_with_replay.jsonl
# saves to adapters/sft/
```

**Linux + CUDA:**

```bash
./train_unsloth.sh sft datasets/v1/sft_with_replay.jsonl
# saves to adapters/sft/
```

Build plan reference latency: ~12 min for 600 iters on Qwen3.5-4B
4-bit on M3/M4 Max. RTX 4090 fits 9B bf16 LoRA at ~22 GB used.

### 6. Stage 2: ORPO factuality

Continue from the SFT adapter:

```bash
./train_mlx.sh orpo datasets/v1/orpo.jsonl adapters/sft
# or
./train_unsloth.sh orpo datasets/v1/orpo.jsonl adapters/sft
# saves to adapters/orpo/
```

Expected build-plan signal: ORPO with grounded-vs-fabricated pairs cuts
hallucination ~5× on Qwen3-8B (TruthfulQA MC1 +17%). Don't expect that
exact number — your data isn't theirs — but anything *less* than +5%
faithfulness improvement is suspicious.

### 7. Eval after — diff against the baseline

Restart inference with the LoRA loaded:

```bash
# mlx-lm
mlx_lm.server --model Qwen/Qwen3.5-4B-Instruct \
              --adapter-path adapters/orpo --port 8081

# Unsloth users export to a merged or vLLM-compatible format and run
# their server of choice; the gateway calls a plain OpenAI-compatible
# endpoint either way.
```

Then eval:

```bash
cd gateway/tests/eval
python run_baseline.py \
    --gateway http://localhost:8000 \
    --password $PASSWORD \
    --output after-tuning.json \
    --label "post-orpo"

# Diff + apply gates
cd ../../training
python eval_lora.py \
    --before ../tests/eval/before-tuning.json \
    --after ../tests/eval/after-tuning.json \
    --gates hyperparams.yaml \
    --output verdict.json
```

### 8. Keep or discard

`eval_lora.py` prints `KEEP` or `DISCARD` based on the build-plan
gates:

- TruthfulQA MC2 < baseline − 2pp → fail
- Faithfulness (rag_grounded pass rate) < 0.85 → fail
- Refusal rate (must_abstain pass rate) < 0.7 → fail
- Any category dropping > 2pp → fail
- Overall pass rate dropping > 2pp → fail

If the verdict is `DISCARD`, **do not deploy the adapter**. Investigate:

- Did training actually converge (check loss curve)?
- Did the replay data dominate (raise replay fraction)?
- Did the LoRA learn to abstain on RAG queries it shouldn't (over-
  abstention — too many `proper_abstention` SFT pairs)?
- Did batch size / lr cause instability?

The reject rule is *non-negotiable* per the build plan — a LoRA that
drops any metric is doing more harm than good.

## Hyperparameter tuning

`hyperparams.yaml` is the source of truth. Both training wrappers
read it. Common tweaks:

- **More epochs:** don't. Build plan: NEVER > 2.
- **Higher LR:** stay below 5e-4 for SFT, 1e-4 for ORPO. Past that
  ORPO becomes unstable on Qwen3.5.
- **Larger LoRA rank:** r=32 buys little over r=16 for behaviour-only
  training. Try only after r=16 hits the gates and you want sharper.
- **More replay:** raise to 50% if the LoRA over-abstains on RAG.

## Hardware paths — picking one

| Hardware | Backend | Model size | Why |
|----------|---------|------------|-----|
| M3/M4 Max 64 GB | mlx-lm 4-bit LoRA | Qwen3.5-4B | Build-plan reference setup; ~12 min/600 iters |
| RTX 4090 24 GB | Unsloth bf16 LoRA | Qwen3.5-9B | Bigger model, more headroom; ~22 GB used |
| RTX 3090 / smaller | Unsloth bf16 LoRA | Qwen3.5-4B | bf16 LoRA on 4B fits ~12 GB; cut max_seq_length to 1024 if tight |
| Mac < 64 GB | not recommended | — | mlx-lm 4-bit on 4B works but training is painfully slow |

Build-plan rule: **NEVER use 4-bit QLoRA on Qwen3.5.** Unsloth itself
recommends against it for this model family. Use bf16 LoRA instead.

## Where this fits in the bigger picture

- Phase 1 (foundation), Phase 2 (hardening), Phase 3 (post-processors)
  are *runtime* work — they sit in the chat path and inspect / annotate
  responses.
- Phase 4 (this directory) is *training* work — it changes the base
  model's behaviour so future responses need less post-processing.

If Phase 1-3 already meets your faithfulness target on the eval
golden set, **don't fine-tune**. RAG + post-processors compose
cleanly; a bad LoRA quietly degrades both. Wait for the
ablation-interpretation report (or capture one yourself) before
deciding.
