#!/usr/bin/env bash
# Apple Silicon training wrapper — primary path for M3/M4 Max owners.
#
# This script does NOT ship its own training code. It calls mlx-lm's
# canonical `lora.py` with hyperparameters from hyperparams.yaml. Why:
# rewriting the trainer in-tree would mean shadowing mlx-lm's
# fast-evolving optimisations (gradient checkpointing improvements,
# new quantisation paths) every time upstream lands a fix.
#
# Prereqs:
#   pip install -U mlx-lm pyyaml
#   huggingface-cli download Qwen/Qwen3.5-4B-Instruct  # or 9B
#
# Usage:
#   ./train_mlx.sh sft   path/to/sft.jsonl
#   ./train_mlx.sh orpo  path/to/orpo.jsonl   adapters/sft
#
# Build-plan reference: ~12 min for 600 iters on Qwen3.5-4B at
# 4-bit on M3/M4 Max. Raise --iters cautiously — 1 epoch is the limit.
set -euo pipefail

cd "$(dirname "$0")"

stage="${1:-}"
data_file="${2:-}"
base_adapter="${3:-}"

if [ -z "$stage" ] || [ -z "$data_file" ]; then
  echo "Usage: $0 {sft|orpo} <data.jsonl> [base_adapter_dir]"
  exit 1
fi

# Pull the relevant section from hyperparams.yaml. mlx-lm reads YAML
# directly via --config; we extract the section we need with python so
# the wrapper stays dep-light (only PyYAML).
section_yaml="$(python3 - <<PY
import yaml, sys, json
with open("hyperparams.yaml") as f:
    h = yaml.safe_load(f)
section = h["${stage}"]
# mlx-lm's lora.py wants flat kwargs — flatten.
flat = {
    "model": section.get("base_model") or h["sft"]["base_model"],
    "data": "${data_file}",
    "lora_layers": section.get("lora", h["sft"]["lora"]).get("r", 16),
    "iters": 600,    # tune; build plan default 1 epoch
    "batch_size": section["batch"]["per_device_train_batch_size"],
    "learning_rate": section["optim"]["learning_rate"],
    "val_batches": 25,
    "steps_per_eval": 50,
    "save_every": 200,
    "adapter_path": section["adapter_dir"],
}
print(yaml.safe_dump(flat))
PY
)"

echo "─── mlx-lm LoRA / ${stage} ───"
echo "$section_yaml"

# Write a transient config file so mlx-lm picks up everything atomically.
cfg=$(mktemp -t mlx-lora.XXXXXX.yaml)
trap 'rm -f "$cfg"' EXIT
echo "$section_yaml" > "$cfg"

# Resume from the SFT adapter when running ORPO.
resume_arg=""
if [ "$stage" = "orpo" ] && [ -n "$base_adapter" ]; then
  resume_arg="--resume-adapter-file ${base_adapter}/adapters.safetensors"
  echo "Resuming from $base_adapter"
fi

# Hand off to mlx-lm. We deliberately don't catch errors here — let the
# trainer's own diagnostics surface.
python -m mlx_lm.lora \
  --config "$cfg" \
  --train \
  $resume_arg

echo "Adapter saved per hyperparams.yaml -> ${section}.adapter_dir"
