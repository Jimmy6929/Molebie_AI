# Molebie sampling presets and thinking-mode budgets

Per-tier sampling parameters and reasoning budgets for the Qwen3.5
backends Molebie uses.

## Instant tier (Qwen3.5-4B)

- temperature: 0.7
- top_p: 0.8
- top_k: 20
- presence_penalty: 1.5 (hard ceiling — above this, Qwen3.5 mixes languages
  mid-response)
- repetition_penalty: 1.0
- max_tokens: 2048
- enable_thinking: false (default for the instant tier)

## Thinking tier (Qwen3.5-9B)

- temperature: 0.6
- top_p: 0.95
- top_k: 20
- presence_penalty: 0.0 (penalties hurt thinking-mode reasoning)
- repetition_penalty: 1.0
- max_tokens: 24576
- thinking budget: 2048 tokens (the model is told to wrap up reasoning by then)
- enable_thinking: true

## Thinking-harder tier

Same as thinking tier, but with:

- max_tokens: 28672
- thinking budget: 8192 tokens

## Why the cap of 1.5 on presence_penalty

Empirically, anything above 1.5 on Qwen3.5 starts producing hybrid
English-Chinese-Japanese tokens mid-response. The hard ceiling is
enforced in `gateway/app/config.py` regardless of what the operator
sets in `.env.local`.
