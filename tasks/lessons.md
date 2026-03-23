# Lessons Learned

_Updated as corrections and patterns emerge. Review at session start._

## Project Rules
- **Small models only**: Never suggest upgrading from Qwen 3.5 9B/4B. All improvements come from system design.
- **Autonomous fixing**: When given a bug, investigate and fix it independently. Don't ask the user how to debug.
- **Plan first**: Non-trivial tasks (3+ steps) get a plan before implementation.

## Bug Patterns
- **MLX "socket.send() raised exception" flood**: The error comes from MLX's process (asyncio/selector_events.py), NOT the gateway. MLX's generation loop doesn't stop on client disconnect — asyncio's transport silently drops writes without raising to Starlette, so generation continues until max_tokens. Gateway-side fixes (cancel_event, task+queue, response.aclose) don't help because the errors originate in MLX. Fix: suppress asyncio warnings in MLX via a wrapper script (`scripts/mlx_server.py`). Don't over-engineer the gateway side — simple `async for` with `is_disconnected()` is sufficient.
