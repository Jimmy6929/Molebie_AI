# Lessons Learned

_Updated as corrections and patterns emerge. Review at session start._

## Project Rules
- **Small models only**: Never suggest upgrading from Qwen 3.5 9B/4B. All improvements come from system design.
- **Autonomous fixing**: When given a bug, investigate and fix it independently. Don't ask the user how to debug.
- **Plan first**: Non-trivial tasks (3+ steps) get a plan before implementation.

## Bug Patterns
- **SSE streaming + MLX disconnect**: When using `async for` to consume an httpx stream from MLX, the gateway only detects client disconnect after the next chunk arrives. This leaves the MLX connection open, causing `socket.send() raised exception` spam. Fix: use an `asyncio.Event` + background disconnect watcher that actively closes the httpx response via `response.aclose()`.
