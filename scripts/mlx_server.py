#!/usr/bin/env python3
"""Start mlx_vlm.server with asyncio socket-disconnect warnings suppressed.

MLX's generation loop doesn't stop when a client disconnects mid-stream.
The asyncio transport logs 'socket.send() raised exception.' for every
remaining token — flooding the terminal and making Ctrl+C unresponsive.
These warnings are harmless (the data is already being dropped), so we
silence them while preserving all other MLX logging.

Additionally, MLX hardcodes reload=True in its uvicorn.run() call, which
spawns a subprocess that wouldn't inherit our logging config. We patch
uvicorn.run to disable reload so everything stays in-process.
"""
import logging

# Suppress asyncio transport warnings (socket.send() raised exception)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# MLX's main() calls uvicorn.run(..., reload=True). reload=True spawns a
# subprocess that loses our logging config. Patch it to run in-process.
import uvicorn

_original_uvicorn_run = uvicorn.run


def _patched_run(*args, **kwargs):
    kwargs["reload"] = False
    return _original_uvicorn_run(*args, **kwargs)


uvicorn.run = _patched_run

print("[mlx_server] Socket warning suppression active")

from mlx_vlm.server import main

main()
