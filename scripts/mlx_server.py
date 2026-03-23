#!/usr/bin/env python3
"""Start mlx_vlm.server with asyncio socket-disconnect warnings suppressed.

MLX's generation loop doesn't stop when a client disconnects mid-stream.
The asyncio transport logs 'socket.send() raised exception.' for every
remaining token — flooding the terminal and making Ctrl+C unresponsive.
These warnings are harmless (the data is already being dropped), so we
silence them while preserving all other MLX logging.
"""
import logging
import runpy

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

runpy.run_module("mlx_vlm.server", run_name="__main__")
