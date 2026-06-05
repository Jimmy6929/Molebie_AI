"""``molebie-satellite serve`` — run the FastAPI blob service in the foreground.

Also the implementation behind ``python -m satellite_storage`` so both
entry points share one code path.
"""

from __future__ import annotations


def run() -> None:
    """Boot uvicorn against the satellite_storage app with config-driven host/port."""
    import uvicorn

    from satellite_storage.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "satellite_storage.app:app",
        host=settings.bind_host,
        port=settings.bind_port,
        reload=False,
    )


def serve_command() -> None:
    """Start the satellite blob service in the foreground."""
    run()
