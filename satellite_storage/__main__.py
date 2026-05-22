"""Entry point for ``python -m satellite_storage``.

Starts the FastAPI app via uvicorn. Honors MOLEBIE_STORAGE_PORT and
MOLEBIE_STORAGE_DATA_DIR. Matches the gateway's invocation shape so
operators have one mental model for "run this Molebie service".
"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    from satellite_storage.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "satellite_storage.app:app",
        host=settings.bind_host,
        port=settings.bind_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
