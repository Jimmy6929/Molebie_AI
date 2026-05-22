"""FastAPI factory + lifespan for the satellite blob service.

Tiny lifespan — log startup, ensure the data dir exists. No probes, no
background tasks, no database. The satellite's job is to be a passive
blob store; everything stateful lives in the filesystem layout.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from satellite_storage import __version__
from satellite_storage.config import get_settings
from satellite_storage.routes import capacity, health, storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[molebie-storage] v{__version__} starting")
    print(f"[molebie-storage]   data_dir: {settings.data_dir}")
    print(f"[molebie-storage]   bind:     {settings.bind_host}:{settings.bind_port}")
    try:
        yield
    finally:
        print("[molebie-storage] shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="molebie-storage",
        description="Satellite-side content-addressable blob storage for Molebie AI",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(storage.router)
    app.include_router(capacity.router)
    app.include_router(health.router)
    return app


app = create_app()
