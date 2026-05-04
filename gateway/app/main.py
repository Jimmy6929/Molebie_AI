"""
AI Assistant Gateway - FastAPI Application

This is the main entry point for the Gateway API that sits between
the web app and inference endpoints.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

# Suppress asyncio "socket.send() raised exception" warnings that occur
# when the frontend disconnects mid-stream (harmless — data already dropped)
logging.getLogger("asyncio").setLevel(logging.WARNING)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import auth, chat, documents, health, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()

    # Initialize SQLite database on startup
    from app.schema import init_database
    db_path = await init_database(
        data_dir=settings.data_dir,
        embedding_dim=settings.embedding_dim,
        auth_mode=getattr(settings, "auth_mode", "single"),
    )

    print(f" Starting {settings.app_name}")
    print(f"   Database: {db_path}")
    print(f"   Auth mode: {getattr(settings, 'auth_mode', 'single')}")
    print(f"   Debug mode: {settings.debug}")
    print(f"   Instant tier: {settings.inference_instant_url or '(not configured — mock)'}")
    print(f"     Model: {settings.get_model_for_mode('instant')}")
    print(f"   Thinking tier: {settings.inference_thinking_url or '(not configured — mock/fallback)'}")
    print(f"     Model: {settings.get_model_for_mode('thinking')}")
    print(f"   Thinking fallback to instant: {settings.routing_thinking_fallback_to_instant}")
    print(f"   RAG enabled: {settings.rag_enabled}")
    if settings.rag_enabled:
        print(f"     Embedding model: {settings.embedding_model}")
        print(f"     Match count: {settings.rag_match_count}, threshold: {settings.rag_match_threshold}")
        print(f"     Hybrid search: {settings.rag_hybrid_enabled}")
        print(f"     Contextual retrieval: {settings.rag_contextual_retrieval_enabled}")
        print(f"     Reranker: {settings.rag_reranker_enabled} ({settings.rag_reranker_model})")
        if settings.embedding_preload:
            print("     Preloading embedding model in background...")
            from app.services.embedding import get_embedding_service
            def _preload_embedding():
                try:
                    get_embedding_service()._load_model()
                except (ImportError, RuntimeError) as exc:
                    print(f"[embedding] Preload skipped: {exc}")
            asyncio.create_task(asyncio.to_thread(_preload_embedding))
        if settings.rag_reranker_preload and settings.rag_reranker_enabled:
            print("     Preloading reranker model in background...")
            from app.services.reranker import get_reranker
            def _preload_reranker():
                try:
                    get_reranker()._load_model()
                except (ImportError, RuntimeError) as exc:
                    print(f"[reranker] Preload skipped: {exc}")
            asyncio.create_task(asyncio.to_thread(_preload_reranker))

    # ── Live monitor probes ───────────────────────────────────────
    # Start background pollers for /metrics/live. Degrade silently if a
    # probe can't start (e.g., psutil missing in some minimal envs) —
    # serving chat traffic is more important than the monitor feature.
    system_probe = None
    backend_probe = None
    storage_probe = None
    try:
        from app.services.system_probe import get_system_probe
        system_probe = get_system_probe()
        await system_probe.start()
    except Exception as exc:
        print(f"[monitor] System probe unavailable: {exc}")
    try:
        from app.services.backend_probe import get_backend_probe
        backend_probe = get_backend_probe(settings)
        await backend_probe.start()
    except Exception as exc:
        print(f"[monitor] Backend probe unavailable: {exc}")
    try:
        from app.services.storage_probe import get_storage_probe
        storage_probe = get_storage_probe(settings.data_dir)
        await storage_probe.start()
    except Exception as exc:
        print(f"[monitor] Storage probe unavailable: {exc}")

    try:
        yield
    finally:
        # Graceful cancel — stop() cancels the task and awaits it.
        if system_probe is not None:
            await system_probe.stop()
        if backend_probe is not None:
            await backend_probe.stop()
        if storage_probe is not None:
            await storage_probe.stop()
        print(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Gateway API for AI Assistant - handles auth, chat, and inference routing",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS — origins from CORS_ORIGINS env var + private IP regex for LAN access.
    # The regex covers RFC1918 private ranges and the 100.64.0.0/10 CGNAT block used by
    # Tailscale, so remote devices on the same tailnet can reach the gateway on :8000.
    # Each branch captures a 2-octet prefix; the trailing \.\d+\.\d+ matches the last two.
    cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"^http://(192\.168|172\.\d+|10\.\d+|100\.\d+)\.\d+\.\d+:3000$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID"],
    )

    # Observability middleware — records each request as a subsystem call so
    # the live monitor shows constant motion at every API hit. Skips
    # /metrics/live to avoid feedback noise (the monitor polls that route
    # itself at 2 Hz).
    from app.middleware.observability import ObservabilityMiddleware
    app.add_middleware(ObservabilityMiddleware)

    # Register routes
    app.include_router(auth.router)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(metrics.router)

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
