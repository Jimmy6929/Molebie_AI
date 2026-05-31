"""
AI Assistant Gateway - FastAPI Application

This is the main entry point for the Gateway API that sits between
the web app and inference endpoints.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

# Suppress asyncio "socket.send() raised exception" warnings that occur
# when the frontend disconnects mid-stream (harmless — data already dropped)
logging.getLogger("asyncio").setLevel(logging.WARNING)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import (
    auth,
    chat,
    documents,
    fleet,
    folder_ingest,
    health,
    metrics,
    vault,
)


async def _has_any_satellites() -> bool:
    """Quick existence check on the fleet_satellites table."""
    from app.services.database import get_database_service
    db = get_database_service()
    conn = await db._get_conn()
    rows = await conn.execute_fetchall(
        "SELECT 1 FROM fleet_satellites LIMIT 1"
    )
    return bool(rows)


async def _maybe_enable_tailscale_serve() -> None:
    """If any satellites are registered and Tailscale Serve isn't yet
    proxying our :8000, enable it and emit a `security.tls_enabled` audit
    event. Idempotent — already-configured is a no-op."""
    from app.services.tailscale_serve import (
        enable_serve,
        get_https_url,
        is_serve_configured,
    )

    if not await _has_any_satellites():
        return
    if is_serve_configured():
        return

    ok = enable_serve()
    if not ok:
        print(
            "[security] tailscale serve auto-enable failed — ensure HTTPS "
            "Certificates are enabled for this tailnet in the Tailscale "
            "admin console"
        )
        return

    url = get_https_url()
    print(f"[security] tailscale serve enabled: {url or '(URL unknown)'}")
    from app.services.audit import record as audit_record
    await audit_record(
        "security.tls_enabled",
        actor="system",
        target=url or "unknown",
        metadata={"port": 8000},
    )


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
    storage_scheduler = None
    try:
        from app.services.system_probe import get_system_probe
        system_probe = get_system_probe()
        await system_probe.start()
    except Exception as exc:
        print(f"[monitor] System probe unavailable: {exc}")
    try:
        from app.services.backend_probe import get_backend_probe
        # The probe resolves the pool list lazily via the inference service's
        # selector — no need to thread settings through anymore.
        backend_probe = get_backend_probe()
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
        from app.services.storage_scheduler import get_storage_scheduler
        storage_scheduler = get_storage_scheduler(settings.data_dir)
        await storage_scheduler.start(settings.storage_auto_migrate_interval_sec)
    except Exception as exc:
        print(f"[fleet] Storage migration scheduler unavailable: {exc}")

    # Resume any folder-ingest jobs that were interrupted by a restart.
    if getattr(settings, "folder_ingest_enabled", False):
        from app.services.ingest_worker import get_ingest_worker
        asyncio.create_task(get_ingest_worker().resume_running_jobs())

    # ── Tailscale Serve auto-bootstrap ────────────────────────────
    # Idempotent every boot. Only kicks in when at least one satellite
    # has registered — no point burning a Let's Encrypt cert when there's
    # nothing to expose. Operators who manage `tailscale serve` themselves
    # can set MOLEBIE_AUTO_TAILSCALE_SERVE=0 to disable.
    if os.getenv("MOLEBIE_AUTO_TAILSCALE_SERVE", "1") == "1":
        try:
            await _maybe_enable_tailscale_serve()
        except Exception as exc:
            print(f"[security] tailscale serve bootstrap skipped: {exc}")

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
        if storage_scheduler is not None:
            await storage_scheduler.stop()
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
    app.include_router(folder_ingest.router)
    app.include_router(vault.router)
    app.include_router(fleet.router)

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
