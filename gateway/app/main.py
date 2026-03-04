"""
AI Assistant Gateway - FastAPI Application

This is the main entry point for the Gateway API that sits between
the web app and inference endpoints.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import health, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()
    print(f" Starting {settings.app_name}")
    print(f"   Supabase URL: {settings.supabase_url}")
    print(f"   Debug mode: {settings.debug}")
    print(f"   Instant tier: {settings.inference_instant_url or '(not configured — mock)'}")
    print(f"     Model: {settings.get_model_for_mode('instant')}")
    print(f"   Thinking tier: {settings.inference_thinking_url or '(not configured — mock/fallback)'}")
    print(f"     Model: {settings.get_model_for_mode('thinking')}")
    print(f"   Thinking fallback to instant: {settings.routing_thinking_fallback_to_instant}")
    yield
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
    
    # Configure CORS for local + LAN development
    # (webapp may be accessed via localhost or the server's LAN IP)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://100.99.189.104:3000",
        ],
        allow_origin_regex=r"^http://192\.168\.\d+\.\d+:3000$|^http://172\.\d+\.\d+\.\d+:3000$|^http://10\.\d+\.\d+\.\d+:3000$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID"],
    )
    
    # Register routes
    app.include_router(health.router)
    app.include_router(chat.router)
    
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
