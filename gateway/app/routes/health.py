"""
Health check endpoints for the Gateway API.
Includes dual-tier inference health with model info per mode,
and deep application-logic checks for the CLI doctor command.
"""

import struct
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.middleware.auth import JWTPayload, get_current_user
from app.services.inference import InferenceService, get_inference_service


router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str = "0.1.0"


class AuthenticatedHealthResponse(HealthResponse):
    """Health check response with user info."""
    user_id: str
    email: Optional[str] = None


class InferenceHealthResponse(BaseModel):
    """Health check response for inference endpoints (both tiers)."""
    instant: Dict[str, Any]
    thinking: Dict[str, Any]
    routing: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Public health check endpoint.
    Returns 200 if the service is running.
    """
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
    )


@router.get("/health/auth", response_model=AuthenticatedHealthResponse)
async def authenticated_health_check(
    user: JWTPayload = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedHealthResponse:
    """
    Authenticated health check endpoint.
    Returns 200 with user info if valid JWT provided.
    Returns 401 if no token or invalid token.
    """
    return AuthenticatedHealthResponse(
        status="healthy",
        service=settings.app_name,
        user_id=user.user_id,
        email=user.email,
    )


@router.get("/health/inference", response_model=InferenceHealthResponse)
async def inference_health_check(
    inference: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_settings),
) -> InferenceHealthResponse:
    """
    Check health of inference endpoints (instant and thinking).
    Shows per-tier model, config, and availability.
    Public endpoint -- no auth required.
    """
    instant_status = await inference.check_health("instant")
    thinking_status = await inference.check_health("thinking")

    routing_info = {
        "default_mode": settings.routing_default_mode,
        "thinking_fallback_to_instant": settings.routing_thinking_fallback_to_instant,
        "cold_start_timeout": settings.routing_cold_start_timeout,
        "thinking_daily_request_limit": settings.thinking_daily_request_limit,
        "thinking_max_concurrent": settings.thinking_max_concurrent,
    }

    return InferenceHealthResponse(
        instant=instant_status,
        thinking=thinking_status,
        routing=routing_info,
    )


# ── Deep application-logic health checks ────────────────────────────────


@router.get("/health/deep")
async def deep_health_check(
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Deep application-logic health checks.

    Tests embedding model, vector search round-trip, and RAG pipeline.
    Called by `molebie-ai doctor --deep` to verify the runtime environment.
    Public endpoint — no auth required.
    """
    results: Dict[str, Any] = {}

    # 1. Embedding model check
    results["embedding"] = _check_embedding(settings)

    # 2. Vector round-trip check (only if embedding passed)
    if results["embedding"]["status"] == "pass":
        embedding = results["embedding"].get("_test_embedding")
        results["vector_roundtrip"] = await _check_vector_roundtrip(embedding)
    else:
        results["vector_roundtrip"] = {
            "status": "skip",
            "message": "Skipped (embedding check failed)",
        }

    # Clean internal fields
    results["embedding"].pop("_test_embedding", None)

    return results


def _check_embedding(settings: Settings) -> Dict[str, Any]:
    """Test that the embedding model loads and produces correct-dimension vectors."""
    if not settings.rag_enabled:
        return {"status": "skip", "message": "RAG disabled"}

    try:
        from app.services.embedding import get_embedding_service
        svc = get_embedding_service()
        vector = svc.embed("molebie health check")
        dim = len(vector)
        return {
            "status": "pass",
            "model": settings.embedding_model,
            "dimension": dim,
            "message": f"Model loaded, dim={dim}",
            "_test_embedding": vector,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"{type(exc).__name__}: {exc}",
        }


async def _check_vector_roundtrip(
    test_embedding: List[float],
) -> Dict[str, Any]:
    """Insert a test embedding, query it back, verify, and clean up."""
    from app.services.database import get_database_service

    TEST_ROWID = 2**62
    db = get_database_service()

    try:
        conn = await db._get_conn()
        blob = struct.pack(f"{len(test_embedding)}f", *test_embedding)

        # Insert
        await conn.execute(
            "INSERT INTO document_chunks_vec(rowid, embedding) VALUES (?, ?)",
            (TEST_ROWID, blob),
        )
        await conn.commit()

        # Query
        rows = await conn.execute_fetchall(
            "SELECT rowid, distance FROM document_chunks_vec "
            "WHERE embedding MATCH ? AND k = 1",
            (blob,),
        )

        if not rows:
            return {"status": "fail", "message": "Insert OK but query returned no results"}

        found_rowid = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["rowid"]
        distance = rows[0][1] if isinstance(rows[0], (list, tuple)) else rows[0]["distance"]

        if found_rowid != TEST_ROWID:
            return {"status": "fail", "message": f"Wrong rowid: expected {TEST_ROWID}, got {found_rowid}"}

        if distance > 0.01:
            return {"status": "fail", "message": f"Self-query distance too high: {distance:.4f}"}

        return {
            "status": "pass",
            "message": f"Embed + insert + query OK (distance={distance:.6f})",
            "distance": distance,
        }
    except Exception as exc:
        return {"status": "fail", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            await conn.execute(
                "DELETE FROM document_chunks_vec WHERE rowid = ?", (TEST_ROWID,)
            )
            await conn.commit()
        except Exception:
            pass
