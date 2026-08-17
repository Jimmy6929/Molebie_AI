"""
Embedding service for RAG vector search.

Uses sentence-transformers to generate embeddings locally on CPU/Apple Silicon.
Dimension configured via EMBEDDING_MODEL in .env.
"""


import asyncio
import threading

from app.config import Settings, get_settings


class EmbeddingService:
    """Generate text embeddings using a local sentence-transformers model."""

    def __init__(self, settings: Settings):
        # One lock for ALL query-time embedding (T6b) — see embed_async.
        # A threading.Lock taken INSIDE the worker thread, not an asyncio
        # lock: if a caller is cancelled (memory's retrieval timeout), the
        # to_thread worker cannot be cancelled and keeps running — an
        # asyncio lock would release at the await point and let the next
        # caller hit the model concurrently with the zombie. The thread
        # lock serializes at the model boundary regardless of cancellation.
        self._embed_thread_lock = threading.Lock()
        self.model_name = settings.embedding_model
        self.local_files_only = settings.embedding_local_only
        self._expected_dim = settings.embedding_dim
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        if self._model is not None:
            return

        # Set offline env vars BEFORE importing sentence_transformers.
        # huggingface_hub reads HF_HUB_OFFLINE at import time — if we set
        # it after the import, the library already decided it's online and
        # custom model code (trust_remote_code) will try network calls.
        if self.local_files_only:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "RAG requires PyTorch and sentence-transformers — "
                "install with: pip install -r requirements-ml.txt"
            )

        print(f"[embedding] Loading model: {self.model_name} (local_files_only={self.local_files_only})")
        try:
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except OSError as exc:
            if self.local_files_only:
                raise RuntimeError(
                    f"Embedding model '{self.model_name}' not found in local cache. "
                    "Run once with internet to download, then offline use will work."
                ) from exc
            raise
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"[embedding] Model loaded — dim={self._dimension}")
        # Guard against silent vec-table dimension mismatches: if the model
        # outputs a different dim than the schema expects, sqlite-vec
        # rejects every query later with "Dimension mismatch for query
        # vector". Fail loud at load time with a fix-it pointer instead.
        if self._dimension != self._expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: model "
                f"'{self.model_name}' produces {self._dimension}-dim "
                f"vectors but EMBEDDING_DIM is {self._expected_dim}. "
                f"Fix one of: (a) change EMBEDDING_MODEL to a "
                f"{self._expected_dim}-dim model, (b) set "
                f"EMBEDDING_DIM={self._dimension} in your env AND run "
                f"gateway/scripts/migrate_embedding_dim.py + reindex, or "
                f"(c) revert to the previous model."
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        self._load_model()
        return self._dimension

    async def embed_async(self, text: str, prefix: str = "search_query") -> list[float]:
        """Async wrapper: runs ``embed`` on a worker thread, serialized by a
        single service-level lock (T6b).

        Callers used to invoke ``embed`` inline (RAG) or on ad-hoc threads
        (memory); once memory retrieval runs CONCURRENTLY with RAG
        retrieval, two threads could hit the model at once — the event
        loop's implicit serialization is gone, so this lock is the explicit
        replacement. All query-time embedding goes through here.
        """
        def _locked_embed() -> list[float]:
            with self._embed_thread_lock:
                return self.embed(text, prefix)

        return await asyncio.to_thread(_locked_embed)

    def embed(self, text: str, prefix: str = "search_query") -> list[float]:
        """Embed a single text string."""
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32, prefix: str = "search_document") -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []
        self._load_model()
        vectors = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get cached EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(get_settings())
    return _embedding_service
