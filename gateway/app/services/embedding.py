"""
Embedding service for RAG vector search.

Uses sentence-transformers to generate embeddings locally on CPU/Apple Silicon.
Dimension configured via EMBEDDING_MODEL in .env.
"""

from typing import List, Optional

from app.config import Settings, get_settings


class EmbeddingService:
    """Generate text embeddings using a local sentence-transformers model."""

    def __init__(self, settings: Settings):
        self.model_name = settings.embedding_model
        self.local_files_only = settings.embedding_local_only
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "RAG requires PyTorch and sentence-transformers — "
                "install with: pip install -r requirements-ml.txt"
            )

        print(f"[embedding] Loading model: {self.model_name} (local_files_only={self.local_files_only})")
        # When running in offline mode, tell ALL HuggingFace libraries
        # (including custom model code loaded via trust_remote_code) to
        # never make network requests. Without this, custom model code
        # can still try httpx calls even with local_files_only=True.
        if self.local_files_only:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        self._load_model()
        return self._dimension

    def embed(self, text: str, prefix: str = "search_query") -> List[float]:
        """Embed a single text string."""
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32, prefix: str = "search_document") -> List[List[float]]:
        """Embed multiple texts."""
        if not texts:
            return []
        self._load_model()
        vectors = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get cached EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(get_settings())
    return _embedding_service
