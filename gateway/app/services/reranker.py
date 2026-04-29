"""
Cross-encoder / Qwen-style reranker for the RAG pipeline.

Takes the top-N candidates from hybrid search and re-scores each
(query, chunk) pair. Two backends are supported, picked automatically:

  1. ``cross_encoder`` — sentence-transformers ``CrossEncoder``, used for
     classic rerankers like ``cross-encoder/ms-marco-MiniLM-L6-v2`` and
     BGE-Reranker. Score is sigmoid-ish in 0..1.

  2. ``qwen_causal`` — Qwen3-Reranker is a causal LM fine-tuned for
     reranking. We compute ``softmax([logit('yes'), logit('no')])`` from
     the next-token distribution and use the ``yes`` probability as the
     score (also 0..1, comparable to the CrossEncoder range).

The 0..1 alignment matters because ``compute_retrieval_confidence`` in
``rag.py`` thresholds at 0.3 / 0.5 / 0.7 — those numbers are rough but
serviceable for both backends. Recalibrate against a held-out query set
when you have time.
"""

from typing import Any

from app.config import Settings, get_settings


# Qwen3-Reranker uses a fixed instruction template — see the model card.
_QWEN_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _is_qwen_style(model_name: str) -> bool:
    """Detect Qwen-family rerankers that need the AutoModel yes/no path."""
    lower = model_name.lower()
    return "qwen" in lower and "rerank" in lower


class RerankerService:
    """Cross-encoder / causal-LM reranker with auto-detected backend."""

    def __init__(self, settings: Settings):
        self.model_name = settings.rag_reranker_model
        self._impl: str | None = None
        # CrossEncoder backend
        self._model = None
        # Qwen causal backend
        self._tokenizer = None
        self._auto_model = None
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def _load_model(self):
        if self._impl is not None:
            return

        if _is_qwen_style(self.model_name):
            self._load_qwen()
        else:
            self._load_cross_encoder()

    def _load_cross_encoder(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Reranking requires PyTorch and sentence-transformers — "
                "install with: pip install -r requirements-ml.txt"
            )

        print(f"[reranker] Loading CrossEncoder: {self.model_name}")
        self._model = CrossEncoder(self.model_name)
        self._impl = "cross_encoder"
        print("[reranker] CrossEncoder loaded")

    def _load_qwen(self):
        # Qwen3-Reranker: causal LM, score = P(next_token == "yes" | prompt).
        # We don't trust local_files_only here — the user controls model
        # availability via huggingface-cli download. If the model isn't
        # cached, the AutoModel call surfaces the original HF error so the
        # operator gets a clean "run huggingface-cli download" pointer.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers is not installed. "
                "Qwen-style rerankers require it — install with: "
                "pip install -r requirements-ml.txt"
            )

        print(f"[reranker] Loading Qwen causal reranker: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._auto_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._auto_model.eval()
        # The yes/no token IDs depend on tokenizer; resolve once.
        self._yes_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("no", add_special_tokens=False)[0]
        self._impl = "qwen_causal"
        print(
            f"[reranker] Qwen reranker loaded "
            f"(yes_id={self._yes_id}, no_id={self._no_id})"
        )

    def _qwen_score(self, query: str, document: str) -> float:
        """Score a single (query, doc) pair via the yes/no logit head."""
        import torch

        prompt = (
            f"<Instruct>: {_QWEN_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        with torch.no_grad():
            logits = self._auto_model(**inputs).logits[0, -1, :]
        yes_no = torch.tensor([logits[self._yes_id], logits[self._no_id]])
        prob_yes = torch.softmax(yes_no, dim=0)[0].item()
        return float(prob_yes)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Re-score and re-rank chunks. Returns top-k sorted by ``rerank_score``."""
        if not chunks:
            return []

        self._load_model()

        if self._impl == "cross_encoder":
            pairs = [(query, c.get("content", "")) for c in chunks]
            scores = self._model.predict(pairs)
            for chunk, score in zip(chunks, scores, strict=False):
                chunk["rerank_score"] = float(score)
        elif self._impl == "qwen_causal":
            for chunk in chunks:
                chunk["rerank_score"] = self._qwen_score(
                    query, chunk.get("content", "")
                )
        else:
            raise RuntimeError(f"Reranker not initialised for backend {self._impl}")

        ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]


_reranker: RerankerService | None = None


def get_reranker() -> RerankerService:
    """Get cached RerankerService instance."""
    global _reranker
    if _reranker is None:
        _reranker = RerankerService(get_settings())
    return _reranker
