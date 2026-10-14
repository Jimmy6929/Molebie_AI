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
``rag.py`` and ``rag_rerank_floor`` threshold on these scores. The values
are calibrated against the Qwen backend exactly as it runs here (prompt,
dtype, device) — after changing any of that, re-run
``gateway/tests/eval/calibrate_thresholds.py``.
"""

import threading
from typing import Any

from app.config import Settings, get_settings

# Qwen3-Reranker uses a fixed instruction template — see the model card.
_QWEN_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
# The chat wrapper the model was trained with, verbatim from the model card.
# Without it the yes/no read happens right after the document's last token
# instead of at the start of the assistant turn.
_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_MAX_LENGTH = 8192


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
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        # One forward pass at a time on the Qwen model. RAG already holds
        # its own rerank lock, but the grounding judge scores through
        # ``_qwen_score`` without it.
        self._qwen_lock = threading.Lock()

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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers is not installed. "
                "Qwen-style rerankers require it — install with: "
                "pip install -r requirements-ml.txt"
            )

        print(f"[reranker] Loading Qwen causal reranker: {self.model_name}")
        # Left padding keeps every pair's final token in the last position,
        # which is where the batched yes/no read happens.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        self._auto_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        # Half precision on GPU, as the model card recommends; CPU has no
        # fast half-precision path.
        dtype = torch.float32 if device == "cpu" else torch.float16
        self._auto_model.to(device=device, dtype=dtype)
        self._auto_model.eval()
        self._prefix_ids = self._tokenizer.encode(_QWEN_PREFIX, add_special_tokens=False)
        self._suffix_ids = self._tokenizer.encode(_QWEN_SUFFIX, add_special_tokens=False)
        # The yes/no token IDs depend on tokenizer; resolve once.
        self._yes_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("no", add_special_tokens=False)[0]
        self._impl = "qwen_causal"
        print(
            f"[reranker] Qwen reranker loaded on {device} "
            f"(yes_id={self._yes_id}, no_id={self._no_id})"
        )

    def _qwen_scores(self, query: str, documents: list[str]) -> list[float]:
        """Score every (query, doc) pair in one batched forward pass."""
        import torch

        prompts = [
            f"<Instruct>: {_QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        wrapper_len = len(self._prefix_ids) + len(self._suffix_ids)
        encoded = self._tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=_QWEN_MAX_LENGTH - wrapper_len,
            return_attention_mask=False,
        )
        encoded["input_ids"] = [
            self._prefix_ids + ids + self._suffix_ids for ids in encoded["input_ids"]
        ]
        inputs = self._tokenizer.pad(encoded, padding=True, return_tensors="pt").to(
            self._auto_model.device
        )
        with self._qwen_lock, torch.inference_mode():
            # Run the decoder only and apply the LM head to the final
            # position: the full-vocabulary logits for every token of every
            # pair are ~150k floats each and none of them are needed.
            last_hidden = self._auto_model.base_model(**inputs).last_hidden_state[:, -1]
            logits = self._auto_model.get_output_embeddings()(last_hidden)
            yes_no = logits[:, [self._yes_id, self._no_id]].float()
            return torch.softmax(yes_no, dim=1)[:, 0].tolist()

    def _qwen_score(self, query: str, document: str) -> float:
        """Score a single (query, doc) pair via the yes/no logit head."""
        return self._qwen_scores(query, [document])[0]

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
            scores = self._qwen_scores(query, [c.get("content", "") for c in chunks])
            for chunk, score in zip(chunks, scores, strict=True):
                chunk["rerank_score"] = score
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
