"""
Tool definitions + executor for Molebie agentic chat (Phase 2 task 2.2).

Four shipped tools, all low-risk:

  * ``search_notes``     — semantic search over the user's RAG corpus.
                           Currently runs every turn; tool form lets the
                           model only fire it when actually needed.
  * ``calculate``        — sandboxed arithmetic. Eliminates the most common
                           hallucination class on small models.
  * ``get_current_time`` — stops "today is..." fabrications cold.
  * ``web_search``       — SearXNG via the existing service. Today a UI
                           toggle; here the model decides.

Filesystem / shell / write-mutation tools are deliberately NOT shipped —
they need explicit user opt-in and per-tool sandboxing. Add later.

The executor validates each tool call against its JSON Schema and runs the
corresponding Python implementation. Failures surface back to the model
as a ``tool`` role message so it can retry or apologise.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from typing import Any

from app.config import Settings

# ── Schemas (OpenAI-compatible function-tool format) ────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Semantic search over the user's local notes (Obsidian / "
                "uploaded documents). Use when the user asks about their "
                "own knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5, max 12).",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression. Use this for ANY "
                "numerical computation — small models fabricate arithmetic. "
                "Only +, -, *, /, **, %, parentheses are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(17*23) + 4**2'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Returns the current UTC timestamp + local-timezone offset. "
                "Use whenever the user asks about 'today', 'now', or "
                "'the current date'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web via SearXNG. Use for real-time "
                "information that wouldn't be in the user's notes — news, "
                "release dates, current events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 10).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ── calculate: AST-based sandbox ────────────────────────────────────────

# Only arithmetic ops are allowed. ``eval()`` would let the model exfiltrate
# any builtin via ``__import__``; an AST walk with a whitelist makes that
# impossible by construction.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_arith(expression: str) -> float | int:
    """Evaluate an arithmetic expression. Raises ValueError on anything
    that isn't a literal-number-and-arithmetic-operator expression."""
    if len(expression) > 200:
        raise ValueError("Expression too long (max 200 chars)")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax: {exc.msg}") from exc

    def _walk(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Disallowed literal: {type(node.value).__name__}")
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_BINOPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed binary op: {type(node.op).__name__}")
            return op(_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _ALLOWED_UNARYOPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")
            return op(_walk(node.operand))
        raise ValueError(f"Disallowed node: {type(node).__name__}")

    return _walk(tree)


# ── Executor ────────────────────────────────────────────────────────────


class ToolExecutor:
    """Dispatches validated tool calls to their implementations.

    Constructed per-request because ``search_notes`` and ``web_search``
    bind to the request's ``user_id`` (RAG isolation) and need access to
    the live RAG / web-search services.
    """

    def __init__(
        self,
        user_id: str,
        rag_service: Any,
        web_search_service: Any,
        settings: Settings,
    ):
        self.user_id = user_id
        self.rag = rag_service
        self.web_search = web_search_service
        self.settings = settings

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a single tool call. Returns ``{ok: bool, result: ..., error?: str}``.

        Errors are caught and structured rather than raised — the chat
        handler appends them to the conversation as a ``tool`` role
        message so the model can recover or apologise.
        """
        try:
            if name == "search_notes":
                return await self._search_notes(args)
            if name == "calculate":
                return self._calculate(args)
            if name == "get_current_time":
                return self._get_current_time()
            if name == "web_search":
                return await self._web_search(args)
            return {"ok": False, "error": f"Unknown tool: {name}"}
        except KeyError as exc:
            return {"ok": False, "error": f"Missing required argument: {exc}"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def _search_notes(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args["query"]
        top_k = min(int(args.get("top_k", 5)), 12)
        if not self.rag.enabled:
            return {"ok": False, "error": "RAG is disabled."}
        if not await self.rag.user_has_documents(self.user_id):
            return {"ok": True, "result": {"chunks": [], "note": "No documents indexed."}}
        chunks = await self.rag.retrieve_context(
            self.user_id, query, limit=top_k * 4,
        )
        chunks = chunks[:top_k]
        return {
            "ok": True,
            "result": {
                "chunks": [
                    {
                        "source": c.get("filename", "unknown"),
                        "heading": (c.get("metadata") or {}).get("heading"),
                        "score": c.get("rerank_score") or c.get("similarity", 0),
                        "content": c.get("content", "")[:1000],
                    }
                    for c in chunks
                ],
            },
        }

    def _calculate(self, args: dict[str, Any]) -> dict[str, Any]:
        expression = str(args["expression"])
        value = _safe_eval_arith(expression)
        return {"ok": True, "result": {"expression": expression, "value": value}}

    def _get_current_time(self) -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        local = datetime.now().astimezone()
        return {
            "ok": True,
            "result": {
                "utc": now_utc.isoformat(),
                "local": local.isoformat(),
                "weekday": local.strftime("%A"),
            },
        }

    async def _web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args["query"]
        max_results = min(int(args.get("max_results", 5)), 10)
        results = await self.web_search.search(query)
        results = results[:max_results]
        return {
            "ok": True,
            "result": {
                "results": [
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": (r.get("content") or r.get("snippet") or "")[:400],
                    }
                    for r in results
                ],
            },
        }
