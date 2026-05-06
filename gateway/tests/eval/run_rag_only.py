"""Run the golden set queries through the RAG retrieval pipeline only.

This is a calibration harness — it bypasses the chat layer entirely so
we don't need a JWT or running gateway. It exists so we can collect
``[rag] RERANKER_SCORES …`` lines from the stdout (consumed by
``calibrate_thresholds.py``) without having to spin up the full HTTP
stack.

Usage (from repo root):
    python gateway/tests/eval/run_rag_only.py \\
        --user-id 00000000-0000-0000-0000-000000000001 \\
        --golden gateway/tests/eval/golden_set.jsonl \\
        2>&1 | tee /tmp/eval.log

It prints one ``[rag] RERANKER_SCORES …`` line per query (plus the
embedding/rerank diagnostic prints already in ``rag.py``). Then run:
    python gateway/tests/eval/calibrate_thresholds.py --log /tmp/eval.log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _ensure_gateway_on_path() -> None:
    """The gateway package lives at <repo>/gateway/ ; running this script
    from anywhere should still find it. We mutate sys.path rather than
    relying on cwd because the script may be invoked via a relative or
    absolute path from any working dir."""
    here = Path(__file__).resolve()
    gateway_root = here.parent.parent.parent  # tests/eval/ → tests/ → gateway/
    if str(gateway_root) not in sys.path:
        sys.path.insert(0, str(gateway_root))


_ensure_gateway_on_path()

from app.services.rag import get_rag_service  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-id", required=True,
                        help="UUID of the user whose document_chunks to search")
    parser.add_argument("--golden",
                        default=Path(__file__).parent / "golden_set.jsonl",
                        type=Path,
                        help="Golden set JSONL")
    parser.add_argument("--limit", type=int, default=0,
                        help="Run only the first N entries (0 = all). Useful for smoke-testing.")
    parser.add_argument("--categories", default="",
                        help="Comma-separated list of categories to include (default: all)")
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"error: golden set not found: {args.golden}", file=sys.stderr)
        return 2

    entries = []
    with args.golden.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        entries = [e for e in entries if e.get("category") in wanted]
    if args.limit > 0:
        entries = entries[: args.limit]

    print(f"[harness] {len(entries)} golden entries selected", file=sys.stderr)

    rag = get_rag_service()
    if not rag.enabled:
        print("error: RAG_ENABLED=false in settings — nothing to score", file=sys.stderr)
        return 3

    fail = 0
    for i, e in enumerate(entries, 1):
        qid = e.get("id", f"q{i}")
        cat = e.get("category", "?")
        query = e["query"]
        print(f"[harness] ── {i}/{len(entries)}  {qid}  [{cat}]  {query!r}", file=sys.stderr)
        try:
            chunks = await rag.retrieve_context(args.user_id, query)
            print(f"[harness] returned {len(chunks)} chunk(s)", file=sys.stderr)
        except Exception as exc:
            fail += 1
            print(f"[harness] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"[harness] done. {len(entries) - fail} ok, {fail} failed", file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
