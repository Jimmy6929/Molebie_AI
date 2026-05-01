"""
Phase 1+2 verification harness.

Exercises the pure-function code paths added in Phase 1 and Phase 2 of
the hallucination-mitigation work. Run before moving to Phase 3 to
confirm the code actually behaves as the changelog claims.

Usage:
    .venv/bin/python -m gateway.tests.verify_phase12

Or, with pytest:
    .venv/bin/pytest gateway/tests/verify_phase12.py -v

Pure-function tests (no network, no models): always run.
Live model loads (Qwen3-Embedding, Qwen3-Reranker): gated behind
``MOLEBIE_VERIFY_LIVE_MODELS=1`` because they trigger ~1.4 GB of HF
downloads on first run.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make ``app.*`` importable when this is run as a script.
GATEWAY_DIR = Path(__file__).resolve().parent.parent
if str(GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(GATEWAY_DIR))

# Pin Phase-1/2 defaults so a stale .env.local doesn't poison the harness.
os.environ.setdefault("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
os.environ.setdefault("RAG_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
# Allow live downloads when the user opts in via MOLEBIE_VERIFY_LIVE_MODELS=1.
# The default `embedding_local_only=True` would block first-time fetches.
if os.environ.get("MOLEBIE_VERIFY_LIVE_MODELS") == "1":
    os.environ["EMBEDDING_LOCAL_ONLY"] = "false"
os.environ.setdefault("RAG_MATCH_COUNT", "30")
os.environ.setdefault("RAG_RERANK_TOP_K", "8")
os.environ.setdefault("RAG_CHUNK_SIZE", "512")
os.environ.setdefault("RAG_CHUNK_OVERLAP", "64")
os.environ.setdefault("RAG_MATCH_THRESHOLD", "0.3")
os.environ.setdefault("RAG_MAX_CONTEXT_CHARS", "12000")
os.environ.setdefault("TOOL_CALLING_ENABLED", "false")
os.environ.setdefault("SELF_CONSISTENCY_ENABLED", "false")


# ── Test runner ───────────────────────────────────────────────────────────

CHECKS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    badge = "PASS" if ok else "FAIL"
    line = f"  [{badge}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ── Phase 1.2: <think> stripping ──────────────────────────────────────────


def test_strip_think_blocks():
    section("Phase 1.2 — defensive <think> strip")
    from app.services.inference import _strip_think_in_messages

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>plotting</think>actual answer"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>x</think>plain"},
                {"type": "image", "image": "..."},
            ],
        },
    ]
    cleaned = _strip_think_in_messages(msgs)
    _record(
        "strips <think> from string content",
        cleaned[1]["content"] == "actual answer",
        f"got {cleaned[1]['content']!r}",
    )
    _record(
        "strips <think> from multimodal text part",
        cleaned[2]["content"][0]["text"] == "plain",
        f"got {cleaned[2]['content'][0]['text']!r}",
    )
    _record(
        "leaves non-text parts untouched",
        cleaned[2]["content"][1]["type"] == "image",
    )
    _record(
        "does not mutate the input list",
        msgs[1]["content"] == "<think>plotting</think>actual answer",
    )


# ── Phase 1.4: chunking + breadcrumbs + code fences ───────────────────────


def test_chunking_and_breadcrumbs():
    section("Phase 1.4 — header-aware chunking + code fences")
    from app.services.document_processor import (
        _embedding_prefix,
        _merge_unbalanced_fences,
        chunk_text_structured,
    )

    md = (
        "# Setup\n\nIntro paragraph.\n\n"
        "## Installation\n\nRun `pip install foo`.\n\n"
        "### Dependencies\n\nNeeds python 3.10+.\n\n"
        "## Usage\n\nCall `foo()`.\n"
    )
    chunks = chunk_text_structured(md, chunk_size=120, chunk_overlap=20)
    by_breadcrumb = {tuple(c["breadcrumb"]): c for c in chunks}

    _record(
        "produces multiple chunks for nested headings",
        len(chunks) >= 3,
        f"got {len(chunks)} chunks",
    )
    _record(
        "h1 > h2 > h3 breadcrumb captured",
        ("Setup", "Installation", "Dependencies") in by_breadcrumb,
        f"breadcrumbs={list(by_breadcrumb.keys())}",
    )
    _record(
        "sibling h2 resets the deeper heading",
        ("Setup", "Usage") in by_breadcrumb,
        f"breadcrumbs={list(by_breadcrumb.keys())}",
    )

    # Code-fence preservation
    fenced = [
        "intro\n```python",
        "def f():\n    return 1\n```\nafter",
    ]
    merged = _merge_unbalanced_fences(fenced)
    _record(
        "merges unbalanced fences",
        len(merged) == 1 and merged[0].count("```") == 2,
        f"got {len(merged)} chunks, fences={[c.count('```') for c in merged]}",
    )

    # Embedding prefix
    prefix = _embedding_prefix(["Setup", "Installation"], "guide.md")
    _record(
        "embedding prefix has breadcrumb + title",
        prefix == "Setup > Installation | guide.md",
        f"got {prefix!r}",
    )
    _record(
        "embedding prefix empty when no metadata",
        _embedding_prefix([], None) == "",
    )


# ── Phase 1.7: LongContextReorder (U-shape) ───────────────────────────────


def test_long_context_reorder():
    section("Phase 1.7 — LongContextReorder U-shape")
    from app.services.rag import _reorder_for_context

    # Build chunks with descending rerank scores 1.0, 0.9, ..., 0.1
    chunks = [
        {"id": i, "rerank_score": 1.0 - 0.1 * i, "content": f"c{i}"}
        for i in range(10)
    ]
    reordered = _reorder_for_context(chunks)
    # The implementation matches LangChain's LongContextReorder: iterate
    # ascending (lowest first), alternately insert(0)/append. With an even
    # length 10 the highest score lands at index -1 and second-highest at 0.
    edges = {reordered[0]["id"], reordered[-1]["id"]}
    _record(
        "U-shape: top two scores land at the two edges",
        edges == {0, 1},
        f"edge ids={edges}, edge scores=({reordered[0]['rerank_score']:.2f}, "
        f"{reordered[-1]['rerank_score']:.2f})",
    )
    middle_scores = [c["rerank_score"] for c in reordered[3:-3]]
    _record(
        "U-shape: lowest scores in the middle",
        all(s <= 0.6 for s in middle_scores) if middle_scores else True,
        f"middle scores={[round(s, 2) for s in middle_scores]}",
    )
    _record(
        "no-op below 3 chunks",
        _reorder_for_context(chunks[:2]) == chunks[:2],
    )


# ── Phase 1.8: Retrieval confidence buckets ───────────────────────────────


def test_retrieval_confidence():
    section("Phase 1.8 — retrieval confidence buckets")
    from app.services.rag import compute_retrieval_confidence

    cases = [
        ("NONE", []),
        ("REFUSE", [{"rerank_score": 0.1}, {"rerank_score": 0.05}]),
        ("LOW", [{"rerank_score": 0.4}, {"rerank_score": 0.45}, {"rerank_score": 0.2}]),
        ("HIGH", [{"rerank_score": 0.9}, {"rerank_score": 0.7}, {"rerank_score": 0.6}]),
        ("MODERATE", [{"rerank_score": 0.55}, {"rerank_score": 0.6}, {"rerank_score": 0.4}]),
    ]
    for expected, chunks in cases:
        got = compute_retrieval_confidence(chunks)
        _record(
            f"confidence({expected}) returns {expected}",
            got == expected,
            f"got {got}",
        )


# ── Phase 2.2: tool sandboxing ────────────────────────────────────────────


def test_calculate_sandbox():
    section("Phase 2.2 — calculate sandbox (AST whitelist)")
    from app.services.tools import _safe_eval_arith

    happy = [
        ("17 * 23 + 4**2", 407),
        ("(2 + 3) * 4", 20),
        ("100 / 4", 25),
        ("2 ** 10", 1024),
        ("-5 + 3", -2),
        ("17 % 5", 2),
    ]
    for expr, expected in happy:
        try:
            got = _safe_eval_arith(expr)
            _record(f"calculate({expr!r}) = {expected}", got == expected, f"got {got}")
        except Exception as exc:
            _record(f"calculate({expr!r}) = {expected}", False, f"raised {exc!r}")

    attacks = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "len([1,2,3])",
        "(1).__class__",
        "1 if True else 2",
        "[1, 2]",
        "{1: 2}",
        "lambda x: x",
        "1; 2",
        "'hi'",  # string literal — disallowed
        "a + b",  # name lookup
        "1" * 250,  # too long
    ]
    for expr in attacks:
        try:
            got = _safe_eval_arith(expr)
            _record(f"reject {expr[:40]!r}", False, f"returned {got}")
        except (ValueError, SyntaxError):
            _record(f"reject {expr[:40]!r}", True)
        except Exception as exc:
            _record(f"reject {expr[:40]!r}", False, f"unexpected {type(exc).__name__}: {exc}")


def test_tool_executor():
    section("Phase 2.2 — ToolExecutor dispatch")
    from app.config import get_settings
    from app.services.tools import ToolExecutor

    executor = ToolExecutor(
        user_id="test",
        rag_service=None,           # not exercised here
        web_search_service=None,    # not exercised here
        settings=get_settings(),
    )

    async def run():
        # calculate
        ok = await executor.execute("calculate", {"expression": "(7+3)*4"})
        _record(
            "executor.calculate happy path",
            ok["ok"] and ok["result"]["value"] == 40,
            str(ok),
        )
        bad = await executor.execute("calculate", {"expression": "__import__('os')"})
        _record(
            "executor.calculate rejects unsafe",
            not bad["ok"] and "Disallowed" in (bad.get("error") or ""),
            str(bad),
        )

        # get_current_time
        t = await executor.execute("get_current_time", {})
        _record(
            "executor.get_current_time returns utc/local/weekday",
            t["ok"] and {"utc", "local", "weekday"}.issubset(t["result"].keys()),
            str(t)[:120],
        )

        # unknown tool
        unk = await executor.execute("does_not_exist", {})
        _record(
            "executor unknown tool errors cleanly",
            not unk["ok"] and "Unknown tool" in (unk.get("error") or ""),
            str(unk),
        )

        # missing required arg
        miss = await executor.execute("calculate", {})
        _record(
            "executor missing required arg errors cleanly",
            not miss["ok"],
            str(miss),
        )

    asyncio.run(run())


def test_tool_schemas():
    section("Phase 2.2 — TOOL_SCHEMAS shape")
    from app.services.tools import TOOL_SCHEMAS

    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    _record(
        "all four tools present",
        names == {"search_notes", "calculate", "get_current_time", "web_search"},
        f"got {names}",
    )
    for tool in TOOL_SCHEMAS:
        fn = tool["function"]
        _record(
            f"tool '{fn['name']}' has parameters object",
            fn["parameters"]["type"] == "object" and "properties" in fn["parameters"],
        )


# ── Phase 2.3: self-consistency ───────────────────────────────────────────


def test_verifiable_query_classifier():
    section("Phase 2.3 — is_verifiable_query trigger")
    from app.services.consistency import is_verifiable_query, normalise_answer

    yes = ["What is the capital of France?", "How many planets in the solar system?",
           "Is it true that...", "Calculate 17 * 23"]
    no = ["Summarize this", "Explain TCP/IP", "Tell me about elephants",
          "Write a poem", "How do I install python?"]
    for q in yes:
        _record(f"verifiable: {q!r}", is_verifiable_query(q))
    for q in no:
        _record(f"NOT verifiable: {q!r}", not is_verifiable_query(q))

    # Normalisation collapses citations / articles / punctuation
    n1 = normalise_answer("The Paris [S1]. ")
    n2 = normalise_answer("Paris")
    _record(f"normalise collapses 'The Paris [S1].' → 'Paris'", n1 == n2 == "paris", f"{n1!r} vs {n2!r}")


def test_self_consistency_voting():
    section("Phase 2.3 — vote_with_self_consistency (mock inference)")
    from app.config import get_settings
    from app.services.consistency import vote_with_self_consistency

    class MockInference:
        def __init__(self, samples):
            self.samples = list(samples)
            self.calls = 0

        async def generate_response(self, **kwargs):
            i = self.calls
            self.calls += 1
            return {"content": self.samples[i % len(self.samples)]}

    async def run():
        settings = get_settings()
        # 4 of 5 say "Paris", one says "Lyon" → confident majority + ESC
        mock = MockInference(["Paris.", "The Paris", "Paris", "Lyon", "Paris [S1]"])
        out = await vote_with_self_consistency(
            inference=mock, messages=[], mode="instant", settings=settings,
        )
        top = max(out["vote_counts"].items(), key=lambda kv: kv[1])
        _record(
            "majority vote picks 'paris'",
            top[0] == "paris" and out["confident"],
            f"vote_counts={out['vote_counts']}",
        )
        _record(
            "early-stopping engaged",
            out["early_stopped"] is True,
            f"early_stopped={out['early_stopped']}",
        )

        # All five disagree → no confident answer, returns the canned message
        mock2 = MockInference(["a", "b", "c", "d", "e"])
        out2 = await vote_with_self_consistency(
            inference=mock2, messages=[], mode="instant", settings=settings,
        )
        _record(
            "no-majority returns 'inconsistent results' message",
            ("inconsistent" in out2["content"].lower()) or (not out2["confident"]),
            f"content={out2['content']!r} confident={out2['confident']}",
        )

    asyncio.run(run())


# ── Phase 1.9 / 2.4: citation validation + footnote ───────────────────────


def test_citation_validation():
    section("Phase 1.9 / 2.4 — citation validation + footnote")
    from app.routes.chat import _maybe_append_footnote, _validate_citations

    chunks = [
        {"content": "Mount Everest is 8849 metres tall, located in Nepal."},
        {"content": "K2 stands at 8611 metres in the Karakoram range."},
        {"content": "Lhotse is the fourth-highest peak in the world."},
    ]

    # Valid citations, supported claims
    good = "Everest is 8849 metres [S1]. K2 is 8611 metres [S2]."
    rep = _validate_citations(good, num_sources=3, rag_chunks=chunks)
    _record(
        "good response: zero invalid, zero unsupported",
        not rep["invalid_indices"] and not rep["unsupported_claims"],
        f"invalid={rep['invalid_indices']} unsupported={len(rep['unsupported_claims'])}",
    )
    _record("good response: cited_count==2", rep["cited_count"] == 2,
            f"got cited_count={rep['cited_count']}")

    # Invalid index ([S5] when only 3 sources)
    bad = "Lhotse is 8516 metres [S5]."
    rep_bad = _validate_citations(bad, num_sources=3, rag_chunks=chunks)
    _record("invalid index detected", rep_bad["invalid_indices"] == [5],
            f"invalid_indices={rep_bad['invalid_indices']}")

    # Uncited specific claim → unsupported_claims fires
    fab = "Annapurna is 8091 metres tall."
    rep_fab = _validate_citations(fab, num_sources=3, rag_chunks=chunks)
    _record(
        "uncited number flagged as unsupported",
        bool(rep_fab["unsupported_claims"]) and rep_fab["needs_footnote"],
        f"unsupported={rep_fab['unsupported_claims']}",
    )

    # Wrong-source citation → weak_citation fires.
    # Note: the 40% token-overlap heuristic only catches this when the
    # sentence and the cited chunk share little vocabulary. Generic words
    # ("metres", "the") shared across chunks defeat the check — see the
    # verification report for the limitation. This case picks distinctive
    # tokens so the mechanism is exercised end-to-end.
    wrong = "Annapurna Manaslu Dhaulagiri peaks share borders [S2]."
    rep_wrong = _validate_citations(wrong, num_sources=3, rag_chunks=chunks)
    _record(
        "wrong-source citation (distinctive vocab) flagged as weak",
        len(rep_wrong["weak_citations"]) >= 1,
        f"weak={rep_wrong['weak_citations']}",
    )

    # Document the limitation: shared generic vocab masks the mismatch.
    soft = "Lhotse is 8516 metres [S2]."
    rep_soft = _validate_citations(soft, num_sources=3, rag_chunks=chunks)
    _record(
        "KNOWN LIMITATION: shared generic vocab ('metres') masks mismatch",
        not rep_soft["weak_citations"],
        f"as expected, no weak flag — phase 3 CoVe/SelfCheck handles this",
    )

    # Footnote idempotency
    text = "Some answer."
    rep_needing = {"needs_footnote": True}
    once = _maybe_append_footnote(text, rep_needing)
    twice = _maybe_append_footnote(once, rep_needing)
    _record(
        "footnote appended once, idempotent on re-apply",
        once != text and once == twice and once.count("Some claims in this response") == 1,
        f"twice ends with footnote: {twice.endswith('your notes.*')}",
    )


# ── Prompt files ───────────────────────────────────────────────────────────


def test_prompts_present():
    section("Phase 1.1 — prompt files")
    base = GATEWAY_DIR / "prompts"
    for fname in ("system.txt", "system_rag.txt", "system_voice.txt"):
        p = base / fname
        _record(f"prompt exists: {fname}", p.exists() and p.stat().st_size > 100,
                f"size={p.stat().st_size if p.exists() else 'missing'}")
    rag = (base / "system_rag.txt").read_text()
    _record(
        "system_rag.txt mentions [S#] citation",
        "[S" in rag and "citation" in rag.lower(),
    )
    _record(
        "system_rag.txt has the exact fallback string",
        "I don't have that in your notes" in rag,
    )
    sysprompt = (base / "system.txt").read_text()
    _record(
        "system.txt includes UNCERTAINTY rule",
        "UNCERTAINTY" in sysprompt or "uncertain" in sysprompt.lower(),
    )


# ── Phase 1.2: sampling presets via config helpers ─────────────────────────


def test_sampling_presets():
    section("Phase 1.2 — sampling presets per mode")
    from app.config import get_settings
    s = get_settings()
    instant_pp = s.get_presence_penalty_for_mode("instant")
    thinking_pp = s.get_presence_penalty_for_mode("thinking")
    instant_rp = s.get_repetition_penalty_for_mode("instant")
    thinking_rp = s.get_repetition_penalty_for_mode("thinking")
    _record("instant presence_penalty == 1.5", instant_pp == 1.5, f"got {instant_pp}")
    _record("thinking presence_penalty == 0.0", thinking_pp == 0.0, f"got {thinking_pp}")
    _record("repetition_penalty == 1.0 both modes",
            instant_rp == 1.0 and thinking_rp == 1.0,
            f"instant={instant_rp} thinking={thinking_rp}")
    _record(
        "presence_penalty hard ceiling = 1.5",
        s.inference_max_presence_penalty == 1.5,
        f"got {s.inference_max_presence_penalty}",
    )
    _record(
        "thinking auto-disable for RAG flag default ON",
        s.inference_thinking_auto_disable_for_rag is True,
    )


# ── Live model loads (gated) ───────────────────────────────────────────────


def test_live_qwen_reranker():
    section("Phase 1.6 — Qwen3-Reranker-0.6B live load (LIVE)")
    from app.config import get_settings
    from app.services.reranker import RerankerService

    settings = get_settings()
    rr = RerankerService(settings)
    try:
        rr._load_model()
    except Exception as exc:
        _record("Qwen3-Reranker loads", False, f"raised {type(exc).__name__}: {exc}")
        return
    _record("Qwen3-Reranker loads", rr._impl == "qwen_causal",
            f"backend={rr._impl}, yes_id={rr._yes_id}, no_id={rr._no_id}")

    # Score one relevant + one irrelevant pair; relevant should outscore
    relevant_score = rr._qwen_score(
        "What is the capital of France?",
        "Paris is the capital and largest city of France.",
    )
    irrelevant_score = rr._qwen_score(
        "What is the capital of France?",
        "Apples are a fruit grown in many temperate climates.",
    )
    _record(
        "relevant > irrelevant",
        relevant_score > irrelevant_score,
        f"relevant={relevant_score:.3f} irrelevant={irrelevant_score:.3f}",
    )
    _record(
        "scores in [0, 1]",
        0.0 <= relevant_score <= 1.0 and 0.0 <= irrelevant_score <= 1.0,
        f"relevant={relevant_score:.3f} irrelevant={irrelevant_score:.3f}",
    )


def test_live_qwen_embedding():
    section("Phase 1.5 — Qwen3-Embedding-0.6B live load (LIVE)")
    from app.config import get_settings
    from app.services.embedding import EmbeddingService

    es = EmbeddingService(get_settings())
    try:
        vec = es.embed("The quick brown fox jumps over the lazy dog.")
    except Exception as exc:
        _record("Qwen3-Embedding loads + embeds", False,
                f"raised {type(exc).__name__}: {exc}")
        return
    _record(
        "Qwen3-Embedding produces 1024-dim vector",
        len(vec) == 1024,
        f"len={len(vec)}",
    )


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    test_strip_think_blocks()
    test_chunking_and_breadcrumbs()
    test_long_context_reorder()
    test_retrieval_confidence()
    test_calculate_sandbox()
    test_tool_executor()
    test_tool_schemas()
    test_verifiable_query_classifier()
    test_self_consistency_voting()
    test_citation_validation()
    test_prompts_present()
    test_sampling_presets()

    if os.environ.get("MOLEBIE_VERIFY_LIVE_MODELS") == "1":
        test_live_qwen_reranker()
        test_live_qwen_embedding()
    else:
        print("\n[SKIP] live model loads — set MOLEBIE_VERIFY_LIVE_MODELS=1 to run")

    total = len(CHECKS)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    failed = total - passed
    print(f"\n=== Summary ===\n  {passed}/{total} pass, {failed} fail")
    if failed:
        print("\nFailures:")
        for name, ok, detail in CHECKS:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
