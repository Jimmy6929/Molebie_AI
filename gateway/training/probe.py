"""
Phase 4 probe — collect base-model responses to seed training data.

Sends every entry in ``probe_set.jsonl`` (or a custom set) through the
running gateway's ``POST /chat`` and captures the full response, latency,
inference metadata, and a heuristic first-pass label. The output feeds
``label.py`` (manual review pass) → ``build_dataset.py`` (SFT + ORPO
formats).

Why probe via the gateway and not the raw inference endpoint:
    The training data has to teach behavior the gateway *will see* at
    runtime — system_rag.txt template, retrieval evidence summary, tool
    schemas, the lot. Probing the bare LLM and training on those traces
    would teach the LoRA to operate without context it'll always have
    in production. Apples-to-apples.

Usage:

    # one-shot probe of the seed set
    python probe.py --gateway http://localhost:8000 \\
        --password smoketest \\
        --output probes/seed-2026-05-01.jsonl

The runner does NOT manage the gateway; you boot it however you like
(typically with all Phase 1-3 layers OFF — we want the *base* behavior
that fine-tuning will improve, not the post-processed behavior).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Reuse the chat() and login() helpers from the eval runner — no point
# duplicating HTTP plumbing.
HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "tests" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_baseline as rb     # noqa: E402


# ── Heuristic labelling ────────────────────────────────────────────────────


CITATION_RE = re.compile(r"\[S\d+\]")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"don't have|don't know|i'm not certain|cannot find|fictional|"
    r"doesn't exist|no such|not aware|no record|can't verify",
    re.IGNORECASE,
)
FALLBACK_STRING = "I don't have that in your notes."


def heuristic_label(
    entry: dict[str, Any],
    response_text: str,
    inference: dict[str, Any],
) -> dict[str, Any]:
    """First-pass label with confidence. Categories:

    - ``ground_truth_match``  — response matches the expected substrings
                                 in the probe entry (if present).
    - ``cited_response``      — has [S#] citations + factual content.
    - ``proper_abstention``   — emitted the canonical fallback or a clean
                                 refusal phrase.
    - ``likely_fabrication``  — response contains specific facts (numbers
                                 / URLs / dates) but no citations AND
                                 no refusal phrases. The most-actionable
                                 case for fine-tuning: relabel as
                                 abstention.
    - ``ambiguous``           — none of the above match cleanly. Needs
                                 manual review.

    Confidence is "high" when at least two signals agree, "medium" when
    one signal fires, "low" otherwise.
    """
    expected = entry.get("expected_substrings") or []
    has_citations = bool(CITATION_RE.search(response_text))
    has_specific_facts = bool(
        NUMBER_RE.search(response_text)
        or URL_RE.search(response_text)
        or DATE_RE.search(response_text)
    )
    has_refusal = bool(REFUSAL_RE.search(response_text))
    matches_fallback = FALLBACK_STRING in response_text

    expected_match = (
        any(s.lower() in response_text.lower() for s in expected)
        if expected else None
    )

    rag_metrics = (inference or {}).get("rag_metrics") or {}
    citations_report = rag_metrics.get("citations") or {}
    cove_report = rag_metrics.get("verification") or {}
    selfcheck_report = rag_metrics.get("selfcheck") or {}
    layer_already_flagged = bool(
        citations_report.get("unsupported_claims")
        or cove_report.get("unsupported_count", 0) > 0
        or selfcheck_report.get("flagged_count", 0) > 0
    )

    if expected_match is True and has_citations:
        return {
            "label": "ground_truth_match", "confidence": "high",
            "signals": ["expected_substring", "citation"],
        }
    if matches_fallback:
        return {
            "label": "proper_abstention", "confidence": "high",
            "signals": ["fallback_string"],
        }
    if has_refusal and not has_specific_facts:
        return {
            "label": "proper_abstention", "confidence": "medium",
            "signals": ["refusal_phrase"],
        }
    if has_specific_facts and not has_citations and not has_refusal:
        return {
            "label": "likely_fabrication",
            "confidence": "high" if layer_already_flagged else "medium",
            "signals": ["specific_facts", "no_citation", "no_refusal"]
                       + (["post_layer_flagged"] if layer_already_flagged else []),
        }
    if has_citations and has_specific_facts:
        return {
            "label": "cited_response",
            "confidence": "medium",
            "signals": ["citation", "specific_facts"],
        }
    return {
        "label": "ambiguous", "confidence": "low",
        "signals": [],
    }


# ── Main loop ─────────────────────────────────────────────────────────────


def load_probe_set(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def probe(
    gateway: str,
    password: str,
    probe_set: Path,
    output: Path,
    mode: str,
    timeout: float,
) -> None:
    entries = load_probe_set(probe_set)
    print(f"[probe] {len(entries)} queries from {probe_set.name} → {gateway}")
    token = rb.login(gateway, password)
    print("[probe] auth OK")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    label_counts: dict[str, int] = {}

    with output.open("w") as f:
        for i, entry in enumerate(entries, 1):
            eid = entry.get("id", f"q{i}")
            cat = entry.get("category", "?")
            query = entry["query"]
            print(f"[probe] {i:>3}/{len(entries)} [{cat:<22}] {eid}", flush=True)
            t0 = time.perf_counter()
            try:
                resp, latency = rb.chat(
                    gateway, token, query, mode=mode, timeout=timeout,
                )
            except Exception as exc:
                rec = {
                    "entry": entry,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_s": round(time.perf_counter() - t0, 3),
                }
                f.write(json.dumps(rec) + "\n")
                continue

            msg = resp.get("message") or {}
            text = msg.get("content") or ""
            inf = resp.get("inference") or {}
            label = heuristic_label(entry, text, inf)
            label_counts[label["label"]] = label_counts.get(label["label"], 0) + 1
            rec = {
                "entry": entry,
                "response": {
                    "content": text,
                    "tool_calls": inf.get("tool_calls") or [],
                    "rag_metrics": inf.get("rag_metrics") or {},
                    "mode_used": inf.get("mode_used"),
                },
                "latency_s": round(latency, 3),
                "heuristic_label": label,
                "needs_manual_review": label["label"] == "ambiguous"
                or label["confidence"] == "low",
            }
            f.write(json.dumps(rec) + "\n")
            written += 1

    print()
    print(f"[probe] wrote {written}/{len(entries)} records → {output}")
    print(f"[probe] heuristic labels: {label_counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--password", required=True)
    parser.add_argument("--probe-set", type=Path,
                        default=HERE / "probe_set.jsonl",
                        help="JSONL probe set (default: bundled seed set).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL of probe records.")
    parser.add_argument("--mode", default="instant",
                        choices=("instant", "thinking"))
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    try:
        probe(
            gateway=args.gateway, password=args.password,
            probe_set=args.probe_set, output=args.output,
            mode=args.mode, timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n[probe] interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
