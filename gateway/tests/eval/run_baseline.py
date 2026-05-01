"""
Phase 2.1 / Phase 4 baseline runner.

Hits the running gateway's ``POST /chat`` endpoint with every entry in
``golden_set.jsonl``, captures the response + latency + every layer's
metadata (citations, CoVe verdicts, Judge verdicts, SelfCheck flags),
and emits a JSON report with per-category pass rates and per-layer
activity statistics.

Why a custom runner instead of just Promptfoo:

  Promptfoo can mark queries pass/fail against assertions. It cannot
  tell you "of the 50 queries we ran, the Judge fired on 30 of them
  and flagged 5; CoVe fired on 12 and flagged 2; SelfCheck fired on
  8 and flagged 4 — average added latency was 1.2s for Judge alone,
  9.4s for the full stack". Those numbers are exactly what you need
  to set thresholds and decide which layers to flip on in production.

  This runner produces both: per-query verdicts AND per-layer activity.
  Promptfoo stays useful for the pure assertion side; this runner is
  for layer ablation and Phase 4 baseline capture.

Usage:

    # one-shot baseline against the gateway running on :8000
    python run_baseline.py --gateway http://localhost:8000 \\
        --password smoketest \\
        --output baseline-2026-05-01.json

    # ablation: vary layer flags via separate gateway boots
    python run_baseline.py --gateway http://localhost:8000 \\
        --password smoketest \\
        --label "all-layers-on" \\
        --output ablation-all-on.json

The gateway must already be running with whatever flags you want to
test — the runner doesn't manage the gateway. See ``ablation.sh`` for
a one-command "boot gateway + run + summarise" loop.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_GOLDEN = HERE / "golden_set.jsonl"

FALLBACK_STRING = "I don't have that in your notes."
REFUSAL_RE = re.compile(
    r"don't have|don't know|i'm not certain|cannot find|fictional|"
    r"doesn't exist|no such|not aware|no record|can't verify",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[S\d+\]")


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def login(gateway: str, password: str) -> str:
    """Get a JWT via the single-user login endpoint. On first use the
    password is set; subsequent calls verify it. The runner doesn't try
    to detect which case it's in — that's the operator's job."""
    res = _post(
        f"{gateway}/auth/login-simple",
        {"password": password},
        headers={},
        timeout=5,
    )
    return res["token"]


def chat(
    gateway: str,
    token: str,
    message: str,
    mode: str = "instant",
    timeout: float = 240.0,
) -> tuple[dict, float]:
    """One /chat round-trip. Returns (response_json, latency_seconds)."""
    t0 = time.perf_counter()
    res = _post(
        f"{gateway}/chat",
        {"message": message, "mode": mode},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    return res, time.perf_counter() - t0


# ── Per-category assertion logic ──────────────────────────────────────────


def _all_substrings(text: str, subs: list[str]) -> bool:
    return all(s.lower() in text.lower() for s in subs)


def _any_substring(text: str, subs: list[str]) -> bool:
    return any(s.lower() in text.lower() for s in subs)


def evaluate(entry: dict, response_text: str, tool_calls: list[dict]) -> dict:
    """Per-entry pass/fail with a structured reason. Categories use
    different assertions; we centralise them here so Promptfoo and
    this runner stay aligned."""
    cat = entry.get("category", "")
    expected = entry.get("expected_substrings") or []
    any_match = entry.get("any_substring", False)

    if cat == "rag_grounded":
        if entry.get("must_cite") and not CITATION_RE.search(response_text):
            return {"pass": False, "reason": "missing [S#] citation"}
        if expected:
            check = _any_substring if any_match else _all_substrings
            if not check(response_text, expected):
                return {
                    "pass": False,
                    "reason": (
                        f"missing expected substring(s): {expected} "
                        f"({'any' if any_match else 'all'})"
                    ),
                }
        return {"pass": True, "reason": "citation + content match"}

    if cat == "adversarial_premise":
        if REFUSAL_RE.search(response_text):
            return {"pass": True, "reason": "refused"}
        return {"pass": False, "reason": "did not refuse fake premise"}

    if cat == "must_abstain":
        if FALLBACK_STRING in response_text:
            return {"pass": True, "reason": "exact fallback string"}
        if REFUSAL_RE.search(response_text):
            return {"pass": True, "reason": "non-exact refusal accepted"}
        return {"pass": False, "reason": "did not abstain"}

    if cat == "rag_grounded_negative":
        if REFUSAL_RE.search(response_text) or FALLBACK_STRING in response_text:
            return {"pass": True, "reason": "disclaimed"}
        return {"pass": False, "reason": "did not disclaim on weak retrieval"}

    if cat == "tool_call":
        expected_tool = entry.get("expected_tool")
        used = [tc.get("name") for tc in tool_calls]
        if expected_tool and expected_tool not in used:
            return {
                "pass": False,
                "reason": f"expected tool {expected_tool!r}, used {used or 'none'}",
            }
        # Optional: check args contain a substring (loose match — small
        # models reword the user's query when they call search_notes /
        # web_search, so exact-match arguments would be too brittle).
        args_check = entry.get("expected_args_contains")
        if args_check:
            args_blob = json.dumps([tc.get("args") for tc in tool_calls]).lower()
            if args_check.lower() not in args_blob:
                return {
                    "pass": False,
                    "reason": f"args missing {args_check!r} in {args_blob[:120]}",
                }
        result_check = entry.get("expected_result_contains")
        if result_check:
            result_blob = json.dumps([tc.get("result") for tc in tool_calls]).lower()
            if result_check.lower() not in result_blob:
                return {
                    "pass": False,
                    "reason": f"result missing {result_check!r}",
                }
        return {"pass": True, "reason": "tool dispatched correctly"}

    return {"pass": False, "reason": f"unknown category: {cat}"}


# ── Per-layer activity extraction ─────────────────────────────────────────


def extract_activity(rag_metrics: dict | None) -> dict:
    """Pull per-layer activity from the inference metadata. Each value is
    optional — the runner doesn't crash when a layer is disabled."""
    if not isinstance(rag_metrics, dict):
        return {}
    out: dict[str, Any] = {}
    cit = rag_metrics.get("citations") or {}
    if cit:
        out["citations"] = {
            "cited": cit.get("cited_count", 0),
            "invalid": len(cit.get("invalid_indices") or []),
            "weak": len(cit.get("weak_citations") or []),
            "unsupported": len(cit.get("unsupported_claims") or []),
        }
    cove = rag_metrics.get("verification") or {}
    if cove:
        out["cove"] = {
            "applied": bool(cove.get("applied")),
            "skipped_reason": cove.get("skipped_reason"),
            "claims_checked": cove.get("claims_checked", 0),
            "unsupported": cove.get("unsupported_count", 0),
            "json_failures": cove.get("verify_json_failures", 0),
            "decompose_fallback": bool(cove.get("decompose_fallback")),
        }
    judge = rag_metrics.get("judge") or {}
    if judge:
        out["judge"] = {
            "applied": bool(judge.get("applied")),
            "skipped_reason": judge.get("skipped_reason"),
            "scored": judge.get("scored_count", 0),
            "flagged": judge.get("flagged_count", 0),
            "threshold": judge.get("threshold"),
        }
    sc = rag_metrics.get("selfcheck") or {}
    if sc:
        out["selfcheck"] = {
            "applied": bool(sc.get("applied")),
            "skipped_reason": sc.get("skipped_reason"),
            "samples": sc.get("samples_used", 0),
            "checked": sc.get("sentences_checked", 0),
            "flagged": sc.get("flagged_count", 0),
            "backend": sc.get("backend"),
        }
    return out


# ── Aggregation ───────────────────────────────────────────────────────────


def summarise(records: list[dict]) -> dict:
    """Per-category pass rates + per-layer activity stats. Inputs are
    the raw per-query records produced by ``run``."""
    by_category: dict[str, dict[str, int]] = {}
    layer_fired = {"cove": 0, "judge": 0, "selfcheck": 0}
    layer_flagged = {"cove": 0, "judge": 0, "selfcheck": 0}
    latencies: list[float] = []
    cove_latencies: list[float] = []

    for r in records:
        cat = r["entry"].get("category", "?")
        bucket = by_category.setdefault(cat, {"pass": 0, "fail": 0, "error": 0})
        if r.get("error"):
            bucket["error"] += 1
        elif r["verdict"]["pass"]:
            bucket["pass"] += 1
        else:
            bucket["fail"] += 1

        latencies.append(r["latency_s"])
        activity = r.get("activity") or {}
        for layer in ("cove", "judge", "selfcheck"):
            la = activity.get(layer) or {}
            if la.get("applied"):
                layer_fired[layer] += 1
                # "Flagged" means the layer found at least one issue.
                # Use the layer's specific counter.
                if layer == "cove" and la.get("unsupported", 0) > 0:
                    layer_flagged[layer] += 1
                if layer == "judge" and la.get("flagged", 0) > 0:
                    layer_flagged[layer] += 1
                if layer == "selfcheck" and la.get("flagged", 0) > 0:
                    layer_flagged[layer] += 1
                if layer == "cove":
                    cove_latencies.append(r["latency_s"])

    by_cat_summary = {}
    for cat, b in by_category.items():
        total = b["pass"] + b["fail"] + b["error"]
        rate = b["pass"] / total if total else 0.0
        by_cat_summary[cat] = {
            **b,
            "total": total,
            "pass_rate": round(rate, 3),
        }

    overall_pass = sum(b["pass"] for b in by_category.values())
    overall_total = sum(
        b["pass"] + b["fail"] + b["error"] for b in by_category.values()
    )
    return {
        "by_category": by_cat_summary,
        "overall": {
            "pass": overall_pass,
            "total": overall_total,
            "pass_rate": round(overall_pass / overall_total, 3) if overall_total else 0.0,
        },
        "layer_activity": {
            "fired": layer_fired,
            "flagged_when_fired": layer_flagged,
        },
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0,
            "median": round(statistics.median(latencies), 3) if latencies else 0,
            "p90": round(_percentile(latencies, 90), 3) if latencies else 0,
            "max": round(max(latencies), 3) if latencies else 0,
            "cove_mean": round(statistics.mean(cove_latencies), 3) if cove_latencies else None,
        },
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── Main run loop ─────────────────────────────────────────────────────────


def load_golden(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def run(
    gateway: str,
    password: str,
    golden: Path,
    output: Path,
    label: str | None,
    mode: str,
    skip_categories: set[str],
    verbose: bool,
) -> dict:
    entries = [
        e for e in load_golden(golden)
        if e.get("category") not in skip_categories
    ]
    if not entries:
        raise SystemExit("No entries to run after applying --skip filters.")

    print(f"[runner] {len(entries)} queries from {golden.name} → {gateway}")
    print(f"[runner] mode={mode} label={label or '(none)'}")
    token = login(gateway, password)
    print("[runner] auth OK")

    records: list[dict] = []
    for i, entry in enumerate(entries, 1):
        eid = entry.get("id", f"q{i}")
        cat = entry.get("category", "?")
        query = entry["query"]
        print(f"[runner] {i:>3}/{len(entries)} [{cat:<22}] {eid}", flush=True)
        try:
            resp, latency = chat(gateway, token, query, mode=mode)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            records.append({
                "entry": entry,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": 0.0,
                "verdict": {"pass": False, "reason": "transport error"},
                "activity": {},
            })
            continue
        msg = resp.get("message") or {}
        text = msg.get("content") or ""
        inf = resp.get("inference") or {}
        rm = inf.get("rag_metrics") or {}
        tool_calls = inf.get("tool_calls") or []
        verdict = evaluate(entry, text, tool_calls)
        activity = extract_activity(rm)
        rec = {
            "entry": entry,
            "response_chars": len(text),
            "latency_s": round(latency, 3),
            "verdict": verdict,
            "activity": activity,
            "mode_used": inf.get("mode_used"),
        }
        if verbose:
            rec["response_text"] = text
            rec["tool_calls"] = tool_calls
        records.append(rec)
        if not verdict["pass"]:
            print(f"          ✗ {verdict['reason']}")

    summary = summarise(records)
    report = {
        "label": label,
        "gateway": gateway,
        "mode": mode,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    output.write_text(json.dumps(report, indent=2))
    print()
    print(f"[runner] wrote {output}")
    print(f"[runner] overall pass: {summary['overall']['pass']}/"
          f"{summary['overall']['total']} ({summary['overall']['pass_rate']:.1%})")
    print(f"[runner] mean latency: {summary['latency_seconds']['mean']}s "
          f"(p90 {summary['latency_seconds']['p90']}s)")
    fired = summary["layer_activity"]["fired"]
    flagged = summary["layer_activity"]["flagged_when_fired"]
    for layer in ("cove", "judge", "selfcheck"):
        print(f"[runner] {layer:<10} fired {fired[layer]:>3}, "
              f"flagged on {flagged[layer]:>3}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://localhost:8000",
                        help="Gateway base URL (default :8000).")
    parser.add_argument("--password", required=True,
                        help="Single-user mode password (or set initial password).")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN,
                        help="JSONL golden set (default golden_set.jsonl).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the JSON report.")
    parser.add_argument("--label", default=None,
                        help="Free-text label for ablation runs (e.g. 'cove-only').")
    parser.add_argument("--mode", default="instant",
                        choices=("instant", "thinking"),
                        help="Inference tier (default instant).")
    parser.add_argument("--skip", default="",
                        help="Comma-separated category names to skip "
                             "(e.g. 'tool_call,rag_grounded' for non-stack tests).")
    parser.add_argument("--verbose", action="store_true",
                        help="Persist full response text + tool calls per record.")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    try:
        run(
            gateway=args.gateway, password=args.password,
            golden=args.golden, output=args.output,
            label=args.label, mode=args.mode,
            skip_categories=skip, verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n[runner] interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
