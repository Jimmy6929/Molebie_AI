"""
Seed a corpus into the ablation data dir so RAG-grounded golden-set
questions actually have content to retrieve from.

The default `ablation.sh` wipes DATA_DIR to /tmp/molebie_ablation_data
(empty), which makes CoVe and Judge gates short-circuit on every query
(no chunks → no fire). This script populates DATA_DIR with a curated
set of project docs that contain the answers the rag_grounded golden
set asks for, so the ablation can produce real Phase 3 signal.

Pre-condition: a gateway must be running at --gateway. The caller
boots it with COVE/JUDGE/SELFCHECK off (we only need RAG infrastructure
here, not the post-processors).

Usage (called by ablation.sh seed step):
    seed_corpus.py --gateway http://127.0.0.1:8765 --password smoketest
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent

DEFAULT_CORPUS = [
    "gateway/tests/eval/corpus/rag-config.md",
    "gateway/tests/eval/corpus/sampling-and-budgets.md",
    "gateway/tests/eval/corpus/prompt-and-citations.md",
    "gateway/tests/eval/corpus/cove-verifier.md",
    "gateway/tests/eval/corpus/selfcheck-and-judge.md",
]


def login_or_register(gateway: str, password: str) -> str:
    body = json.dumps({"password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"{gateway}/auth/login-simple", data=body, method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["token"]


def upload(gateway: str, token: str, file_path: Path) -> dict:
    """Multipart upload of one file to /documents/upload."""
    suffix = file_path.suffix.lower()
    if suffix == ".md":
        content_type = "text/markdown"
    elif suffix == ".txt":
        content_type = "text/plain"
    else:
        content_type = mimetypes.guess_type(file_path.name)[0] or "text/plain"

    boundary = f"----MolebieFormBoundary{int(time.time() * 1000)}"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{file_path.name}"\r\n'
    ).encode()
    body += f"Content-Type: {content_type}\r\n\r\n".encode()
    body += file_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{gateway}/documents/upload", data=body, method="POST",
    )
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://127.0.0.1:8765")
    parser.add_argument("--password", default="smoketest")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--files", nargs="+",
        help="Override the default corpus with explicit paths (relative to --root).",
    )
    args = parser.parse_args()

    corpus = args.files if args.files else DEFAULT_CORPUS

    print(f"[seed] login → {args.gateway}")
    try:
        token = login_or_register(args.gateway, args.password)
    except Exception as exc:
        print(f"[seed] login failed: {exc}")
        return 2

    uploaded = 0
    skipped = 0
    for rel in corpus:
        path = args.root / rel
        if not path.exists():
            print(f"[seed]   SKIP missing: {rel}")
            skipped += 1
            continue
        size = path.stat().st_size
        print(f"[seed]   upload {rel} ({size:>6} bytes)", flush=True)
        t0 = time.perf_counter()
        try:
            r = upload(args.gateway, token, path)
            chunks = r.get("chunks_processed") or r.get("chunk_count") or "?"
            doc_id = r.get("document_id") or r.get("id") or "?"
            print(
                f"[seed]     OK chunks={chunks} doc_id={doc_id} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
            uploaded += 1
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            print(f"[seed]     HTTP {exc.code}: {detail}")
            skipped += 1
        except Exception as exc:
            print(f"[seed]     ERROR: {type(exc).__name__}: {exc}")
            skipped += 1

    print(f"[seed] DONE: uploaded={uploaded} skipped={skipped}")
    return 0 if uploaded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
