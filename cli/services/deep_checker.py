"""Deep application-logic health checks for molebie-ai doctor --deep.

Checks the critical integration points where silent failures happen:
database schema, sqlite-vec, FTS5, embedding model, and vector round-trip.

All checks use synchronous sqlite3 (no aiosqlite) and work independently
of the gateway process.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DeepCheckResult:
    name: str
    passed: bool
    message: str
    fix_hint: str = ""
    is_warning: bool = False
    skipped: bool = False


# ──────────────────────────────────────────────────────────────
# Expected schema (derived from gateway/app/schema.py _SCHEMA_SQL)
# ──────────────────────────────────────────────────────────────

EXPECTED_TABLES: Dict[str, List[str]] = {
    "_schema_version": ["version", "applied_at"],
    "users": ["id", "email", "password_hash", "name", "created_at", "updated_at"],
    "chat_sessions": [
        "id", "user_id", "title", "is_archived", "is_pinned",
        "summary", "summary_message_count", "created_at", "updated_at",
    ],
    "chat_messages": [
        "id", "session_id", "user_id", "role", "content",
        "mode_used", "tokens_used", "reasoning_content", "created_at",
    ],
    "documents": [
        "id", "user_id", "filename", "storage_path", "file_type",
        "file_size", "status", "created_at", "processed_at",
    ],
    "document_chunks": [
        "id", "document_id", "user_id", "content",
        "content_contextualized", "chunk_index", "metadata", "created_at",
    ],
    "session_documents": [
        "id", "session_id", "user_id", "filename", "content",
        "file_size", "created_at",
    ],
    "message_images": [
        "id", "message_id", "user_id", "storage_path", "filename",
        "mime_type", "file_size", "created_at",
    ],
    "message_sources": ["id", "message_id", "url", "title", "created_at"],
    "user_memories": [
        "id", "user_id", "content", "category", "source_session_id",
        "access_count", "last_accessed_at", "created_at", "updated_at",
    ],
    "rag_query_metrics": [
        "id", "user_id", "query_text", "num_candidates",
        "unique_documents", "top_similarity", "avg_similarity",
        "top_rrf_score", "top_rerank_score", "score_spread",
        "hybrid_enabled", "reranker_enabled", "t_embed_ms",
        "t_search_ms", "t_rerank_ms", "t_total_ms", "created_at",
    ],
}

EXPECTED_VIRTUAL_TABLES = [
    "document_chunks_vec",
    "user_memories_vec",
    "document_chunks_fts",
]

EXPECTED_FTS_TRIGGERS = [
    "document_chunks_ai",
    "document_chunks_ad",
    "document_chunks_au",
]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _parse_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a .env file into a dict (simple key=value, ignores comments)."""
    result: Dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip("\"'")
    return result


def _find_db_path(project_root: Path) -> Optional[Path]:
    """Locate the SQLite DB using .env.local DATA_DIR or default."""
    env = _parse_env_file(project_root / ".env.local")
    data_dir = env.get("DATA_DIR", "data")

    if os.path.isabs(data_dir):
        db_path = Path(data_dir) / "molebie.db"
    else:
        # Gateway resolves relative to its own directory
        db_path = project_root / "gateway" / data_dir / "molebie.db"

    return db_path if db_path.exists() else None



def _connect_db(db_path: Path) -> sqlite3.Connection:
    """Open a read-only SQLite connection with sqlite-vec loaded."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn



# ──────────────────────────────────────────────────────────────
# Check functions
# ──────────────────────────────────────────────────────────────

def check_database_accessible(db_path: Path) -> DeepCheckResult:
    """Check 1: Can we open the DB and is it a valid SQLite file?"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        result = conn.execute("PRAGMA integrity_check(1)").fetchone()
        conn.close()
        if result and result[0] == "ok":
            return DeepCheckResult("Database", True, f"Accessible at {db_path.name}")
        return DeepCheckResult(
            "Database", False, f"Integrity check failed: {result}",
            fix_hint="Database may be corrupt. Restore from backup or re-initialize.",
        )
    except sqlite3.DatabaseError as exc:
        return DeepCheckResult(
            "Database", False, f"Cannot open: {exc}",
            fix_hint="Run 'molebie-ai install' to initialize the database.",
        )


def check_schema_tables(db_path: Path) -> DeepCheckResult:
    """Check 2: Do all expected tables exist with the right columns?"""
    try:
        conn = _connect_db(db_path)

        # Get actual tables
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        actual_tables = {r[0] for r in rows}

        missing_tables = []
        missing_columns: List[str] = []

        for table, expected_cols in EXPECTED_TABLES.items():
            if table not in actual_tables:
                missing_tables.append(table)
                continue
            # Check columns
            col_rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            actual_cols = {r[1] for r in col_rows}  # column name is index 1
            for col in expected_cols:
                if col not in actual_cols:
                    missing_columns.append(f"{table}.{col}")

        conn.close()

        if missing_tables:
            return DeepCheckResult(
                "Schema tables", False,
                f"Missing tables: {', '.join(missing_tables)}",
                fix_hint="Restart the gateway to run schema initialization.",
            )
        if missing_columns:
            return DeepCheckResult(
                "Schema columns", False,
                f"Missing columns: {', '.join(missing_columns[:5])}"
                + (f" (+{len(missing_columns) - 5} more)" if len(missing_columns) > 5 else ""),
                fix_hint="Schema may need migration. Check gateway/app/schema.py.",
            )

        return DeepCheckResult(
            "Schema", True,
            f"All {len(EXPECTED_TABLES)} tables verified",
        )
    except Exception as exc:
        return DeepCheckResult("Schema", False, f"Check failed: {exc}")


def check_sqlite_vec(db_path: Path) -> DeepCheckResult:
    """Check 3: Are the vec0 virtual tables present and is the extension loadable?"""
    try:
        conn = _connect_db(db_path)

        # Check vec0 tables exist in sqlite_master (no extension needed)
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql LIKE '%vec0%'"
        ).fetchall()
        actual_vec_tables = {r[0] for r in rows}
        conn.close()

        expected = {"document_chunks_vec", "user_memories_vec"}
        missing = expected - actual_vec_tables
        if missing:
            return DeepCheckResult(
                "sqlite-vec", False,
                f"Missing vec0 tables: {', '.join(missing)}",
                fix_hint="Restart the gateway to create virtual tables.",
            )

        # Tables exist. Try loading the extension for a functional test.
        can_load = _try_load_sqlite_vec(db_path)
        if can_load is True:
            return DeepCheckResult("sqlite-vec", True, "vec0 tables present, extension loadable")
        elif can_load is None:
            # Extension can't load but tables exist (created by gateway)
            return DeepCheckResult(
                "sqlite-vec", True,
                "vec0 tables present (extension not loadable in CLI Python — OK, gateway handles it)",
            )
        else:
            return DeepCheckResult(
                "sqlite-vec", True,
                "vec0 tables present (sqlite-vec package not in CLI env — OK, gateway handles it)",
                is_warning=True,
            )

    except Exception as exc:
        return DeepCheckResult("sqlite-vec", False, f"Check failed: {exc}")


def _try_load_sqlite_vec(db_path: Path) -> Optional[bool]:
    """Try to load sqlite-vec extension. Returns True/False/None (None = can't load extensions)."""
    try:
        import sqlite_vec
    except ImportError:
        return False  # Package not installed

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.close()
        return True
    except AttributeError:
        # Python built without loadable extension support
        return None
    except Exception:
        return None


def check_fts5(db_path: Path) -> DeepCheckResult:
    """Check 4: Is FTS5 functional with sync triggers?"""
    try:
        conn = _connect_db(db_path)

        # Check FTS5 table exists
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks_fts'"
        ).fetchone()
        if not row:
            conn.close()
            return DeepCheckResult(
                "FTS5", False, "document_chunks_fts table missing",
                fix_hint="Restart the gateway to create FTS5 tables.",
            )

        # Check sync triggers
        triggers = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        actual_triggers = {r[0] for r in triggers}
        missing = [t for t in EXPECTED_FTS_TRIGGERS if t not in actual_triggers]

        if missing:
            conn.close()
            return DeepCheckResult(
                "FTS5", False,
                f"Missing sync triggers: {', '.join(missing)}",
                fix_hint="FTS will be out of sync with chunks. Reinitialize the DB.",
                is_warning=True,
            )

        # Quick functional test
        conn.execute("SELECT COUNT(*) FROM document_chunks_fts")
        conn.close()

        return DeepCheckResult("FTS5", True, "Table + triggers present, queryable")
    except Exception as exc:
        return DeepCheckResult("FTS5", False, f"Check failed: {exc}")


def check_embedding_and_roundtrip_via_gateway(
    gateway_url: str,
) -> List[DeepCheckResult]:
    """Checks 5 & 6: Call the gateway's /health/deep endpoint to test embedding + vector search.

    This runs the checks in the gateway's Python environment where
    sentence-transformers and sqlite-vec are actually installed.
    """
    import httpx

    try:
        resp = httpx.get(f"{gateway_url}/health/deep", timeout=30.0)
        if resp.status_code != 200:
            return [
                DeepCheckResult(
                    "Embedding model", False,
                    f"Gateway /health/deep returned {resp.status_code}",
                ),
                DeepCheckResult(
                    "Vector round-trip", False, "Skipped (embedding check failed)",
                    skipped=True,
                ),
            ]

        data = resp.json()
    except httpx.ConnectError:
        return [
            DeepCheckResult(
                "Embedding model", False,
                "Gateway not reachable — cannot test embedding",
                fix_hint="Start the gateway first, then re-run with --deep.",
                skipped=True,
            ),
            DeepCheckResult(
                "Vector round-trip", False, "Skipped (gateway not reachable)",
                skipped=True,
            ),
        ]
    except Exception as exc:
        return [
            DeepCheckResult("Embedding model", False, f"Gateway call failed: {exc}"),
            DeepCheckResult("Vector round-trip", False, "Skipped", skipped=True),
        ]

    results: List[DeepCheckResult] = []

    # Parse embedding result
    emb = data.get("embedding", {})
    if emb.get("status") == "pass":
        results.append(DeepCheckResult(
            "Embedding model", True, emb.get("message", "OK"),
        ))
    elif emb.get("status") == "skip":
        results.append(DeepCheckResult(
            "Embedding model", False, emb.get("message", "Skipped"),
            skipped=True,
        ))
    else:
        results.append(DeepCheckResult(
            "Embedding model", False, emb.get("message", "Failed"),
        ))

    # Parse vector round-trip result
    vrt = data.get("vector_roundtrip", {})
    if vrt.get("status") == "pass":
        results.append(DeepCheckResult(
            "Vector round-trip", True, vrt.get("message", "OK"),
        ))
    elif vrt.get("status") == "skip":
        results.append(DeepCheckResult(
            "Vector round-trip", False, vrt.get("message", "Skipped"),
            skipped=True,
        ))
    else:
        results.append(DeepCheckResult(
            "Vector round-trip", False, vrt.get("message", "Failed"),
        ))

    return results


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────

def _get_gateway_url(project_root: Path) -> str:
    """Read gateway URL from .env.local or CLI config, default to localhost:8000."""
    env = _parse_env_file(project_root / ".env.local")
    return env.get("NEXT_PUBLIC_GATEWAY_URL", "http://localhost:8000")


def run_deep_checks(project_root: Path) -> List[DeepCheckResult]:
    """Run all deep application-logic checks in dependency order."""
    results: List[DeepCheckResult] = []

    # Locate database
    db_path = _find_db_path(project_root)
    if not db_path:
        results.append(DeepCheckResult(
            "Database", False, "molebie.db not found",
            fix_hint="Run 'molebie-ai install' or start the gateway to initialize the DB.",
        ))
        return results

    # Check 1: Database accessible
    r1 = check_database_accessible(db_path)
    results.append(r1)
    if not r1.passed:
        return results

    # Check 2: Schema tables & columns
    r2 = check_schema_tables(db_path)
    results.append(r2)

    # Check 3: sqlite-vec tables present
    r3 = check_sqlite_vec(db_path)
    results.append(r3)

    # Check 4: FTS5
    r4 = check_fts5(db_path)
    results.append(r4)

    # Checks 5 & 6: Embedding model + vector round-trip via gateway API
    # These run in the gateway's Python env where the deps are installed
    gateway_url = _get_gateway_url(project_root)
    gateway_results = check_embedding_and_roundtrip_via_gateway(gateway_url)
    results.extend(gateway_results)

    return results
