"""
Migrate the sqlite-vec virtual tables to a new embedding dimension.

When you change the embedding model, the new vectors won't match the
existing ``document_chunks_vec`` / ``user_memories_vec`` tables — sqlite-vec
locks the dim at table-creation time and there's no ALTER. This script
DROPs and recreates them at the configured ``embedding_dim``.

What it preserves:
  * ``documents`` rows + their files in storage
  * ``document_chunks`` rows (content + metadata) — the text is fine,
    only the vectors are bound to the dim
  * Everything outside RAG (sessions, messages, memories' rows, etc.)

What it destroys:
  * All vectors in ``document_chunks_vec`` and ``user_memories_vec``

After running this, restart the gateway and call
``POST /documents/reindex/full`` to repopulate vectors with the new model.

Usage:
    cd gateway
    ../.venv/bin/python -m scripts.migrate_embedding_dim
    ../.venv/bin/python -m scripts.migrate_embedding_dim --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Ensure ``app`` package resolves when run via ``python -m scripts.X`` from gateway/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.schema import (  # noqa: E402
    _detect_vec_dim,
    _get_db_path,
    _load_vec_extension,
    _vec_table_sql,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect mismatch and print the planned action without changing anything.",
    )
    args = parser.parse_args()

    settings = get_settings()
    target_dim = settings.embedding_dim
    db_path = _get_db_path(settings.data_dir)

    if not Path(db_path).exists():
        print(f"[migrate] No database at {db_path} — nothing to do.")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        _load_vec_extension(conn)
        actual_dim = _detect_vec_dim(conn)

        print(f"[migrate] Database: {db_path}")
        print(f"[migrate] Configured embedding_dim: {target_dim}")
        print(f"[migrate] Detected vec table dim:   {actual_dim}")

        if actual_dim is None:
            print("[migrate] Vec tables not present — initial schema init will create them.")
            return 0

        if actual_dim == target_dim:
            print("[migrate] Already aligned — nothing to do.")
            return 0

        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM document_chunks_vec"
        ).fetchone()[0]
        mem_count = conn.execute(
            "SELECT COUNT(*) FROM user_memories_vec"
        ).fetchone()[0]
        print(
            f"[migrate] Will DROP {chunk_count} chunk vectors + "
            f"{mem_count} memory vectors and recreate at dim={target_dim}."
        )
        print("[migrate] Underlying chunk / memory rows are preserved.")
        print(
            "[migrate] After this, restart the gateway and run "
            "POST /documents/reindex/full to rebuild chunk vectors."
        )

        if args.dry_run:
            print("[migrate] --dry-run: no changes made.")
            return 0

        conn.execute("DROP TABLE IF EXISTS document_chunks_vec")
        conn.execute("DROP TABLE IF EXISTS user_memories_vec")
        for stmt in _vec_table_sql(target_dim).strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()
        print(f"[migrate] Done. Vec tables recreated at dim={target_dim}.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
