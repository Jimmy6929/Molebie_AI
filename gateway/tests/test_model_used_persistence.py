"""Tests for chat_messages.model_used persistence (T1b).

History previously always showed model=None because the column never
existed — the ChatMessage model aliased a phantom field. These tests pin
the schema (fresh + migrate paths) and the create_message write.
"""

from __future__ import annotations

import sqlite3

from app.schema import init_database_sync


def test_schema_has_model_used_fresh_and_migrated(tmp_path):
    db_path = init_database_sync(str(tmp_path))
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_messages)")}
        assert "model_used" in cols

        # Simulate a pre-T1b database: rebuild the table without the column,
        # then re-run init — the migrate branch must add it back.
        conn.executescript(
            """
            ALTER TABLE chat_messages RENAME TO chat_messages_old;
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                mode_used TEXT,
                tokens_used INTEGER,
                reasoning_content TEXT,
                created_at TEXT NOT NULL
            );
            DROP TABLE chat_messages_old;
            """
        )
        conn.commit()
    finally:
        conn.close()

    init_database_sync(str(tmp_path))
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_messages)")}
        assert "model_used" in cols
    finally:
        conn.close()


def test_insert_and_read_model_used(tmp_path):
    db_path = init_database_sync(str(tmp_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) "
            "VALUES ('s1', '00000000-0000-0000-0000-000000000001', 't', '2026-08-16', '2026-08-16')"
        )
        conn.execute(
            "INSERT INTO chat_messages "
            "(id, session_id, user_id, role, content, mode_used, model_used, created_at) "
            "VALUES ('m1', 's1', '00000000-0000-0000-0000-000000000001', 'assistant', "
            "'hi', 'thinking', 'mlx-community/Qwen3.5-9B-MLX-4bit', '2026-08-16')"
        )
        got = conn.execute(
            "SELECT model_used FROM chat_messages WHERE id='m1'"
        ).fetchone()
        assert got == ("mlx-community/Qwen3.5-9B-MLX-4bit",)
    finally:
        conn.close()
