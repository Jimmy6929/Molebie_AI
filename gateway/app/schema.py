"""
SQLite schema initialization for Molebie AI.

Replaces the 14 Supabase migration files with a single Python module
that creates all tables, FTS5 virtual tables, and sqlite-vec virtual tables.
"""

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import sqlite_vec

SCHEMA_VERSION = 1

# Default single-user ID (used when AUTH_MODE=single)
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_USER_EMAIL = "local@molebie.local"


def _get_db_path(data_dir: str) -> str:
    """Return the full path to the SQLite database file."""
    return os.path.join(data_dir, "molebie.db")


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a synchronous connection."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


async def _load_vec_extension_async(db: aiosqlite.Connection) -> None:
    """Load the sqlite-vec extension into an async connection.

    Uses aiosqlite's thread-safe async wrappers to load the extension
    on the correct internal thread.
    """
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)


_SCHEMA_SQL = """
-- ============================================
-- Schema version tracking
-- ============================================
CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);

-- ============================================
-- USERS TABLE (replaces Supabase Auth)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ============================================
-- CHAT SESSIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Chat',
    is_archived INTEGER NOT NULL DEFAULT 0,
    is_pinned INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    summary_message_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
    ON chat_sessions(user_id, updated_at DESC);

-- ============================================
-- CHAT MESSAGES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    mode_used TEXT CHECK (mode_used IN ('instant', 'thinking', 'thinking_harder')),
    tokens_used INTEGER,
    reasoning_content TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages(session_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON chat_messages(user_id);

-- ============================================
-- DOCUMENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_created
    ON documents(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- ============================================
-- DOCUMENT CHUNKS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_contextualized TEXT,
    chunk_index INTEGER NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
    ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id ON document_chunks(user_id);

-- ============================================
-- SESSION DOCUMENTS TABLE (Attach to Chat)
-- ============================================
CREATE TABLE IF NOT EXISTS session_documents (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_documents_session_id
    ON session_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_session_documents_user_id
    ON session_documents(user_id);

-- ============================================
-- MESSAGE IMAGES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS message_images (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    filename TEXT,
    mime_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_message_images_message_id ON message_images(message_id);
CREATE INDEX IF NOT EXISTS idx_message_images_user_id ON message_images(user_id);

-- ============================================
-- MESSAGE SOURCES TABLE (Web Search Results)
-- ============================================
CREATE TABLE IF NOT EXISTS message_sources (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_message_sources_message_id ON message_sources(message_id);

-- ============================================
-- USER MEMORIES TABLE (Cross-Session Facts)
-- ============================================
CREATE TABLE IF NOT EXISTS user_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'preference'
        CHECK (category IN ('preference', 'background', 'project', 'instruction')),
    source_session_id TEXT REFERENCES chat_sessions(id) ON DELETE SET NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memories_user_category
    ON user_memories(user_id, category);

-- ============================================
-- RAG QUERY METRICS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS rag_query_metrics (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query_text TEXT,
    num_candidates INTEGER,
    unique_documents INTEGER,
    top_similarity REAL,
    avg_similarity REAL,
    top_rrf_score REAL,
    top_rerank_score REAL,
    score_spread REAL,
    hybrid_enabled INTEGER,
    reranker_enabled INTEGER,
    t_embed_ms REAL,
    t_search_ms REAL,
    t_rerank_ms REAL,
    t_total_ms REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rag_query_metrics_user_created
    ON rag_query_metrics(user_id, created_at DESC);
"""


def _fts5_sql() -> str:
    """Return SQL for FTS5 virtual table creation."""
    return """
    CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
        content,
        content_contextualized,
        content=document_chunks,
        content_rowid=rowid
    );

    -- Triggers to keep FTS in sync with document_chunks
    CREATE TRIGGER IF NOT EXISTS document_chunks_ai AFTER INSERT ON document_chunks BEGIN
        INSERT INTO document_chunks_fts(rowid, content, content_contextualized)
        VALUES (new.rowid, new.content, new.content_contextualized);
    END;

    CREATE TRIGGER IF NOT EXISTS document_chunks_ad AFTER DELETE ON document_chunks BEGIN
        INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content, content_contextualized)
        VALUES ('delete', old.rowid, old.content, old.content_contextualized);
    END;

    CREATE TRIGGER IF NOT EXISTS document_chunks_au AFTER UPDATE ON document_chunks BEGIN
        INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content, content_contextualized)
        VALUES ('delete', old.rowid, old.content, old.content_contextualized);
        INSERT INTO document_chunks_fts(rowid, content, content_contextualized)
        VALUES (new.rowid, new.content, new.content_contextualized);
    END;
    """


def _vec_table_sql(embedding_dim: int) -> str:
    """Return SQL for sqlite-vec virtual tables with the given dimension."""
    return f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_vec USING vec0(
        embedding float[{embedding_dim}]
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS user_memories_vec USING vec0(
        embedding float[{embedding_dim}]
    );
    """


_VEC_DIM_RE = re.compile(r"float\[(\d+)\]")


def _detect_vec_dim(conn: sqlite3.Connection) -> int | None:
    """Read the embedding dim baked into ``document_chunks_vec``'s CREATE SQL.

    Returns the integer dim or None if the table doesn't exist / SQL doesn't
    match. Used by init_database_sync to surface dim mismatches loudly.
    """
    try:
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=? AND type='table'",
            ("document_chunks_vec",),
        )
        row = cursor.fetchone()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    match = _VEC_DIM_RE.search(row[0])
    return int(match.group(1)) if match else None


def init_database_sync(
    data_dir: str,
    embedding_dim: int = 1024,
    auth_mode: str = "single",
    default_password_hash: str | None = None,
) -> str:
    """
    Initialize the SQLite database synchronously.
    Returns the database file path.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    db_path = _get_db_path(data_dir)
    conn = sqlite3.connect(db_path)

    try:
        # Enable WAL mode for concurrent reads
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec extension
        _load_vec_extension(conn)

        # Check if already initialized
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_version'"
        )
        if cursor.fetchone():
            # Migrate: add message_sources if it doesn't exist (added in v2)
            cursor2 = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_sources'"
            )
            if not cursor2.fetchone():
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS message_sources (
                        id TEXT PRIMARY KEY,
                        message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
                        url TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_message_sources_message_id ON message_sources(message_id);
                """)
                print("[schema] Migrated: added message_sources table")

            # Check vec table dim against the configured embedding_dim. We
            # don't auto-migrate (DROP+recreate would silently destroy
            # vectors on an env-var typo); we just surface the mismatch
            # loudly so the operator runs scripts/migrate_embedding_dim.py.
            actual_dim = _detect_vec_dim(conn)
            if actual_dim is not None and actual_dim != embedding_dim:
                print(
                    f"[schema] WARNING: vec table dim={actual_dim} but "
                    f"config embedding_dim={embedding_dim}. Inserts will "
                    "fail until you run: "
                    "python -m gateway.scripts.migrate_embedding_dim "
                    "(then POST /documents/reindex/full)."
                )
            print(f"[schema] Database already initialized at {db_path}")
            conn.close()
            return db_path

        # Create all tables
        conn.executescript(_SCHEMA_SQL)

        # Create FTS5 virtual table
        conn.executescript(_fts5_sql())

        # Create sqlite-vec virtual tables
        for stmt in _vec_table_sql(embedding_dim).strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

        # Record schema version
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO _schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, now),
        )

        # In single-user mode, create the default user
        if auth_mode == "single":
            conn.execute(
                "INSERT OR IGNORE INTO users (id, email, password_hash, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    DEFAULT_USER_ID,
                    DEFAULT_USER_EMAIL,
                    default_password_hash,
                    "Local User",
                    now,
                    now,
                ),
            )

        conn.commit()
        print(f"[schema] Database initialized at {db_path} (v{SCHEMA_VERSION}, dim={embedding_dim})")
    finally:
        conn.close()

    return db_path


async def init_database(
    data_dir: str,
    embedding_dim: int = 1024,
    auth_mode: str = "single",
    default_password_hash: str | None = None,
) -> str:
    """
    Initialize the SQLite database asynchronously.
    Wraps the sync version since schema creation uses executescript.
    Returns the database file path.
    """
    import asyncio
    return await asyncio.to_thread(
        init_database_sync,
        data_dir,
        embedding_dim,
        auth_mode,
        default_password_hash,
    )


async def get_connection(data_dir: str) -> aiosqlite.Connection:
    """
    Open an async connection to the database with extensions loaded.
    Caller is responsible for closing the connection.
    """
    db_path = _get_db_path(data_dir)
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await _load_vec_extension_async(db)
    return db
