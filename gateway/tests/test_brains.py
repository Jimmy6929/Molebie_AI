"""Tests for brain (corpus) scoping — Phase 3.

A "brain" is the top-level folder of a vault note's relative_path, derived at
query time (no stored column). Covers the request normalizer (`_norm_brain`)
and the SQL derivation (`_BRAIN_SQL`: top-level folder, root → "Inbox") run
directly against plain in-memory SQLite — the exact expression the real
search/`list_brains` queries embed.
"""

import sqlite3

from app.routes.chat import _norm_brain
from app.services.database import _BRAIN_SQL


def test_norm_brain_treats_none_empty_all_as_no_scope():
    assert _norm_brain(None) is None
    assert _norm_brain("") is None
    assert _norm_brain("All") is None
    assert _norm_brain("all") is None
    assert _norm_brain("  ALL  ") is None


def test_norm_brain_keeps_and_trims_a_real_brain():
    assert _norm_brain("Areas") == "Areas"
    assert _norm_brain("  Projects ") == "Projects"


def test_brain_sql_derives_top_folder_root_inbox_and_status_filter():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE documents (id TEXT, user_id TEXT, relative_path TEXT, status TEXT)"
    )
    conn.executemany(
        "INSERT INTO documents VALUES (?,?,?,?)",
        [
            ("1", "u1", "Areas/Health.md", "completed"),
            ("2", "u1", "Areas/Career.md", "completed"),
            ("3", "u1", "Projects/Molebie.md", "completed"),
            ("4", "u1", "Dashboard.md", "completed"),            # root -> Inbox
            ("5", "u1", "One Up on Wall St/Ch1.md", "completed"),
            ("6", "u1", "Areas/Wip.md", "processing"),           # excluded (status)
            ("7", "u2", "Areas/Other.md", "completed"),          # excluded (user)
        ],
    )
    cur = conn.execute(
        f"SELECT {_BRAIN_SQL} AS brain, COUNT(*) AS n FROM documents AS d "
        f"WHERE d.user_id = ? AND d.status = 'completed' GROUP BY brain",
        ("u1",),
    )
    brains = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    assert brains == {"Areas": 2, "Projects": 1, "Inbox": 1, "One Up on Wall St": 1}


def test_brain_sql_handles_empty_and_null_paths_as_inbox():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE documents (relative_path TEXT)")
    conn.executemany("INSERT INTO documents VALUES (?)", [(None,), ("",), ("note.md",)])
    cur = conn.execute(f"SELECT {_BRAIN_SQL} FROM documents AS d")
    vals = [row[0] for row in cur.fetchall()]
    conn.close()
    assert vals == ["Inbox", "Inbox", "Inbox"]
