"""
memory/long_term.py
────────────────────
Long-term persistent memory backed by SQLite.
Stores:
  - user preferences  (key/value)
  - recurring patterns (e.g. "opens VS Code every morning")
  - persistent facts   (things the user has explicitly told AIDA)
  - action log         (tool calls + outcomes, for self-improvement)
  - suggestion log     (suggestions shown + whether accepted)

Designed to evolve toward embedding/vector search if needed.
Currently uses simple text matching and recency.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("aida.memory.long_term")

_DB_PATH = Path(__file__).parent.parent / "data" / "long_term.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS preferences (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT NOT NULL,
            source      TEXT DEFAULT 'user',
            created_at  TEXT NOT NULL,
            tags        TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS patterns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern     TEXT NOT NULL,
            count       INTEGER DEFAULT 1,
            last_seen   TEXT NOT NULL,
            metadata    TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS action_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            tool        TEXT NOT NULL,
            input       TEXT,
            outcome     TEXT,
            success     INTEGER DEFAULT 1,
            duration_ms INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS suggestion_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            suggestion  TEXT NOT NULL,
            context     TEXT,
            accepted    INTEGER DEFAULT NULL
        );
    """)
    conn.commit()


class LongTermMemory:
    def __init__(self):
        self._conn = _get_conn()
        _init_db(self._conn)
        log.info("LongTermMemory ready: %s", _DB_PATH)

    # ── Preferences ───────────────────────────────────────────────────────────

    def set_preference(self, key: str, value: Any) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value), now)
        )
        self._conn.commit()

    def get_preference(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key=?", (key,)
        ).fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except Exception:
                return row["value"]
        return default

    def all_preferences(self) -> dict:
        rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except Exception:
                result[row["key"]] = row["value"]
        return result

    # ── Facts ────────────────────────────────────────────────────────────────

    def add_fact(self, content: str, source: str = "user",
                 tags: Optional[list] = None) -> int:
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO facts (content, source, created_at, tags) VALUES (?,?,?,?)",
            (content, source, now, json.dumps(tags or []))
        )
        self._conn.commit()
        log.debug("Fact added: %r", content[:60])
        return cur.lastrowid

    def search_facts(self, query: str, limit: int = 5) -> list[dict]:
        q = f"%{query.lower()}%"
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE LOWER(content) LIKE ? ORDER BY created_at DESC LIMIT ?",
            (q, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_facts(self, n: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM facts ORDER BY created_at DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Patterns ─────────────────────────────────────────────────────────────

    def record_pattern(self, pattern: str, metadata: Optional[dict] = None) -> None:
        now = datetime.now().isoformat()
        existing = self._conn.execute(
            "SELECT id, count FROM patterns WHERE pattern=?", (pattern,)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE patterns SET count=count+1, last_seen=?, metadata=? WHERE id=?",
                (now, json.dumps(metadata or {}), existing["id"])
            )
        else:
            self._conn.execute(
                "INSERT INTO patterns (pattern, count, last_seen, metadata) VALUES (?,1,?,?)",
                (pattern, now, json.dumps(metadata or {}))
            )
        self._conn.commit()

    def top_patterns(self, n: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM patterns ORDER BY count DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Action log ────────────────────────────────────────────────────────────

    def log_action(self, tool: str, input_text: str, outcome: str,
                   success: bool = True, duration_ms: int = 0) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO action_log (timestamp, tool, input, outcome, success, duration_ms) "
            "VALUES (?,?,?,?,?,?)",
            (now, tool, input_text[:500], outcome[:500], int(success), duration_ms)
        )
        self._conn.commit()

    def recent_actions(self, n: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM action_log ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Suggestion log ────────────────────────────────────────────────────────

    def log_suggestion(self, suggestion: str, context: str = "") -> int:
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO suggestion_log (timestamp, suggestion, context) VALUES (?,?,?)",
            (now, suggestion, context[:300])
        )
        self._conn.commit()
        return cur.lastrowid

    def mark_suggestion(self, suggestion_id: int, accepted: bool) -> None:
        self._conn.execute(
            "UPDATE suggestion_log SET accepted=? WHERE id=?",
            (int(accepted), suggestion_id)
        )
        self._conn.commit()

    def suggestion_acceptance_rate(self) -> float:
        row = self._conn.execute(
            "SELECT COUNT(*) total, SUM(CASE WHEN accepted=1 THEN 1 ELSE 0 END) accepted "
            "FROM suggestion_log WHERE accepted IS NOT NULL"
        ).fetchone()
        if row and row["total"] > 0:
            return row["accepted"] / row["total"]
        return 0.0
