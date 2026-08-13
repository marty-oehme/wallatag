"""Optional SQLite decision log.

The decision log provides seen-dedupe (an article is marked when it is PICKED
UP, so concurrent runs: Prefect + manual: never double-process it) and
records accept/reject tag decisions (for later ranking and phase-2 AI
training). It is strictly optional: ``path=None`` selects history-less mode in
which no database is ever opened and every method is a safe no-op.

Concurrency: dedupe relies on the atomic ``INSERT OR IGNORE`` on the seen
primary key, which is safe across processes/threads as long as each process or
thread uses its own ``Store`` instance (one connection per process/thread:
sharing a single instance across threads raises ``sqlite3.ProgrammingError``
because of ``check_same_thread``); ``busy_timeout`` serializes write
contention. The natural fallback dedupe is wallabag itself: tagged articles
stop matching the untagged filter.
"""

from __future__ import annotations

import sqlite3
import time

_VALID_ACTIONS = ("accept", "reject")
# The WAL-mode transition needs an exclusive lock; on a fresh, not-yet-existing
# database two concurrent constructors can collide with "database is locked".
# Bound the retry so startup can never hang.
_SETUP_RETRY_ATTEMPTS = 10
_SETUP_RETRY_DELAY = 0.05  # seconds


class Store:
    """Persistent SQLite decision log.

    ``path=None`` selects history-less mode: a valid no-op store that does not
    touch the filesystem.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        if path is None:
            self._conn = None
            return
        self._conn = sqlite3.connect(path, timeout=30.0)
        # busy_timeout must be set EARLY so concurrent writers serialize on
        # the WAL-transition exclusive lock instead of failing immediately.
        self._conn.execute("PRAGMA busy_timeout=5000")
        # On a fresh database the WAL-mode transition needs an exclusive lock;
        # two processes constructing Store() at the same time can collide with
        # "database is locked" (the busy handler alone does not resolve this).
        # Retry the whole connection setup a bounded number of times.
        for attempt in range(_SETUP_RETRY_ATTEMPTS):
            try:
                self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS seen (
                        entry_id INTEGER PRIMARY KEY,
                        picked_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    CREATE TABLE IF NOT EXISTS decisions (
                        entry_id INTEGER NOT NULL,
                        tag TEXT NOT NULL,
                        action TEXT NOT NULL CHECK (action IN ('accept', 'reject')),
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_decisions_entry_id
                        ON decisions (entry_id);
                    """
                )
                self._conn.commit()
                break
            except sqlite3.OperationalError:
                if attempt == _SETUP_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_SETUP_RETRY_DELAY)

    # -- dedupe ----------------------------------------------------------

    def mark_seen(self, entry_id: int) -> bool:
        """Mark an article as picked up.

        Returns True if the row was newly inserted, False if it was already
        seen (or in history-less mode). Atomic across processes/threads via
        INSERT OR IGNORE on the primary key.
        """
        if entry_id is None:
            raise ValueError("entry_id must be an integer")
        if self._conn is None:
            return False
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO seen (entry_id) VALUES (?)", (entry_id,)
        )
        self._conn.commit()
        return cur.rowcount == 1

    def is_seen(self, entry_id: int) -> bool:
        """True if the entry_id has been picked up before."""
        if self._conn is None:
            return False
        cur = self._conn.execute(
            "SELECT 1 FROM seen WHERE entry_id = ?", (entry_id,)
        )
        return cur.fetchone() is not None

    def unmark_seen(self, entry_id: int) -> bool:
        """Remove a pick-up marker so the article is presented again on the next run.

        Used to defer articles whose tagging failed (e.g. LLM unavailable): the
        article stays in the queue instead of being permanently lost. Returns
        True if a row was deleted; False if it was not seen (or in history-less
        mode).
        """
        if entry_id is None:
            raise ValueError("entry_id must be an integer")
        if self._conn is None:
            return False
        cur = self._conn.execute(
            "DELETE FROM seen WHERE entry_id = ?", (entry_id,)
        )
        self._conn.commit()
        return cur.rowcount == 1

    # -- decisions -------------------------------------------------------

    def record_decision(
        self, entry_id: int, tag: str, action: str, source: str
    ) -> None:
        """Record an accept/reject decision for an entry's tag.

        No-op in history-less mode. ``action`` is validated before the
        database is touched.
        """
        if action not in _VALID_ACTIONS:
            raise ValueError(
                "invalid action %r; must be one of: %s"
                % (action, ", ".join(_VALID_ACTIONS))
            )
        if tag is None:
            raise ValueError("tag must be a string")
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO decisions (entry_id, tag, action, source)"
            " VALUES (?, ?, ?, ?)",
            (entry_id, tag, action, source),
        )
        self._conn.commit()

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Close the connection if open. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
