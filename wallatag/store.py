"""Optional SQLite decision log.

The decision log provides seen-dedupe (an article is CLAIMED when it is PICKED
UP, so concurrent runs: Prefect + manual: never double-process it) and
records accept/reject tag decisions (for later ranking and phase-2 AI
training). A claim is a COOLDOWN, not a permanent exclusion: once its
``picked_at`` timestamp is older than ``SEEN_TTL_DAYS`` the article becomes
eligible again, so one that was skipped, rejected, or interrupted is retried
eventually instead of being lost forever (bug 8a7d089). It is strictly
optional: ``path=None`` selects history-less mode in which no database is ever
opened and every method is a safe no-op.

Concurrency: a claim is an atomic UPSERT on the seen primary key, which is
safe across processes/threads as long as each process or thread uses its own
``Store`` instance (one connection per process/thread: sharing a single
instance across threads raises ``sqlite3.ProgrammingError`` because of
``check_same_thread``); ``busy_timeout`` serializes write contention. The
natural fallback dedupe is wallabag itself: tagged articles stop matching the
untagged filter.
"""

from __future__ import annotations

import os
import sqlite3
import time

_VALID_ACTIONS = ("accept", "reject")
# Default reconsider cooldown, in days, for a pick-up claim: how long it
# excludes an article from the untagged feed. This is a COOLDOWN, not a
# permanent exclusion: a skipped, rejected, or interrupted article reappears
# after this many days (bug 8a7d089). Successful tagging is normally final
# anyway because wallabag stops offering a tagged article to the untagged
# filter; the cooldown only matters for otherwise-stuck articles. Overridable
# per store via `[store] reconsider_after_days`.
SEEN_TTL_DAYS = 7
# The WAL-mode transition needs an exclusive lock; on a fresh, not-yet-existing
# database two concurrent constructors can collide with "database is locked".
# Bound the retry so startup can never hang.
_SETUP_RETRY_ATTEMPTS = 10
_SETUP_RETRY_DELAY = 0.05  # seconds


class Store:
    """Persistent SQLite decision log.

    ``path=None`` selects history-less mode: a valid no-op store that does not
    touch the filesystem.

    ``reconsider_after_days`` is the pick-up cooldown in days (see
    ``SEEN_TTL_DAYS``): a claim younger than this excludes the article, an
    older one is refreshable. ``0`` disables the cooldown, so every claim
    succeeds (the article is always reconsidered).
    """

    def __init__(
        self,
        path: str | None,
        reconsider_after_days: int = SEEN_TTL_DAYS,
    ) -> None:
        self.path = path
        self.reconsider_after_days = reconsider_after_days
        # Precomputed SQLite datetime modifier, e.g. "-7 days".
        self._ttl_modifier = f"-{reconsider_after_days} days"
        if path is None:
            self._conn = None
            return
        # Create the store's parent directory on first use so a fresh
        # `[store] path` under a not-yet-existing directory works instead of
        # failing with "unable to open database file". Mode 0700 because the
        # decision log may sit beside credential files; an existing directory
        # is left untouched. A failure here (e.g. the parent path is a file)
        # is surfaced as an OperationalError so callers handle it exactly like
        # any other unopenable store.
        parent = os.path.dirname(path)
        if parent:
            try:
                os.makedirs(parent, mode=0o700, exist_ok=True)
            except OSError as exc:
                raise sqlite3.OperationalError(
                    f"cannot create store directory {parent!r}: {exc}"
                ) from exc
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
        """Unconditionally mark an article picked up (TTL-ignoring).

        Returns True if the row was newly inserted, False if it was already
        present (or in history-less mode). Atomic across processes/threads via
        INSERT OR IGNORE on the primary key. This is the low-level marker used
        to pre-seed state (tests, backfills); the tagging drivers use
        :meth:`claim`, which is TTL-aware so stale claims can be refreshed.
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

    def claim(self, entry_id: int) -> bool:
        """Atomically claim an article for this run; True if it may proceed.

        Returns True when there was no claim, or when the existing claim is
        STALE (older than ``reconsider_after_days``) and is refreshed; False
        while a FRESH claim exists, i.e. another run is (or recently was) on
        the same article. This makes the dedupe a cooldown rather than a
        permanent exclusion, while still letting concurrent runs race safely
        for the same article: the UPSERT is one atomic statement on the
        primary key, so exactly one racer gets True.

        In history-less mode there is no dedupe, so this always returns True
        (the article may always be processed).
        """
        if entry_id is None:
            raise ValueError("entry_id must be an integer")
        if self._conn is None:
            return True
        cur = self._conn.execute(
            "INSERT INTO seen (entry_id) VALUES (?) "
            "ON CONFLICT(entry_id) DO UPDATE SET picked_at = datetime('now') "
            "WHERE seen.picked_at <= datetime('now', ?)",
            (entry_id, self._ttl_modifier),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def is_seen(self, entry_id: int) -> bool:
        """True if the entry_id has a FRESH pick-up claim (within the TTL).

        A stale claim (``picked_at`` older than ``reconsider_after_days``) is
        treated as not seen, so the article is offered again.
        """
        if self._conn is None:
            return False
        cur = self._conn.execute(
            "SELECT 1 FROM seen WHERE entry_id = ? "
            "AND picked_at > datetime('now', ?)",
            (entry_id, self._ttl_modifier),
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
