"""Optional SQLite decision log.

The decision log provides seen-dedupe and records accept/reject tag decisions
(for later ranking and phase-2 AI training). An active processing lease prevents
two runs from working the same article concurrently; on completion, a separate
reconsider cooldown delays another attempt. If a process disappears, its lease
expires after ``CLAIM_LEASE_SECONDS`` so the article is recoverable without
waiting through the cooldown (bug 8a7d089). It is strictly optional:
``path=None`` selects history-less mode in which no database is ever opened
and every method is a safe no-op.

Concurrency: acquiring and settling a claim are atomic operations on the seen
primary key. Each claim has an ownership token so an expired worker cannot
settle a replacement worker's lease. Active leases are renewed by a background
heartbeat; ``busy_timeout`` serializes SQLite write contention. The natural
fallback dedupe is wallabag itself: tagged articles stop matching the untagged
filter.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

_VALID_ACTIONS = ("accept", "reject")
# Default reconsider cooldown, in days, after a completed attempt. It applies
# to skipped, rejected, and otherwise-untagged articles; active leases are
# separate and expire independently (bug 8a7d089). Overridable per store via
# `[store] reconsider_after_days`.
SEEN_TTL_DAYS = 7
# An abandoned in-progress claim is recoverable after this lease expires.
CLAIM_LEASE_SECONDS = 300
CLAIM_HEARTBEAT_SECONDS = 60
# The WAL-mode transition needs an exclusive lock; on a fresh, not-yet-existing
# database two concurrent constructors can collide with "database is locked".
# Bound the retry so startup can never hang.
_SETUP_RETRY_ATTEMPTS = 10
_SETUP_RETRY_DELAY = 0.05  # seconds


class Store:
    """Persistent SQLite decision log.

    ``path=None`` selects history-less mode: a valid no-op store that does not
    touch the filesystem.

    ``reconsider_after_days`` is the post-attempt cooldown in days (see
    ``SEEN_TTL_DAYS``). ``0`` disables only that cooldown; the active lease
    still prevents concurrent processing.
    """

    def __init__(
        self,
        path: str | None,
        reconsider_after_days: int = SEEN_TTL_DAYS,
    ) -> None:
        self.path = path
        self.reconsider_after_days = reconsider_after_days
        self._claim_tokens: dict[int, str] = {}
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
                        picked_at TEXT NOT NULL DEFAULT (datetime('now')),
                        status TEXT NOT NULL DEFAULT 'cooldown',
                        lease_until TEXT,
                        claim_token TEXT
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
                self._migrate_seen_table()
                self._conn.commit()
                break
            except sqlite3.OperationalError:
                if attempt == _SETUP_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_SETUP_RETRY_DELAY)

    def _migrate_seen_table(self) -> None:
        """Add lease state to existing stores without changing old cooldowns."""
        assert self._conn is not None
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(seen)")
        }
        if "status" not in columns:
            self._conn.execute(
                "ALTER TABLE seen ADD COLUMN status TEXT NOT NULL DEFAULT 'cooldown'"
            )
        if "lease_until" not in columns:
            self._conn.execute("ALTER TABLE seen ADD COLUMN lease_until TEXT")
        if "claim_token" not in columns:
            self._conn.execute("ALTER TABLE seen ADD COLUMN claim_token TEXT")

    # -- dedupe ----------------------------------------------------------

    def mark_seen(self, entry_id: int) -> bool:
        """Insert a completed cooldown marker (TTL-ignoring).

        Returns True if the row was newly inserted, False if it was already
        present (or in history-less mode). Atomic across processes/threads via
        INSERT OR IGNORE on the primary key. This is the low-level marker used
        to pre-seed state (tests, backfills); tagging drivers use :meth:`claim`
        to acquire an in-progress lease.
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
        """Atomically acquire an in-progress lease; True if it may proceed.

        An active lease blocks other runs independently of the reconsider
        cooldown. A completed claim is reclaimable after that cooldown; an
        abandoned in-progress claim is reclaimable after its lease expires.
        The UPSERT is atomic across processes.

        In history-less mode there is no dedupe, so this always returns True
        (the article may always be processed).
        """
        if entry_id is None:
            raise ValueError("entry_id must be an integer")
        if self._conn is None:
            return True
        token = uuid.uuid4().hex
        lease_modifier = f"+{CLAIM_LEASE_SECONDS} seconds"
        cur = self._conn.execute(
            "INSERT INTO seen "
            "(entry_id, picked_at, status, lease_until, claim_token) "
            "VALUES (?, datetime('now'), 'in_progress', datetime('now', ?), ?) "
            "ON CONFLICT(entry_id) DO UPDATE SET "
            "picked_at = datetime('now'), status = 'in_progress', "
            "lease_until = datetime('now', ?), claim_token = ? "
            "WHERE (seen.status = 'cooldown' "
            "AND seen.picked_at <= datetime('now', ?)) "
            "OR (seen.status = 'in_progress' "
            "AND seen.lease_until <= datetime('now'))",
            (
                entry_id,
                lease_modifier,
                token,
                lease_modifier,
                token,
                self._ttl_modifier,
            ),
        )
        self._conn.commit()
        claimed = cur.rowcount == 1
        if claimed:
            self._claim_tokens[entry_id] = token
        return claimed

    def is_seen(self, entry_id: int) -> bool:
        """True if the article is in progress or in its reconsider cooldown.

        Expired leases and elapsed cooldowns are not considered seen.
        """
        if self._conn is None:
            return False
        cur = self._conn.execute(
            "SELECT 1 FROM seen WHERE entry_id = ? "
            "AND ((status = 'in_progress' "
            "AND lease_until > datetime('now')) "
            "OR (status = 'cooldown' "
            "AND picked_at > datetime('now', ?)))",
            (entry_id, self._ttl_modifier),
        )
        return cur.fetchone() is not None

    def complete_claim(self, entry_id: int) -> bool:
        """Finish this run's claim and start the reconsider cooldown."""
        token = self._claim_tokens.get(entry_id)
        if self._conn is None or token is None:
            return False
        cur = self._conn.execute(
            "UPDATE seen SET picked_at = datetime('now'), status = 'cooldown', "
            "lease_until = NULL, claim_token = NULL "
            "WHERE entry_id = ? AND status = 'in_progress' AND claim_token = ?",
            (entry_id, token),
        )
        self._conn.commit()
        if cur.rowcount == 1:
            self._claim_tokens.pop(entry_id, None)
            return True
        return False

    def release_claim(self, entry_id: int) -> bool:
        """Release this run's claim without starting a cooldown."""
        token = self._claim_tokens.get(entry_id)
        if self._conn is None or token is None:
            return False
        cur = self._conn.execute(
            "DELETE FROM seen WHERE entry_id = ? AND status = 'in_progress' "
            "AND claim_token = ?",
            (entry_id, token),
        )
        self._conn.commit()
        self._claim_tokens.pop(entry_id, None)
        return cur.rowcount == 1

    def heartbeat_claim(self, entry_id: int) -> bool:
        """Extend this Store instance's active lease, if it has not expired."""
        token = self._claim_tokens.get(entry_id)
        if self._conn is None or self.path is None or token is None:
            return False
        return self._renew_claim(self.path, entry_id, token)

    @staticmethod
    def _renew_claim(path: str, entry_id: int, token: str) -> bool:
        """Renew using a thread-local connection (SQLite connections aren't shared)."""
        lease_modifier = f"+{CLAIM_LEASE_SECONDS} seconds"
        conn = sqlite3.connect(path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            cur = conn.execute(
                "UPDATE seen SET lease_until = datetime('now', ?) "
                "WHERE entry_id = ? AND status = 'in_progress' "
                "AND claim_token = ? AND lease_until > datetime('now')",
                (lease_modifier, entry_id, token),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    @contextmanager
    def keep_claim_alive(self, entry_id: int) -> Iterator[None]:
        """Renew a claim in the background during long work or user prompts."""
        token = self._claim_tokens.get(entry_id)
        if self._conn is None or self.path is None or token is None:
            yield
            return
        path = self.path
        stop = threading.Event()

        def heartbeat_loop() -> None:
            while not stop.wait(CLAIM_HEARTBEAT_SECONDS):
                try:
                    if not self._renew_claim(path, entry_id, token):
                        return
                except sqlite3.Error:
                    # The bounded lease remains the fallback if renewal fails.
                    continue

        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=CLAIM_HEARTBEAT_SECONDS + 1)

    def unmark_seen(self, entry_id: int) -> bool:
        """Remove a marker so the article is presented again on the next run.

        For a claim owned by this instance, releases only that claim. Otherwise
        deletes a low-level marker. Used for retryable failures and deliberate
        interruption. Returns True if a row was deleted; False if it was not
        seen (or in history-less mode).
        """
        if entry_id is None:
            raise ValueError("entry_id must be an integer")
        if self._conn is None:
            return False
        if entry_id in self._claim_tokens:
            return self.release_claim(entry_id)
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
