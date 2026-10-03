"""Tests for wallatag.store: the optional SQLite decision log.

Normal mode uses a fresh tempfile DB per test; history-less mode (Store(None))
must never create a file. Concurrency is exercised with two multiprocessing
writers racing to mark_seen/claim the same entry through a barrier.
"""

import contextlib
import multiprocessing
import os
import sqlite3
import stat
import tempfile
import time
import unittest
from unittest.mock import patch

from tests._tags import IntegrationTest

from wallatag.store import Store


def _concurrent_writer(db_path, barrier, entry_id, results):
    """Worker: open a fresh Store connection and race to mark_seen."""
    store = Store(db_path)
    try:
        barrier.wait(timeout=30)
    except Exception:
        results.put(None)
        store.close()
        return
    results.put(store.mark_seen(entry_id))
    store.close()


def _concurrent_claimer(
    db_path, barrier, entry_id, results, reconsider_after_days=7
):
    """Worker: open a fresh Store connection and race to claim an entry."""
    store = Store(db_path, reconsider_after_days=reconsider_after_days)
    try:
        barrier.wait(timeout=30)
    except Exception:
        results.put(None)
        store.close()
        return
    results.put(store.claim(entry_id))
    store.close()


def _expire(db_path, entry_id, days=8):
    """Backdate a completed claim past its cooldown."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE seen SET picked_at = datetime('now', ?), "
            "status = 'cooldown', lease_until = NULL, claim_token = NULL"
            " WHERE entry_id = ?",
            (f"-{days} days", entry_id),
        )
        conn.commit()


def _expire_lease(db_path, entry_id, seconds=600):
    """Expire an in-progress lease without changing its cooldown timestamp."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE seen SET lease_until = datetime('now', ?) "
            "WHERE entry_id = ?",
            (f"-{seconds} seconds", entry_id),
        )
        conn.commit()


def _concurrent_constructor(db_path, barrier, entry_id, results):
    """Worker: synchronize BEFORE Store() so construction itself races.

    The barrier sits before the constructor, making two fresh, concurrent
    Store() calls collide on the WAL-mode transition of a brand-new database.
    Any exception is reported back to the parent.
    """
    try:
        barrier.wait(timeout=30)
        store = Store(db_path)  # constructor races here
        store.mark_seen(entry_id)
        store.close()
        results.put(None)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the process
        results.put(repr(exc))


def db_path(tmp: str) -> str:
    return os.path.join(tmp, "store.db")


class MarkSeenTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Store(db_path(self._tmp.name))

    def tearDown(self):
        self.store.close()

    def test_mark_seen_true_then_false(self):
        self.assertTrue(self.store.mark_seen(1))
        self.assertFalse(self.store.mark_seen(1))

    def test_is_seen_after_marking(self):
        self.assertFalse(self.store.is_seen(7))
        self.store.mark_seen(7)
        self.assertTrue(self.store.is_seen(7))

    def test_is_seen_false_for_unmarked(self):
        self.store.mark_seen(1)
        self.assertFalse(self.store.is_seen(999))

    def test_mark_seen_none_raises_valueerror(self):
        # None must never be silently auto-assigned a rowid by SQLite.
        with self.assertRaises(ValueError):
            self.store.mark_seen(None)
        with contextlib.closing(
            sqlite3.connect(db_path(self._tmp.name))
        ) as conn:
            count = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
        self.assertEqual(count, 0)

    def test_separate_store_instances_no_double_mark(self):
        # Models "run twice -> no double-tagging" at the store level.
        first = Store(db_path(self._tmp.name))
        second = Store(db_path(self._tmp.name))
        try:
            self.assertTrue(first.mark_seen(42))
            self.assertFalse(second.mark_seen(42))
        finally:
            first.close()
            second.close()
        with contextlib.closing(
            sqlite3.connect(db_path(self._tmp.name))
        ) as conn:
            count = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
        self.assertEqual(count, 1)


class UnmarkSeenTest(unittest.TestCase):
    """Deferral: remove a pick-up marker so the article is presented again."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Store(db_path(self._tmp.name))

    def tearDown(self):
        self.store.close()

    def test_unmark_seen_after_mark_removes_row(self):
        self.assertTrue(self.store.mark_seen(1))
        self.assertTrue(self.store.is_seen(1))
        self.assertTrue(self.store.unmark_seen(1))
        self.assertFalse(self.store.is_seen(1))

    def test_unmark_seen_when_not_seen_returns_false(self):
        self.assertFalse(self.store.unmark_seen(1))

    def test_unmark_seen_history_less_returns_false(self):
        store = Store(None)
        self.assertFalse(store.unmark_seen(1))
        store.close()

    def test_unmark_seen_none_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.store.unmark_seen(None)

    def test_unmark_then_mark_again_allows_repickup(self):
        # The deferred article can be marked again on the next run.
        self.store.mark_seen(7)
        self.store.unmark_seen(7)
        self.assertTrue(self.store.mark_seen(7))


class ClaimTtlTest(unittest.TestCase):
    """Completed claims are a TTL cooldown, not a permanent exclusion.

    An article that was completed but never tagged (skipped or rejected
    wholesale) must reappear once ``SEEN_TTL_DAYS`` have passed. Active claims
    have separate lease-expiry tests below.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = db_path(self._tmp.name)
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()

    def test_claim_true_then_false_while_fresh(self):
        # The first pick-up wins; a concurrent pick-up within the window loses.
        self.assertTrue(self.store.claim(1))
        self.assertFalse(self.store.claim(1))

    def test_is_seen_false_once_claim_is_stale(self):
        self.store.claim(1)
        self.assertTrue(self.store.is_seen(1))
        _expire(self.path, 1)
        self.assertFalse(self.store.is_seen(1))

    def test_claim_refreshes_a_stale_claim(self):
        # After the cooldown the entry is claimable again and the timestamp is
        # refreshed, so a second claim within the new window is refused.
        self.store.claim(1)
        _expire(self.path, 1)
        self.assertTrue(self.store.claim(1))
        self.assertTrue(self.store.is_seen(1))
        self.assertFalse(self.store.claim(1))

    def test_claim_history_less_always_true(self):
        # No database -> no dedupe -> every pick-up is granted.
        store = Store(None)
        try:
            self.assertTrue(store.claim(1))
            self.assertTrue(store.claim(1))
        finally:
            store.close()

    def test_claim_none_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.store.claim(None)


class ReconsiderAfterDaysTest(unittest.TestCase):
    """The pick-up cooldown is configurable per store (bug bffbc29)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = db_path(self._tmp.name)

    def test_default_cooldown_is_seven_days(self):
        store = Store(self.path)
        try:
            self.assertEqual(store.reconsider_after_days, 7)
        finally:
            store.close()

    def test_short_cooldown_expires_sooner(self):
        # With a 1-day cooldown, a claim backdated 2 days is already stale,
        # while the default 7-day cooldown would still hold it.
        store = Store(self.path, reconsider_after_days=1)
        try:
            self.assertTrue(store.claim(1))
            self.assertFalse(store.claim(1))
            _expire(self.path, 1, days=2)
            self.assertFalse(store.is_seen(1))
            self.assertTrue(store.claim(1))
        finally:
            store.close()

    def test_long_cooldown_holds_a_claim_the_default_would_release(self):
        # A 30-day cooldown keeps a claim backdated 8 days (past the default)
        # fresh, so the article stays excluded.
        store = Store(self.path, reconsider_after_days=30)
        try:
            self.assertTrue(store.claim(1))
            _expire(self.path, 1, days=8)
            self.assertTrue(store.is_seen(1))
            self.assertFalse(store.claim(1))
        finally:
            store.close()

    def test_zero_cooldown_keeps_active_lock_but_reclaims_on_completion(self):
        # 0 disables only the post-attempt cooldown, not the active lease.
        store = Store(self.path, reconsider_after_days=0)
        try:
            self.assertTrue(store.claim(1))
            self.assertFalse(store.claim(1))
            self.assertTrue(store.complete_claim(1))
            self.assertFalse(store.is_seen(1))
            self.assertTrue(store.claim(1))
        finally:
            store.close()

    def test_expired_lease_is_reclaimable_before_long_cooldown(self):
        store = Store(self.path, reconsider_after_days=30)
        second = Store(self.path, reconsider_after_days=30)
        try:
            self.assertTrue(store.claim(1))
            _expire_lease(self.path, 1)
            self.assertFalse(store.is_seen(1))
            self.assertTrue(second.claim(1))
            # A stale owner cannot release or complete the replacement claim.
            self.assertFalse(store.release_claim(1))
            self.assertFalse(store.complete_claim(1))
            self.assertTrue(second.is_seen(1))
            self.assertTrue(second.complete_claim(1))
            self.assertTrue(second.is_seen(1))
        finally:
            store.close()
            second.close()

    def test_release_makes_an_unfinished_claim_immediately_available(self):
        store = Store(self.path)
        second = Store(self.path)
        try:
            self.assertTrue(store.claim(1))
            self.assertTrue(store.release_claim(1))
            self.assertFalse(store.is_seen(1))
            self.assertTrue(second.claim(1))
        finally:
            store.close()
            second.close()

    def test_heartbeat_renews_an_active_lease(self):
        store = Store(self.path)
        try:
            self.assertTrue(store.claim(1))
            with contextlib.closing(sqlite3.connect(self.path)) as conn:
                conn.execute(
                    "UPDATE seen SET lease_until = datetime('now', '+10 seconds') "
                    "WHERE entry_id = 1"
                )
                conn.commit()
            self.assertTrue(store.heartbeat_claim(1))
            with contextlib.closing(sqlite3.connect(self.path)) as conn:
                future = conn.execute(
                    "SELECT lease_until > datetime('now', '+240 seconds') "
                    "FROM seen WHERE entry_id = 1"
                ).fetchone()[0]
            self.assertEqual(future, 1)
        finally:
            store.close()

    def test_background_heartbeat_keeps_a_claim_active(self):
        store = Store(self.path)
        try:
            self.assertTrue(store.claim(1))
            with (
                patch("wallatag.store.CLAIM_HEARTBEAT_SECONDS", 0.01),
                patch("wallatag.store.CLAIM_LEASE_SECONDS", 2),
                store.keep_claim_alive(1),
            ):
                time.sleep(0.05)
                second = Store(self.path)
                try:
                    self.assertFalse(second.claim(1))
                finally:
                    second.close()
        finally:
            store.close()

    def test_legacy_seen_table_migrates_as_cooldown(self):
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "CREATE TABLE seen (entry_id INTEGER PRIMARY KEY, "
                "picked_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            conn.execute("INSERT INTO seen (entry_id) VALUES (1)")
            conn.commit()
        store = Store(self.path)
        try:
            self.assertTrue(store.is_seen(1))
            self.assertFalse(store.claim(1))
        finally:
            store.close()


class DecisionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Store(db_path(self._tmp.name))

    def tearDown(self):
        self.store.close()

    def test_record_decision_inserts_row(self):
        self.store.record_decision(1, "python", "accept", "rules")
        self.store.record_decision(1, "cooking", "reject", "vocabulary")

        with contextlib.closing(
            sqlite3.connect(db_path(self._tmp.name))
        ) as conn:
            rows = conn.execute(
                "SELECT entry_id, tag, action, source FROM decisions"
                " ORDER BY rowid"
            ).fetchall()
        self.assertEqual(
            rows,
            [
                (1, "python", "accept", "rules"),
                (1, "cooking", "reject", "vocabulary"),
            ],
        )

    def test_record_decision_invalid_action_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.store.record_decision(1, "python", "nonsense", "rules")
        # Nothing was written by the rejected call.
        with contextlib.closing(
            sqlite3.connect(db_path(self._tmp.name))
        ) as conn:
            count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[
                0
            ]
        self.assertEqual(count, 0)

    def test_record_decision_none_tag_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.store.record_decision(1, None, "accept", "rules")

    def test_db_level_check_rejects_invalid_action(self):
        # The schema CHECK constraint enforces the action whitelist too.
        with contextlib.closing(
            sqlite3.connect(db_path(self._tmp.name))
        ) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO decisions (entry_id, tag, action, source)"
                    " VALUES (1, 'a', 'bogus', 'rules')"
                )


class HistoryLessTest(unittest.TestCase):
    def test_history_less_is_noop_and_creates_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(None)
            self.assertFalse(store.mark_seen(1))
            self.assertFalse(store.is_seen(1))
            store.record_decision(1, "a", "accept", "rules")  # must not raise
            store.close()  # must be safe
            store.close()  # idempotent
            self.assertEqual(os.listdir(tmp), [])

    def test_history_less_rejects_invalid_action(self):
        # Validation still applies even with no database.
        store = Store(None)
        with self.assertRaises(ValueError):
            store.record_decision(1, "a", "nonsense", "rules")
        store.close()


class ContextManagerTest(unittest.TestCase):
    def test_context_manager_marks_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Store(db_path(tmp)) as store:
                self.assertTrue(store.mark_seen(1))
            # After exit the connection is closed; the store degrades to the
            # safe no-op path (returns False) rather than raising.
            self.assertFalse(store.mark_seen(2))
            self.assertFalse(store.is_seen(1))


class ParentDirCreationTest(unittest.TestCase):
    """A fresh store path under a missing directory is created on first use."""

    def test_creates_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "data", "wallatag.db")
            with Store(path) as store:
                self.assertTrue(store.mark_seen(1))
                self.assertTrue(store.is_seen(1))
            self.assertTrue(os.path.exists(path))

    def test_created_directory_is_private(self):
        # The decision log may sit beside credential files; mode 0700.
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "data")
            path = os.path.join(parent, "wallatag.db")
            with Store(path):
                pass
            mode = stat.S_IMODE(os.stat(parent).st_mode)
            self.assertEqual(mode, 0o700)

    def test_existing_directory_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "existing")
            os.makedirs(parent)
            marker = os.path.join(parent, "keep.txt")
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("keep")
            with Store(os.path.join(parent, "wallatag.db")):
                pass
            self.assertTrue(os.path.exists(marker))

    def test_bare_filename_has_no_parent(self):
        # A path with no directory component must not attempt makedirs("").
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with Store("wallatag.db") as store:
                    self.assertTrue(store.mark_seen(1))
            finally:
                os.chdir(cwd)
            self.assertTrue(os.path.exists(os.path.join(tmp, "wallatag.db")))


class ConcurrencyTest(IntegrationTest):
    def test_two_writers_leave_single_seen_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = db_path(tmp)
            barrier = multiprocessing.Barrier(2)
            results = multiprocessing.Queue()
            procs = [
                multiprocessing.Process(
                    target=_concurrent_writer,
                    args=(path, barrier, 42, results),
                )
                for _ in range(2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=30)
            for proc in procs:
                self.assertFalse(
                    proc.is_alive(), "writer process did not exit"
                )

            outcomes = [results.get(timeout=5) for _ in procs]
            self.assertEqual(sorted(outcomes), [False, True])

            with contextlib.closing(sqlite3.connect(path)) as conn:
                count = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
            self.assertEqual(count, 1)


class ClaimConcurrencyTest(IntegrationTest):
    def _assert_one_claimer_wins(self, reconsider_after_days):
        with tempfile.TemporaryDirectory() as tmp:
            path = db_path(tmp)
            barrier = multiprocessing.Barrier(2)
            results = multiprocessing.Queue()
            procs = [
                multiprocessing.Process(
                    target=_concurrent_claimer,
                    args=(path, barrier, 42, results, reconsider_after_days),
                )
                for _ in range(2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=30)
            for proc in procs:
                self.assertFalse(
                    proc.is_alive(), "claimer process did not exit"
                )

            outcomes = [results.get(timeout=5) for _ in procs]
            self.assertEqual(sorted(outcomes), [False, True])

            with contextlib.closing(sqlite3.connect(path)) as conn:
                count = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
            self.assertEqual(count, 1)

    def test_two_claimers_one_wins(self):
        self._assert_one_claimer_wins(reconsider_after_days=7)

    def test_two_claimers_one_wins_with_zero_cooldown(self):
        self._assert_one_claimer_wins(reconsider_after_days=0)


class ConstructorRaceTest(IntegrationTest):
    def test_concurrent_constructors_on_fresh_path(self):
        # Two processes each construct Store() on the SAME fresh (non-existent)
        # path, synchronized BEFORE the constructor so the WAL-mode transition
        # collides. Neither may raise OperationalError; both writes land.
        with tempfile.TemporaryDirectory() as tmp:
            path = db_path(tmp)
            barrier = multiprocessing.Barrier(2)
            results = multiprocessing.Queue()
            procs = [
                multiprocessing.Process(
                    target=_concurrent_constructor,
                    args=(path, barrier, entry_id, results),
                )
                for entry_id in (1, 2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=60)
            for proc in procs:
                self.assertFalse(
                    proc.is_alive(), "constructor process did not exit"
                )

            errors = []
            for _ in procs:
                outcome = results.get(timeout=5)
                if outcome is not None:
                    errors.append(outcome)
            self.assertEqual(
                errors, [], "concurrent Store() raised: %r" % (errors,)
            )

            with contextlib.closing(sqlite3.connect(path)) as conn:
                rows = conn.execute(
                    "SELECT entry_id FROM seen ORDER BY entry_id"
                ).fetchall()
            self.assertEqual(rows, [(1,), (2,)])

    def test_concurrent_migration_of_legacy_seen_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = db_path(tmp)
            with contextlib.closing(sqlite3.connect(path)) as conn:
                conn.execute(
                    "CREATE TABLE seen (entry_id INTEGER PRIMARY KEY, "
                    "picked_at TEXT NOT NULL DEFAULT (datetime('now')))"
                )
                conn.execute("INSERT INTO seen (entry_id) VALUES (1)")
                conn.commit()

            barrier = multiprocessing.Barrier(2)
            results = multiprocessing.Queue()
            procs = [
                multiprocessing.Process(
                    target=_concurrent_constructor,
                    args=(path, barrier, entry_id, results),
                )
                for entry_id in (2, 3)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=60)
            for proc in procs:
                self.assertFalse(
                    proc.is_alive(), "migration process did not exit"
                )

            errors = [results.get(timeout=5) for _ in procs]
            self.assertEqual(errors, [None, None])
            with contextlib.closing(sqlite3.connect(path)) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(seen)")
                }
                rows = conn.execute(
                    "SELECT entry_id, status FROM seen ORDER BY entry_id"
                ).fetchall()
            self.assertTrue(
                {"status", "lease_until", "claim_token"}.issubset(columns)
            )
            self.assertEqual(
                rows,
                [(1, "cooldown"), (2, "cooldown"), (3, "cooldown")],
            )


if __name__ == "__main__":
    unittest.main()
