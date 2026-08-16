"""Tests for wallatag.store: the optional SQLite decision log.

Normal mode uses a fresh tempfile DB per test; history-less mode (Store(None))
must never create a file. Concurrency is exercised with two multiprocessing
writers racing to mark_seen the same entry through a barrier.
"""

import multiprocessing
import os
import sqlite3
import tempfile
import unittest

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
        with sqlite3.connect(db_path(self._tmp.name)) as conn:
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
        with sqlite3.connect(db_path(self._tmp.name)) as conn:
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

        with sqlite3.connect(db_path(self._tmp.name)) as conn:
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
        with sqlite3.connect(db_path(self._tmp.name)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[
                0
            ]
        self.assertEqual(count, 0)

    def test_record_decision_none_tag_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.store.record_decision(1, None, "accept", "rules")

    def test_db_level_check_rejects_invalid_action(self):
        # The schema CHECK constraint enforces the action whitelist too.
        with sqlite3.connect(db_path(self._tmp.name)) as conn:
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


class ConcurrencyTest(unittest.TestCase):
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

            with sqlite3.connect(path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
            self.assertEqual(count, 1)


class ConstructorRaceTest(unittest.TestCase):
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

            with sqlite3.connect(path) as conn:
                rows = conn.execute(
                    "SELECT entry_id FROM seen ORDER BY entry_id"
                ).fetchall()
            self.assertEqual(rows, [(1,), (2,)])


if __name__ == "__main__":
    unittest.main()
