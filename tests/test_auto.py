"""Tests for the headless batch run (wallatag.auto + cli.cmd_run).

The wallabag API is faked with a small FakeClient; logging output is captured
via unittest's assertLogs for run_auto and via stdout capture for cmd_run
(which configures a root StreamHandler through logging.basicConfig).
"""

import argparse
import contextlib
import dataclasses
import io
import logging
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from wallatag.auto import AutoSummary, run_auto, summary_line
from wallatag.cli import cmd_run
from wallatag.config import Config, FocusGroup, StoreConfig, TaggerConfig, WallabagConfig
from wallatag.llm import LLMError
from wallatag.store import Store
from wallatag.tagger import KeywordTagger
from wallatag.wallabag import WallabagError, _should_fetch, _tag_labels


class RaisingTagger:
    """Tagger stub whose suggest() always raises LLMError (model down)."""

    def suggest(self, entry):
        raise LLMError("model unavailable", status=500)


def entry(eid, title, url="https://example.com/x", domain="example.com",
          content="", reading_time=5, tags=()):
    return {
        "id": eid,
        "title": title,
        "url": url,
        "domain_name": domain,
        "content": content,
        "reading_time": reading_time,
        "language": "en",
        "tags": list(tags),
    }


class FakeClient:
    def __init__(self, entries=(), tags=(), fail_first=0, feed_error=None,
                 feed_fail_after=None):
        self.entries = list(entries)
        self.tags = list(tags)
        self.fail_first = fail_first
        self.feed_error = feed_error
        self.feed_fail_after = feed_fail_after
        self.add_calls = []
        self.closed = False

    def iter_untagged(self, per_page=30, ignored_tags=()):
        if self.feed_error is not None:
            raise self.feed_error
        # Faithful to WallabagClient.iter_untagged: drop entries whose tags
        # are non-empty and not all in the ignore-any list.
        ignored = frozenset(t.casefold() for t in ignored_tags)
        entries = [
            dict(item)
            for item in self.entries
            if _should_fetch(_tag_labels(item.get("tags")), ignored)
        ]
        if self.feed_fail_after is not None:
            for i, item in enumerate(entries):
                if i >= self.feed_fail_after:
                    raise WallabagError("network died")
                yield item
            return
        yield from entries

    def get_tags(self):
        return [{"label": t, "slug": t, "nbEntries": 0} for t in self.tags]

    def add_tags(self, entry_id, tags):
        if self.fail_first > 0:
            self.fail_first -= 1
            raise WallabagError("boom")
        self.add_calls.append((entry_id, sorted(tags)))

    def close(self):
        self.closed = True


def make_args(**overrides):
    defaults = {
        "config": None,
        "max": None,
        "focus": None,
        "tag_policy": None,
        "no_history": False,
        "no_apply": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_tagger(existing_tags=(), groups=None, tag_policy="all", max_suggestions=10):
    return KeywordTagger(
        groups or {},
        max_suggestions=max_suggestions,
        tag_policy=tag_policy,
        existing_tags=list(existing_tags),
    )


def make_cfg(url="https://wallabag.example.com", max_articles=None, store_path=None):
    return dataclasses.replace(
        Config(),
        wallabag=WallabagConfig(
            url=url,
            client_id="cid",
            client_secret="secret",
            username="alice",
            password="wonderland",
        ),
        store=StoreConfig(path=store_path),
        max_articles=max_articles,
    )


def decision_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT entry_id, tag, action, source FROM decisions ORDER BY rowid"
        ).fetchall()


class AutoBase(unittest.TestCase):
    def setUp(self):
        # Clear root handlers so cmd_run's basicConfig re-binds to the current
        # sys.stdout and assertLogs is not duplicated to stale streams.
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    def run_auto(self, client, tagger=None, store=None, cfg=None, dry_run=False):
        tagger = tagger if tagger is not None else make_tagger()
        store = store if store is not None else Store(None)
        cfg = cfg if cfg is not None else Config()
        with self.assertLogs("wallatag.auto", level="INFO") as cm:
            summary = run_auto(client, tagger, store, cfg, dry_run=dry_run)
        return summary, cm.output

    def cmd_run(self, client=None, cfg=None, args=None):
        cfg = cfg if cfg is not None else make_cfg()
        args = args if args is not None else make_args()
        out = io.StringIO()
        err = io.StringIO()
        ctx = (
            patch("wallatag.cli.WallabagClient", return_value=client)
            if client is not None
            else contextlib.nullcontext()
        )
        with ctx, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cmd_run(cfg, args)
        return code, out.getvalue(), err.getvalue()


class IdempotencyTest(AutoBase):
    def test_double_run_no_double_tagging(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "pomodoro again")],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            tagger = make_tagger(existing_tags=["Pomodoro"])
            try:
                first, _ = self.run_auto(client, tagger=tagger, store=store)
                # Second run: everything is already seen, so nothing is
                # processed and no log records are emitted.
                second = run_auto(client, tagger, store, Config())
            finally:
                store.close()

        self.assertEqual((first.presented, first.tagged), (2, 2))
        self.assertEqual((second.presented, second.tagged), (0, 0))
        # One add_tags call per article across both runs: no double-tagging.
        self.assertEqual(len(client.add_calls), 2)
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])])


class NoHistoryTest(AutoBase):
    def test_history_less_works_and_reprocesses(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            first, _ = self.run_auto(
                client,
                tagger=make_tagger(existing_tags=["Pomodoro"]),
                store=Store(None),
            )
            second, _ = self.run_auto(
                client,
                tagger=make_tagger(existing_tags=["Pomodoro"]),
                store=Store(None),
            )
            # No database file is ever created.
            self.assertEqual(os.listdir(tmp), [])

        self.assertEqual(first.tagged, 1)
        # Documented trade-off: without history, is_seen is always False, so
        # the same article is re-presented (and re-tagged) next run.
        self.assertEqual(second.presented, 1)


class DryRunTest(AutoBase):
    def test_dry_run_is_side_effect_free(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    dry_run=True,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            with sqlite3.connect(db) as conn:
                seen = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
                decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            self.assertEqual((seen, decisions), (0, 0))
            self.assertTrue(summary.dry_run)
            self.assertEqual((summary.tagged, summary.tags_applied), (1, 1))
        self.assertTrue(any("would tag 1: Pomodoro" in m for m in messages))


class IgnoredTagsTest(AutoBase):
    """[tagger] ignore_tags flows through run_auto to the client filter."""

    def _cfg(self, ignore_tags=()):
        return dataclasses.replace(
            Config(), tagger=TaggerConfig(ignore_tags=ignore_tags)
        )

    def test_ignored_tag_article_is_presented_and_tagged(self):
        # ignore_tags=("fix",): an article tagged ["fix"] is still fetched
        # (every one of its tags is in the ignore-any list).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["fix"])],
            tags=["Pomodoro"],
        )
        summary, _ = self.run_auto(
            client,
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            cfg=self._cfg(ignore_tags=("fix",)),
        )
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertEqual((summary.presented, summary.tagged), (1, 1))

    def test_mixed_tags_article_not_presented(self):
        # A tag outside the ignore list disqualifies the article.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["fix", "something"])],
            tags=["Pomodoro"],
        )
        # Direct run_auto: nothing is presented, so no logs are emitted
        # (the assertLogs helper would fail on an empty log stream).
        summary = run_auto(
            client,
            make_tagger(existing_tags=["Pomodoro"]),
            Store(None),
            self._cfg(ignore_tags=("fix",)),
        )
        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 0)

    def test_tagged_article_not_presented_with_default_ignore(self):
        # Default empty ignore list: a ["fix"]-tagged article is NOT fetched,
        # matching the real client contract.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["fix"])],
            tags=["Pomodoro"],
        )
        summary = run_auto(
            client,
            make_tagger(existing_tags=["Pomodoro"]),
            Store(None),
            Config(),
        )
        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 0)


class TagPolicyTest(AutoBase):
    def _groups(self):
        return {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}

    def test_only_existing_drops_rule_tags(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")])
        summary, messages = self.run_auto(
            client,
            tagger=make_tagger(groups=self._groups(), tag_policy="only-existing"),
        )
        self.assertEqual(client.add_calls, [])
        self.assertEqual((summary.presented, summary.tagged, summary.skipped), (1, 0, 1))
        self.assertTrue(any("no suggestions: 1" in m for m in messages))

    def test_all_applies_rule_tags(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")])
        summary, _ = self.run_auto(
            client,
            tagger=make_tagger(groups=self._groups(), tag_policy="all"),
        )
        self.assertEqual(client.add_calls, [(1, ["productivity"])])
        self.assertEqual((summary.tagged, summary.tags_applied), (1, 1))


class AddTagsFailureTest(AutoBase):
    def test_failure_logged_and_run_continues(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro one"), entry(2, "pomodoro two")],
            tags=["Pomodoro"],
            fail_first=1,  # only the first add_tags fails
        )
        summary, messages = self.run_auto(
            client, tagger=make_tagger(existing_tags=["Pomodoro"])
        )
        self.assertEqual(summary.tagged, 1)
        self.assertTrue(any("tagging failed 1" in m for m in messages))
        self.assertTrue(any("tagged 2" in m for m in messages))
        # No decisions for the failed article; only the successful one logged.
        self.assertEqual(client.add_calls, [(2, ["Pomodoro"])])


class FeedErrorTest(AutoBase):
    def test_total_feed_failure_no_traceback(self):
        client = FakeClient(feed_error=WallabagError("cannot connect"))
        summary, messages = self.run_auto(client)
        self.assertEqual(summary.presented, 0)
        self.assertTrue(summary.feed_error)
        self.assertTrue(any("error fetching entries" in m for m in messages))

    def test_partial_feed_failure_stops_after_presented(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "pomodoro again"),
                     entry(3, "pomodoro third")],
            tags=["Pomodoro"],
            feed_fail_after=2,  # two entries delivered, then the feed dies
        )
        summary, messages = self.run_auto(
            client, tagger=make_tagger(existing_tags=["Pomodoro"])
        )
        self.assertEqual(summary.presented, 2)
        self.assertTrue(summary.feed_error)
        self.assertTrue(any("error fetching entries" in m for m in messages))


class DeterminismTest(AutoBase):
    def test_payload_sorted_regardless_of_suggestion_order(self):
        # Vocabulary order follows existing_tags order (beta, alpha), but the
        # applied payload must be sorted.
        client = FakeClient(entries=[entry(1, "alpha beta")], tags=["alpha", "beta"])
        summary, _ = self.run_auto(
            client, tagger=make_tagger(existing_tags=["beta", "alpha"])
        )
        self.assertEqual(client.add_calls, [(1, ["alpha", "beta"])])
        self.assertEqual(summary.tags_applied, 2)


class SkipTest(AutoBase):
    def test_no_suggestions_skips(self):
        client = FakeClient(entries=[entry(1, "unrelated soup")])
        summary, messages = self.run_auto(client)
        self.assertEqual(client.add_calls, [])
        self.assertEqual((summary.presented, summary.tagged, summary.skipped), (1, 0, 1))
        self.assertTrue(any("no suggestions: 1" in m for m in messages))


class LLMErrorTest(AutoBase):
    def test_llm_error_skips_entry_and_run_continues(self):
        # The tagger is a constructor arg to run_auto: inject a stub whose
        # suggest() raises LLMError for every entry.
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        summary, messages = self.run_auto(client, tagger=RaisingTagger())

        self.assertEqual((summary.presented, summary.tagged, summary.skipped), (2, 0, 2))
        self.assertEqual(client.add_calls, [])
        # An LLM failure is an article-level skip, NOT a feed error.
        self.assertFalse(summary.feed_error)
        self.assertTrue(any("LLM tagging failed 1" in m for m in messages))
        self.assertTrue(any("LLM tagging failed 2" in m for m in messages))


class DecisionsTest(AutoBase):
    def test_accept_rows_written_with_sources(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro guide")],
            tags=["Pomodoro"],
        )
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client,
                    tagger=make_tagger(
                        existing_tags=["Pomodoro"], groups=groups, tag_policy="all"
                    ),
                    store=store,
                )
            finally:
                store.close()

            self.assertEqual(summary.tags_applied, 2)
            self.assertEqual(
                decision_rows(db),
                [
                    (1, "Pomodoro", "accept", "vocabulary"),
                    (1, "productivity", "accept", "rules"),
                ],
            )


class SummaryWordingTest(AutoBase):
    def test_summary_line_variants(self):
        summary = AutoSummary(presented=3, tagged=2, tags_applied=3, skipped=1)
        self.assertEqual(
            summary_line(summary),
            "run: tagged 2 articles (3 tags applied), skipped 1",
        )
        self.assertEqual(
            summary_line(summary, dry_run=True),
            "dry run: would tag 2 articles (3 tags), skipped 1",
        )
        summary.feed_error = True
        self.assertEqual(
            summary_line(summary),
            "run: tagged 2 articles (3 tags applied), skipped 1, feed error",
        )


class CmdRunExitCodesTest(AutoBase):
    def test_success_exit_0(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        code, out, _ = self.cmd_run(client=client)
        self.assertEqual(code, 0)
        self.assertTrue(client.closed)
        self.assertIn("run: tagged 1 articles (1 tags applied), skipped 0", out)

    def test_missing_url_exit_2(self):
        code, _, err = self.cmd_run(cfg=Config())
        self.assertEqual(code, 2)
        self.assertIn("url", err)

    def test_missing_credentials_exit_2(self):
        cfg = dataclasses.replace(
            Config(),
            wallabag=WallabagConfig(url="https://wallabag.example.com"),
        )
        code, _, err = self.cmd_run(cfg=cfg)
        self.assertEqual(code, 2)
        self.assertIn("username and password", err)

    def test_get_tags_failure_exit_2(self):
        client = FakeClient(entries=[entry(1, "x")])

        def failing_get_tags():
            raise WallabagError("bad creds")

        client.get_tags = failing_get_tags
        code, _, err = self.cmd_run(client=client)
        self.assertEqual(code, 2)
        self.assertIn("could not fetch existing tags", err)
        self.assertTrue(client.closed)

    def test_store_open_failure_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "no_such_dir", "store.db")
            client = FakeClient(entries=[entry(1, "x")])
            code, _, err = self.cmd_run(
                client=client, cfg=make_cfg(store_path=bad_path)
            )
        self.assertEqual(code, 2)
        self.assertIn("could not open store", err)
        self.assertTrue(client.closed)

    def test_total_feed_error_exit_2(self):
        client = FakeClient(feed_error=WallabagError("cannot connect"))
        code, out, _ = self.cmd_run(client=client)
        self.assertEqual(code, 2)
        self.assertIn("feed error", out)

    def test_partial_feed_error_exit_0(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "pomodoro again"),
                     entry(3, "pomodoro third")],
            tags=["Pomodoro"],
            feed_fail_after=2,
        )
        code, out, _ = self.cmd_run(client=client)
        self.assertEqual(code, 0)
        self.assertIn("feed error", out)

    def test_max_zero_exit_0(self):
        code, out, _ = self.cmd_run(
            cfg=dataclasses.replace(Config(), max_articles=0)
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

    def test_keyboard_interrupt_exit_130(self):
        client = FakeClient(feed_error=KeyboardInterrupt())
        code, _, err = self.cmd_run(client=client)
        self.assertEqual(code, 130)
        self.assertTrue(client.closed)
        self.assertIn("interrupted", err)


if __name__ == "__main__":
    unittest.main()
