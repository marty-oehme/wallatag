"""Tests for the manual interactive review loop (wallatag.manual).

The wallabag API is faked with a small FakeClient; input is scripted via
``patch("builtins.input", side_effect=[...])``. A real KeywordTagger and a real
(optional) Store are used so decisions/mark_seen are observable.
"""

import argparse
import contextlib
import dataclasses
import io
import os
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from wallatag.cli import cmd_manual
from wallatag.config import Config, StoreConfig, TaggerConfig, WallabagConfig
from wallatag.llm import LLMError
from wallatag.manual import run_manual, summary_line
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
    def __init__(self, entries=(), tags=(), fail_on_add=None):
        self.entries = list(entries)
        self.tags = list(tags)
        self.fail_on_add = fail_on_add
        self.add_calls = []
        self.closed = False

    def iter_untagged(self, per_page=30, ignored_tags=(), ignored_regex=()):
        # Faithful to WallabagClient.iter_untagged: drop entries whose tags
        # are non-empty and not all in the ignore-any list (literal or regex).
        ignored = frozenset(t.casefold() for t in ignored_tags)
        patterns = tuple(re.compile(p, re.IGNORECASE) for p in ignored_regex)
        for item in self.entries:
            if _should_fetch(_tag_labels(item.get("tags")), ignored, patterns):
                yield dict(item)

    def get_tags(self):
        return [{"label": t, "slug": t, "nbEntries": 0} for t in self.tags]

    def add_tags(self, entry_id, tags):
        if self.fail_on_add is not None:
            raise self.fail_on_add
        self.add_calls.append((entry_id, sorted(tags)))

    def close(self):
        self.closed = True


class FailingFeedClient:
    """Client whose iter_untagged dies mid-stream (network failure)."""

    def __init__(self):
        self.add_calls = []
        self.closed = False

    def iter_untagged(self, per_page=30, ignored_tags=(), ignored_regex=()):
        yield entry(1, "first")
        yield entry(2, "second")
        raise WallabagError("network died")

    def get_tags(self):
        return []

    def add_tags(self, entry_id, tags):
        self.add_calls.append((entry_id, sorted(tags)))

    def close(self):
        self.closed = True


class EagerFailingClient:
    """Client whose iter_untagged is a plain function failing on first call.

    A non-generator iter_untagged raises at CALL time, not at next()-time:
    this exercises the guard around the genexpr's eager outer iterable.
    """

    def __init__(self):
        self.closed = False

    def iter_untagged(self, per_page=30, ignored_tags=(), ignored_regex=()):
        raise WallabagError("cannot connect")

    def get_tags(self):
        return []

    def add_tags(self, entry_id, tags):
        raise AssertionError("add_tags must not be called")

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


def make_tagger(existing_tags=(), groups=None, tag_policy="all", max_applied_tags=10):
    return KeywordTagger(
        groups or {},
        max_applied_tags=max_applied_tags,
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
            "SELECT tag, action, source FROM decisions ORDER BY rowid"
        ).fetchall()


class ManualBase(unittest.TestCase):
    def run_manual(self, client, inputs, *, tagger=None, cfg=None, store=None,
                   dry_run=False, fallback_tagger=None):
        tagger = tagger if tagger is not None else make_tagger()
        cfg = cfg if cfg is not None else Config()
        out = io.StringIO()
        with patch("builtins.input", side_effect=list(inputs)), \
             contextlib.redirect_stdout(out):
            summary = run_manual(
                client, tagger, store, cfg, dry_run=dry_run,
                fallback_tagger=fallback_tagger,
            )
        return summary, out.getvalue()


class NextFlowTest(ManualBase):
    def test_next_on_untouched_suggestions_applies(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    [""],  # enter = next
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
            self.assertEqual(
                decision_rows(db), [("Pomodoro", "accept", "vocabulary")]
            )
            with sqlite3.connect(db) as conn:
                seen = conn.execute(
                    "SELECT COUNT(*) FROM seen WHERE entry_id = 1"
                ).fetchone()[0]
            self.assertEqual(seen, 1)
        self.assertEqual((summary.tagged, summary.accepted), (1, 1))


class IgnoredTagsTest(ManualBase):
    """[tagger] ignore_tags flows through run_manual to the client filter."""

    def _cfg(self, ignore_tags=(), ignore_tags_regex=()):
        return dataclasses.replace(
            Config(),
            tagger=TaggerConfig(
                ignore_tags=ignore_tags, ignore_tags_regex=ignore_tags_regex
            ),
        )

    def test_ignored_tag_article_is_presented_and_tagged(self):
        # ignore_tags=("fix",): an article tagged ["fix"] is still fetched
        # (every one of its tags is in the ignore-any list).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["fix"])],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    [""],  # enter = next
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    cfg=self._cfg(ignore_tags=("fix",)),
                )
            finally:
                store.close()

        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertEqual((summary.presented, summary.tagged), (1, 1))

    def test_mixed_tags_article_not_presented(self):
        # A tag outside the ignore list disqualifies the article.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["fix", "something"])],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    [""],
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    cfg=self._cfg(ignore_tags=("fix",)),
                )
            finally:
                store.close()

        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 0)

    def test_tagged_article_not_presented_with_default_ignore(self):
        # Default empty ignore list: a ["fix"]-tagged article is NOT fetched,
        # matching the real client contract.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["fix"])],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    [""],
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                )
            finally:
                store.close()

        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 0)

    def test_ignored_regex_tag_article_is_presented_and_tagged(self):
        # ignore_tags_regex=("todo|fix",): an article tagged ["todo"] is
        # still fetched (its only tag matches a pattern).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["todo"])],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    [""],  # enter = next
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    cfg=self._cfg(ignore_tags_regex=("todo|fix",)),
                )
            finally:
                store.close()

        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertEqual((summary.presented, summary.tagged), (1, 1))

    def test_mixed_tags_regex_article_not_presented(self):
        # A tag matching neither a literal entry nor a regex pattern
        # disqualifies the article.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["todo", "something"])],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    [""],
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    cfg=self._cfg(ignore_tags_regex=("todo|fix",)),
                )
            finally:
                store.close()

        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 0)


class DropFlowTest(ManualBase):
    def test_drop_numbers_redisplays_and_applies_remaining(self):
        # Two suggestions; drop index 1, keep index 2, then next.
        client = FakeClient(
            entries=[entry(1, "pomodoro productivity guide")],
            tags=["Pomodoro", "productivity"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, out = self.run_manual(
                    client,
                    ["d", "1", ""],
                    tagger=make_tagger(
                        existing_tags=["Pomodoro", "productivity"]
                    ),
                    store=store,
                )
            finally:
                store.close()

            # The article was re-displayed after the drop (loop stayed).
            self.assertGreaterEqual(out.count("pomodoro productivity guide"), 2)
            self.assertEqual(client.add_calls, [(1, ["productivity"])])
            self.assertEqual(
                decision_rows(db),
                [
                    ("productivity", "accept", "vocabulary"),
                    ("Pomodoro", "reject", "vocabulary"),
                ],
            )
        self.assertEqual((summary.tagged, summary.dropped), (1, 1))


class DropAllTest(ManualBase):
    def test_drop_all_commits_nothing_but_records_rejects(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    ["d", "", ""],  # drop all, then next
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(
                decision_rows(db), [("Pomodoro", "reject", "vocabulary")]
            )
        self.assertEqual((summary.tagged, summary.dropped), (0, 1))


class AddFlowTest(ManualBase):
    def test_add_custom_tags_redisplays_and_applies(self):
        client = FakeClient(entries=[entry(1, "soup recipe")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, out = self.run_manual(
                    client,
                    ["a", " cooking ,  dinner ", ""],
                    store=store,
                )
            finally:
                store.close()

            # The article was re-displayed after the add (loop stayed).
            self.assertGreaterEqual(out.count("soup recipe"), 2)
            self.assertEqual(client.add_calls, [(1, ["cooking", "dinner"])])
            self.assertEqual(
                decision_rows(db),
                [("cooking", "accept", "manual"), ("dinner", "accept", "manual")],
            )
        self.assertEqual(summary.added, 2)


class AddDropCustomTest(ManualBase):
    def test_added_then_dropped_custom_is_not_recorded(self):
        client = FakeClient(entries=[entry(1, "soup recipe")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    ["a", "cooking", "d", "1", ""],
                    store=store,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(decision_rows(db), [])
        self.assertEqual(summary.added, 0)


class SkipTest(ManualBase):
    def test_skip_applies_and_records_nothing(self):
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, out = self.run_manual(client, ["s", "q"], store=store)
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(decision_rows(db), [])
            self.assertEqual(summary.tagged, 0)
            # The second article was still presented (loop continued).
            self.assertIn("second", out)


class SubPromptQuitTest(ManualBase):
    def test_q_at_add_prompt_aborts_back_to_menu(self):
        client = FakeClient(entries=[entry(1, "soup")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(client, ["a", "q", "q"], store=store)
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(decision_rows(db), [])
        self.assertEqual(summary.added, 0)

    def test_q_at_drop_prompt_aborts_back_to_menu(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client,
                    ["d", "q", "q"],
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(decision_rows(db), [])
        self.assertEqual(summary.dropped, 0)


class AddDedupTest(ManualBase):
    def test_add_dedups_duplicate_tags(self):
        client = FakeClient(entries=[entry(1, "soup")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_manual(
                    client, ["a", "cooking, cooking, dinner", ""], store=store
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [(1, ["cooking", "dinner"])])
            self.assertEqual(
                decision_rows(db),
                [("cooking", "accept", "manual"), ("dinner", "accept", "manual")],
            )
        self.assertEqual(summary.added, 2)


class DryRunTest(ManualBase):
    def test_dry_run_makes_no_writes(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, out = self.run_manual(
                    client,
                    ["d", "1", "a", "cooking", ""],
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    dry_run=True,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(decision_rows(db), [])
            with sqlite3.connect(db) as conn:
                seen = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
            self.assertEqual(seen, 0)
            self.assertIn("(dry run) would apply: cooking", out)
            self.assertIn("(dry run) would reject: Pomodoro", out)
        self.assertEqual((summary.tagged, summary.dropped), (1, 1))


class ZeroTagsArticleTest(ManualBase):
    def test_zero_tag_article_next_commits_nothing(self):
        # First article has no matching tags (empty working list); next with an
        # empty list applies nothing; then skip the second article.
        client = FakeClient(
            entries=[entry(1, "unrelated soup"), entry(2, "pomodoro focus")],
            tags=["Pomodoro"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, out = self.run_manual(
                    client,
                    ["", "s"],
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [])
            self.assertEqual(decision_rows(db), [])
            self.assertIn("(no tags)", out)
            # Loop continued: the second article was presented.
            self.assertIn("pomodoro focus", out)


class SummaryWordingTest(ManualBase):
    def test_summary_reflects_new_counts(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        summary, _ = self.run_manual(
            client,
            ["a", "cooking", ""],
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            store=Store(None),
        )

        self.assertEqual(
            summary_line(summary),
            "tagged 1 articles, accepted 1 tags, dropped 0, added 1 custom",
        )
        self.assertEqual(
            summary_line(summary, dry_run=True),
            "dry run: would tag 1 articles, accept 1 tags, drop 0, add 1 custom",
        )


class QuitTest(ManualBase):
    def test_quit_stops_early_and_exits_0(self):
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        out = io.StringIO()
        cfg = make_cfg()
        with patch("wallatag.cli.WallabagClient", return_value=client), \
             patch("builtins.input", side_effect=["q"]), \
             contextlib.redirect_stdout(out):
            code = cmd_manual(cfg, make_args())
        self.assertEqual(code, 0)
        self.assertTrue(client.closed)
        # Only the first article was presented; the second never appears.
        self.assertNotIn("second", out.getvalue())


class SeenDedupeTest(ManualBase):
    def test_store_pre_seen_entry_not_presented(self):
        client = FakeClient(entries=[entry(1, "already done"), entry(2, "new one")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            store.mark_seen(1)  # pre-seen by a previous run
            try:
                summary, out = self.run_manual(client, ["q"], store=store)
            finally:
                store.close()

        self.assertEqual(summary.presented, 1)
        self.assertNotIn("already done", out)
        self.assertIn("new one", out)


class MaxLimitTest(ManualBase):
    def test_max_caps_articles_presented(self):
        client = FakeClient(
            entries=[entry(1, "one"), entry(2, "two"), entry(3, "three")]
        )
        summary, out = self.run_manual(
            client, ["", ""], cfg=make_cfg(max_articles=2), store=Store(None)
        )

        self.assertEqual(summary.presented, 2)
        self.assertNotIn("three", out)


class AddTagsErrorTest(ManualBase):
    def test_add_tags_error_continues_loop(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "another")],
            tags=["Pomodoro"],
            fail_on_add=WallabagError("boom"),
        )
        out = io.StringIO()
        with patch("builtins.input", side_effect=["", "q"]), \
             contextlib.redirect_stdout(out):
            summary = run_manual(
                client, make_tagger(existing_tags=["Pomodoro"]), Store(None), Config()
            )

        self.assertIn("error tagging entry 1", out.getvalue())
        self.assertIn("boom", out.getvalue())
        self.assertEqual(summary.tagged, 0)
        # The loop did not crash: nothing raised out of run_manual.


class KeyboardInterruptTest(ManualBase):
    def test_keyboard_interrupt_clean_exit_closes_client(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        out = io.StringIO()
        err = io.StringIO()
        with patch("wallatag.cli.WallabagClient", return_value=client), \
             patch("builtins.input", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            code = cmd_manual(make_cfg(), make_args())

        self.assertEqual(code, 130)
        self.assertTrue(client.closed)
        self.assertIn("interrupted", err.getvalue())


class NoUrlTest(ManualBase):
    def test_missing_url_exits_2_without_client(self):
        err = io.StringIO()
        with patch("wallatag.cli.WallabagClient") as client_cls, \
             contextlib.redirect_stderr(err):
            code = cmd_manual(Config(), make_args())

        self.assertEqual(code, 2)
        self.assertIn("url", err.getvalue())
        client_cls.assert_not_called()


class InvalidChoiceTest(ManualBase):
    def test_invalid_choice_reprompts(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        out = io.StringIO()
        with patch("builtins.input", side_effect=["x", "q"]), \
             contextlib.redirect_stdout(out):
            summary = run_manual(
                client, make_tagger(existing_tags=["Pomodoro"]), Store(None), Config()
            )

        self.assertIn("invalid choice", out.getvalue())
        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 1)


class FeedFetchErrorTest(ManualBase):
    def test_mid_run_feed_error_breaks_loop_cleanly(self):
        # iter_untagged raises on the third next(): no traceback, a message is
        # printed, and the run completes with the articles already presented.
        client = FailingFeedClient()
        out = io.StringIO()
        err = io.StringIO()
        with patch("builtins.input", side_effect=["", ""]), \
             contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            summary = run_manual(client, make_tagger(), Store(None), Config())

        self.assertEqual(summary.presented, 2)
        self.assertIn("error fetching entries", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


class LLMErrorTest(ManualBase):
    def test_llm_error_skips_entry_and_session_continues(self):
        # The tagger is a constructor arg to run_manual: inject a stub whose
        # suggest() raises LLMError for every entry. Both articles are skipped
        # without ever reaching the input loop.
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        out = io.StringIO()
        err = io.StringIO()
        with patch("builtins.input", side_effect=["q"]), \
             contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            summary = run_manual(client, RaisingTagger(), Store(None), Config())

        self.assertEqual((summary.presented, summary.tagged), (2, 0))
        self.assertEqual(client.add_calls, [])
        self.assertFalse(summary.feed_error)
        self.assertIn("LLM tagging failed 1", err.getvalue())
        self.assertIn("LLM tagging failed 2", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_llm_error_unmarks_article_from_store(self):
        # Deferral: after an LLM failure the article's seen marker is removed,
        # so the next session presents it again instead of losing it.
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            err = io.StringIO()
            with patch("builtins.input", side_effect=["q"]), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(err):
                summary = run_manual(client, RaisingTagger(), store, Config())

            self.assertFalse(store.is_seen(1))
            self.assertFalse(store.is_seen(2))
            self.assertEqual((summary.presented, summary.tagged), (2, 0))
            self.assertIn("LLM tagging failed 1", err.getvalue())
            store.close()

    def test_llm_error_dry_run_defers_nothing(self):
        # Dry-run counterpart of the deferral test: dry run never marks an
        # article (mark_seen is skipped), so an LLM failure must not unmark
        # anything either — is_seen stays exactly as it was (nothing marked,
        # nothing unmarked).
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            err = io.StringIO()
            try:
                with patch("builtins.input", side_effect=["q"]), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(err), \
                     patch.object(
                         store, "unmark_seen", wraps=store.unmark_seen
                     ) as unmark:
                    summary = run_manual(
                        client, RaisingTagger(), store, Config(), dry_run=True
                    )

                self.assertFalse(unmark.called)
                # Nothing was marked, so nothing was unmarked.
                self.assertFalse(store.is_seen(1))
                self.assertFalse(store.is_seen(2))
                self.assertEqual((summary.presented, summary.tagged), (2, 0))
                self.assertIn("LLM tagging failed 1", err.getvalue())
            finally:
                store.close()


class LLMFallbackTest(ManualBase):
    """[ai] fallback_on_fail in manual mode: the keyword tagger's suggestions
    enter the normal review loop; an empty fallback keeps the error path."""

    def test_fallback_suggestions_enter_review_loop(self):
        # (a) The LLM fails, the keyword fallback suggests: the suggestions
        # enter the review loop (not the error path) — "next" applies them, no
        # LLM-failure message is printed, and the article is NOT unmarked.
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            out = io.StringIO()
            err = io.StringIO()
            try:
                with patch("builtins.input", side_effect=[""]), \
                     contextlib.redirect_stdout(out), \
                     contextlib.redirect_stderr(err):
                    summary = run_manual(
                        client,
                        RaisingTagger(),
                        store,
                        Config(),
                        fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
                    )
                # Fallback success: the article stays seen (like any reviewed
                # article) and the vocabulary source lands in the decision log.
                self.assertTrue(store.is_seen(1))
                self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
                self.assertEqual(
                    decision_rows(db), [("Pomodoro", "accept", "vocabulary")]
                )
                self.assertEqual((summary.presented, summary.tagged), (1, 1))
                self.assertNotIn("LLM tagging failed", err.getvalue())
            finally:
                store.close()

    def test_fallback_empty_keeps_error_path_and_unmarks(self):
        # (b) The fallback yields nothing: today's error path — the message is
        # printed and the article is unmarked (deferred to the next session).
        client = FakeClient(entries=[entry(1, "first")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            err = io.StringIO()
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(err):
                    summary = run_manual(
                        client,
                        RaisingTagger(),
                        store,
                        Config(),
                        fallback_tagger=make_tagger(),  # no suggestions
                    )
                self.assertFalse(store.is_seen(1))
                self.assertIn("LLM tagging failed 1", err.getvalue())
            finally:
                store.close()
        self.assertEqual((summary.presented, summary.tagged), (1, 0))
        self.assertEqual(client.add_calls, [])

    def test_fallback_empty_dry_run_defers_nothing(self):
        # Dry-run counterpart: nothing is marked, so nothing is unmarked.
        client = FakeClient(entries=[entry(1, "first")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            err = io.StringIO()
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(err), \
                     patch.object(
                         store, "unmark_seen", wraps=store.unmark_seen
                     ) as unmark:
                    summary = run_manual(
                        client,
                        RaisingTagger(),
                        store,
                        Config(),
                        dry_run=True,
                        fallback_tagger=make_tagger(),
                    )
                self.assertFalse(unmark.called)
                self.assertFalse(store.is_seen(1))
                self.assertIn("LLM tagging failed 1", err.getvalue())
            finally:
                store.close()
        self.assertEqual((summary.presented, summary.tagged), (1, 0))

    def test_fallback_dry_run_enters_review_loop_and_applies_nothing(self):
        # Dry-run counterpart of test_fallback_suggestions_enter_review_loop:
        # the fallback's suggestions still enter the review loop (the article
        # is presented and the working list is offered), but _apply in dry-run
        # mode writes NOTHING: no mark_seen, no add_tags, no decisions.
        client = FakeClient(entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            out = io.StringIO()
            err = io.StringIO()
            try:
                with patch("builtins.input", side_effect=[""]), \
                     contextlib.redirect_stdout(out), \
                     contextlib.redirect_stderr(err):
                    summary = run_manual(
                        client,
                        RaisingTagger(),
                        store,
                        Config(),
                        dry_run=True,
                        fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
                    )
                # Review loop was entered with the fallback's suggestions.
                self.assertIn("--- entry 1 ---", out.getvalue())
                self.assertIn("Pomodoro", out.getvalue())
                self.assertIn("(dry run) would apply: Pomodoro", out.getvalue())
                self.assertNotIn("LLM tagging failed", err.getvalue())
                # _apply did nothing: no mark_seen, no add_tags, no decisions.
                self.assertFalse(store.is_seen(1))
                self.assertEqual(client.add_calls, [])
                self.assertEqual(decision_rows(db), [])
            finally:
                store.close()
        self.assertEqual((summary.presented, summary.tagged), (1, 1))


class EofExitTest(ManualBase):
    def test_eof_clean_exit_0(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")])
        out = io.StringIO()
        err = io.StringIO()
        with patch("wallatag.cli.WallabagClient", return_value=client), \
             patch("builtins.input", side_effect=EOFError), \
             contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            code = cmd_manual(make_cfg(), make_args())
        self.assertEqual(code, 0)
        self.assertTrue(client.closed)
        self.assertIn("no input", err.getvalue())


class MissingCredentialsTest(ManualBase):
    def test_missing_credentials_exit_2(self):
        cfg = dataclasses.replace(
            Config(),
            wallabag=WallabagConfig(url="https://wallabag.example.com"),
        )
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cmd_manual(cfg, make_args())
        self.assertEqual(code, 2)
        self.assertIn("username and password", err.getvalue())


class GetTagsErrorTest(ManualBase):
    def test_get_tags_failure_message_and_exit_2(self):
        client = FakeClient(entries=[entry(1, "x")])

        def failing_get_tags():
            raise WallabagError("bad creds")

        client.get_tags = failing_get_tags
        err = io.StringIO()
        with patch("wallatag.cli.WallabagClient", return_value=client), \
             contextlib.redirect_stderr(err):
            code = cmd_manual(make_cfg(), make_args())
        self.assertEqual(code, 2)
        self.assertIn("could not fetch existing tags", err.getvalue())
        self.assertTrue(client.closed)


class MaxZeroTest(ManualBase):
    def test_max_zero_exits_0_even_without_config(self):
        # max_articles == 0 short-circuits before the url check.
        cfg = dataclasses.replace(Config(), max_articles=0)
        code = cmd_manual(cfg, make_args())
        self.assertEqual(code, 0)


class EagerFeedErrorTest(ManualBase):
    def test_eager_feed_error_no_traceback_and_flag_set(self):
        # A non-generator iter_untagged raises at call time; the guard must
        # catch it, set feed_error and never dump a traceback.
        client = EagerFailingClient()
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            summary = run_manual(client, make_tagger(), Store(None), Config())

        self.assertEqual(summary.presented, 0)
        self.assertTrue(summary.feed_error)
        self.assertIn("error fetching entries", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


class FeedErrorExitCodeTest(ManualBase):
    def test_zero_presented_feed_error_exits_2(self):
        # Total feed failure (nothing presented) must not look like success to
        # a scheduler: exit 2 while still printing the summary.
        client = EagerFailingClient()
        out = io.StringIO()
        err = io.StringIO()
        with patch("wallatag.cli.WallabagClient", return_value=client), \
             contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            code = cmd_manual(make_cfg(), make_args())
        self.assertEqual(code, 2)
        self.assertTrue(client.closed)
        self.assertIn("error fetching entries", err.getvalue())
        self.assertIn("tagged 0 articles", out.getvalue())

    def test_partial_run_feed_error_exits_0(self):
        # Articles were presented before the feed died: partial run, exit 0.
        client = FailingFeedClient()
        out = io.StringIO()
        err = io.StringIO()
        with patch("wallatag.cli.WallabagClient", return_value=client), \
             patch("builtins.input", side_effect=["", ""]), \
             contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(err):
            code = cmd_manual(make_cfg(), make_args())
        self.assertEqual(code, 0)
        self.assertIn("error fetching entries", err.getvalue())
        self.assertIn("tagged 0 articles", out.getvalue())


class StoreOpenErrorTest(ManualBase):
    def test_store_open_failure_message_and_exit_2(self):
        client = FakeClient(entries=[entry(1, "x")])
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "no_such_dir", "store.db")
            err = io.StringIO()
            with patch("wallatag.cli.WallabagClient", return_value=client), \
                 contextlib.redirect_stderr(err):
                code = cmd_manual(make_cfg(store_path=bad_path), make_args())
        self.assertEqual(code, 2)
        self.assertIn("could not open store", err.getvalue())
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
