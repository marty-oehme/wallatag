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
import re
import requests
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from wallatag.auto import AutoSummary, run_auto, summary_line
from wallatag.cli import cmd_run
from wallatag.config import (
    Config,
    FocusGroup,
    StoreConfig,
    TaggerConfig,
    WallabagConfig,
)
from wallatag.llm import LLMError
from wallatag.store import Store
from wallatag.tagger import KeywordTagger
from wallatag.wallabag import WallabagError, _should_fetch, _tag_labels


class RaisingTagger:
    """Tagger stub whose suggest() always raises LLMError (model down)."""

    def suggest(self, entry):
        raise LLMError("model unavailable", status=500)


def entry(
    eid,
    title,
    url="https://example.com/x",
    domain="example.com",
    content="",
    reading_time=5,
    tags=(),
):
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
    def __init__(
        self,
        entries=(),
        tags=(),
        fail_first=0,
        feed_error=None,
        feed_fail_after=None,
    ):
        self.entries = list(entries)
        self.tags = list(tags)
        self.fail_first = fail_first
        self.feed_error = feed_error
        self.feed_fail_after = feed_fail_after
        self.add_calls = []
        self.closed = False

    def iter_untagged(self, per_page=30, ignored_tags=(), ignored_regex=()):
        if self.feed_error is not None:
            raise self.feed_error
        # Faithful to WallabagClient.iter_untagged: drop entries whose tags
        # are non-empty and not all in the ignore-any list (literal or regex).
        ignored = frozenset(t.casefold() for t in ignored_tags)
        patterns = tuple(re.compile(p, re.IGNORECASE) for p in ignored_regex)
        entries = [
            dict(item)
            for item in self.entries
            if _should_fetch(_tag_labels(item.get("tags")), ignored, patterns)
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


class FlakyAddTagsClient(FakeClient):
    """add_tags raises a scripted sequence of exceptions before succeeding.

    ``errors`` is a list of exception instances (or None entries) popped per
    add_tags call; when the list is exhausted the call succeeds. Every attempt
    is counted in ``add_attempts`` (callers use it to assert retry counts);
    ``add_calls`` keeps the base FakeClient contract of recording only
    successful calls.
    """

    def __init__(self, errors=(), **kwargs):
        super().__init__(**kwargs)
        self.errors = list(errors)
        self.add_attempts = []

    def add_tags(self, entry_id, tags):
        self.add_attempts.append((entry_id, sorted(tags)))
        if self.errors:
            raise self.errors.pop(0)
        self.add_calls.append((entry_id, sorted(tags)))


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


def make_tagger(
    existing_tags=(), groups=None, tag_policy="all", max_applied_tags=10
):
    return KeywordTagger(
        groups or {},
        max_applied_tags=max_applied_tags,
        tag_policy=tag_policy,
        existing_tags=list(existing_tags),
    )


def make_cfg(
    url="https://wallabag.example.com", max_articles=None, store_path=None
):
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
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
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

    def run_auto(
        self,
        client,
        tagger=None,
        store=None,
        cfg=None,
        dry_run=False,
        fallback_tagger=None,
        add_tags_retries=None,
        add_tags_retry_delay=None,
    ):
        tagger = tagger if tagger is not None else make_tagger()
        store = store if store is not None else Store(None)
        cfg = cfg if cfg is not None else Config()
        retry_kwargs = {}
        if add_tags_retries is not None:
            retry_kwargs["add_tags_retries"] = add_tags_retries
        if add_tags_retry_delay is not None:
            retry_kwargs["add_tags_retry_delay"] = add_tags_retry_delay
        with self.assertLogs("wallatag.auto", level="INFO") as cm:
            summary = run_auto(
                client,
                tagger,
                store,
                cfg,
                dry_run=dry_run,
                fallback_tagger=fallback_tagger,
                **retry_kwargs,
            )
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
        with (
            ctx,
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
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
        self.assertEqual(
            client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])]
        )


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
            with contextlib.closing(sqlite3.connect(db)) as conn:
                seen = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
                decisions = conn.execute(
                    "SELECT COUNT(*) FROM decisions"
                ).fetchone()[0]
            self.assertEqual((seen, decisions), (0, 0))
            self.assertTrue(summary.dry_run)
            self.assertEqual((summary.tagged, summary.tags_applied), (1, 1))
        self.assertTrue(any("would tag 1: Pomodoro" in m for m in messages))


class IgnoredTagsTest(AutoBase):
    """[tagger] ignore_tags flows through run_auto to the client filter."""

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

    def test_ignored_regex_tag_article_is_presented_and_tagged(self):
        # ignore_tags_regex=("todo|fix",): an article tagged ["todo"] is
        # still fetched (its only tag matches a pattern).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["todo"])],
            tags=["Pomodoro"],
        )
        summary, _ = self.run_auto(
            client,
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            cfg=self._cfg(ignore_tags_regex=("todo|fix",)),
        )
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertEqual((summary.presented, summary.tagged), (1, 1))

    def test_mixed_tags_regex_article_not_presented(self):
        # A tag matching neither a literal entry nor a regex pattern
        # disqualifies the article.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus", tags=["todo", "something"])],
            tags=["Pomodoro"],
        )
        # Direct run_auto: nothing is presented, so no logs are emitted
        # (the assertLogs helper would fail on an empty log stream).
        summary = run_auto(
            client,
            make_tagger(existing_tags=["Pomodoro"]),
            Store(None),
            self._cfg(ignore_tags_regex=("todo|fix",)),
        )
        self.assertEqual(client.add_calls, [])
        self.assertEqual(summary.presented, 0)


class TagPolicyTest(AutoBase):
    def _groups(self):
        return {
            "a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))
        }

    def test_only_existing_drops_rule_tags(self):
        client = FakeClient(entries=[entry(1, "pomodoro focus")])
        summary, messages = self.run_auto(
            client,
            tagger=make_tagger(
                groups=self._groups(), tag_policy="only-existing"
            ),
        )
        self.assertEqual(client.add_calls, [])
        self.assertEqual(
            (summary.presented, summary.tagged, summary.skipped), (1, 0, 1)
        )
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
    def test_transient_failure_retried_and_run_continues(self):
        # One transient add_tags failure (WallabagError status=None) is
        # retried by the engine (bug d968e0f): article 1 is tagged on the
        # retry, and article 2 is still processed — a per-article failure
        # never aborts the run.
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro one"), entry(2, "pomodoro two")],
            tags=["Pomodoro"],
            errors=[WallabagError("boom")],
        )
        summary, messages = self.run_auto(
            client,
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            add_tags_retry_delay=0,
        )
        self.assertEqual(summary.tagged, 2)
        self.assertEqual(
            client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])]
        )
        self.assertTrue(any("tagging failed 1" in m for m in messages))
        self.assertTrue(any("retry 1/2" in m for m in messages))
        self.assertTrue(any("tagged 1" in m for m in messages))
        self.assertTrue(any("tagged 2" in m for m in messages))


class AddTagsRetryTest(AutoBase):
    """Engine-owned, classified retries around add_tags (bug d968e0f).

    Transient failures (transport-level WallabagError with status None, or a
    retryable HTTP status) are re-attempted with exponential backoff, then the
    article is deferred (unmarked) on final failure; deterministic 4xx
    failures are never retried and the article stays seen.
    """

    def test_transient_twice_then_success_tags(self):
        # (1) Two transient transport failures (status None) then success:
        # up to 3 attempts total, warning logged per retry, article tagged
        # and NOT unmarked (it stays seen like any tagged article).
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[WallabagError("boom"), WallabagError("boom again")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    add_tags_retry_delay=0,
                )
                self.assertTrue(store.is_seen(1))
            finally:
                store.close()

        self.assertEqual(summary.tagged, 1)
        self.assertEqual(len(client.add_attempts), 3)
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertTrue(any("retry 1/2" in m for m in messages))
        self.assertTrue(any("retry 2/2" in m for m in messages))
        self.assertTrue(any("tagged 1" in m for m in messages))

    def test_503_then_success_tags(self):
        # (2) A retryable HTTP status (503) is re-attempted once and succeeds.
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[WallabagError("server hiccup", status=503)],
        )
        summary, messages = self.run_auto(
            client,
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            add_tags_retry_delay=0,
        )
        self.assertEqual(summary.tagged, 1)
        self.assertEqual(len(client.add_attempts), 2)
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertTrue(any("retry 1/2" in m for m in messages))

    def test_permanent_400_called_once_stays_seen(self):
        # (3) A deterministic 400 is NOT retried: exactly one add_tags call,
        # outcome "tagging failed", and the article stays seen (deliberately
        # dropped — re-processing would fail identically).
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[WallabagError("bad request", status=400)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    add_tags_retry_delay=0,
                )
                self.assertTrue(store.is_seen(1))
            finally:
                store.close()

        self.assertEqual(len(client.add_attempts), 1)
        self.assertEqual(client.add_calls, [])
        self.assertEqual(
            (summary.presented, summary.tagged, summary.skipped), (1, 0, 0)
        )
        self.assertTrue(any("tagging failed 1" in m for m in messages))
        self.assertFalse(any("retry" in m for m in messages))

    def test_transient_exhausted_unmarks_for_next_run(self):
        # (4) Three transient failures exhaust the retries: outcome "tagging
        # failed", and the article is UNMARKED exactly once (deferred to the
        # next run, mirroring the LLM-failure path).
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[
                WallabagError("boom"),
                WallabagError("boom again"),
                WallabagError("boom thrice"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=make_tagger(existing_tags=["Pomodoro"]),
                    store=store,
                    add_tags_retry_delay=0,
                )
                self.assertFalse(store.is_seen(1))
            finally:
                store.close()

        self.assertEqual(len(client.add_attempts), 3)
        self.assertEqual(client.add_calls, [])
        self.assertEqual(
            (summary.presented, summary.tagged, summary.skipped), (1, 0, 0)
        )
        self.assertTrue(any("tagging failed 1" in m for m in messages))

    def test_raw_request_exception_is_retried(self):
        # (5) A raw requests.RequestException from a duck-typed client is
        # transport-level and retried (here retries=1 -> 2 attempts total).
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[requests.ConnectionError("connection refused")],
        )
        summary, messages = self.run_auto(
            client,
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            add_tags_retries=1,
            add_tags_retry_delay=0,
        )
        self.assertEqual(summary.tagged, 1)
        self.assertEqual(len(client.add_attempts), 2)
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertTrue(any("retry 1/1" in m for m in messages))

    def test_run_auto_defaults_flow_through(self):
        # (7) run_auto's default retry params reach process_entry: a transient
        # failure is retried (default ADD_TAGS_RETRIES=2 -> 2 attempts) even
        # though only add_tags_retry_delay=0 is passed at the run_auto level.
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[WallabagError("boom")],
        )
        summary, messages = self.run_auto(
            client,
            tagger=make_tagger(existing_tags=["Pomodoro"]),
            add_tags_retry_delay=0,
        )
        self.assertEqual(summary.tagged, 1)
        self.assertEqual(len(client.add_attempts), 2)
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertTrue(any("retry 1/2" in m for m in messages))


class FeedErrorTest(AutoBase):
    def test_total_feed_failure_no_traceback(self):
        client = FakeClient(feed_error=WallabagError("cannot connect"))
        summary, messages = self.run_auto(client)
        self.assertEqual(summary.presented, 0)
        self.assertTrue(summary.feed_error)
        self.assertTrue(any("error fetching entries" in m for m in messages))

    def test_partial_feed_failure_stops_after_presented(self):
        client = FakeClient(
            entries=[
                entry(1, "pomodoro focus"),
                entry(2, "pomodoro again"),
                entry(3, "pomodoro third"),
            ],
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
        client = FakeClient(
            entries=[entry(1, "alpha beta")], tags=["alpha", "beta"]
        )
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
        self.assertEqual(
            (summary.presented, summary.tagged, summary.skipped), (1, 0, 1)
        )
        self.assertTrue(any("no suggestions: 1" in m for m in messages))


class LLMErrorTest(AutoBase):
    def test_llm_error_skips_entry_and_run_continues(self):
        # The tagger is a constructor arg to run_auto: inject a stub whose
        # suggest() raises LLMError for every entry.
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        summary, messages = self.run_auto(client, tagger=RaisingTagger())

        self.assertEqual(
            (summary.presented, summary.tagged, summary.skipped), (2, 0, 2)
        )
        # Every LLM-failed article is counted separately (subset of skipped).
        self.assertEqual(summary.llm_failed, 2)
        self.assertEqual(client.add_calls, [])
        # An LLM failure is an article-level skip, NOT a feed error.
        self.assertFalse(summary.feed_error)
        self.assertTrue(any("LLM tagging failed 1" in m for m in messages))
        self.assertTrue(any("LLM tagging failed 2" in m for m in messages))


class LLMDeferralTest(AutoBase):
    """LLM failures defer the article: its seen marker is removed so the next
    run presents it again instead of losing it permanently."""

    def test_llm_failure_unmarks_article_in_real_store(self):
        # (a) After the LLM failure the articles are NOT seen anymore.
        client = FakeClient(entries=[entry(1, "first"), entry(2, "second")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client, tagger=RaisingTagger(), store=store
                )
                self.assertFalse(store.is_seen(1))
                self.assertFalse(store.is_seen(2))
            finally:
                store.close()

        self.assertEqual((summary.skipped, summary.llm_failed), (2, 2))

    def test_llm_failure_counts(self):
        # (b) one article, one LLM failure: llm_failed and skipped are both 1.
        client = FakeClient(entries=[entry(1, "first")])
        summary, _ = self.run_auto(client, tagger=RaisingTagger())
        self.assertEqual(summary.llm_failed, 1)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(summary.tagged, 0)

    def test_summary_line_llm_failure_suffix(self):
        # (c) the summary line gains the ", N llm failures" suffix (plural
        # "failures" even for 1, keeping it greppable).
        summary = AutoSummary(presented=1, tagged=0, skipped=1, llm_failed=1)
        self.assertEqual(
            summary_line(summary),
            "run: tagged 0 articles (0 tags applied), skipped 1, 1 llm failures",
        )
        summary.llm_failed = 2
        self.assertEqual(
            summary_line(summary),
            "run: tagged 0 articles (0 tags applied), skipped 1, 2 llm failures",
        )

    def test_summary_line_dry_run_llm_failure_suffix(self):
        summary = AutoSummary(presented=1, tagged=0, skipped=1, llm_failed=1)
        self.assertEqual(
            summary_line(summary, dry_run=True),
            "dry run: would tag 0 articles (0 tags), skipped 1, 1 llm failures",
        )

    def test_summary_line_combines_feed_error_and_llm_failure(self):
        # Both failure suffixes stack, in order, after the base "run: tagged
        # ..." wording: ", feed error" first, then ", N llm failures".
        summary = AutoSummary(
            presented=2,
            tagged=1,
            tags_applied=2,
            skipped=2,
            llm_failed=2,
            feed_error=True,
        )
        self.assertEqual(
            summary_line(summary),
            "run: tagged 1 articles (2 tags applied), skipped 2, "
            "feed error, 2 llm failures",
        )

    def test_dry_run_llm_failure_has_no_side_effects(self):
        # (d) dry run marks nothing, so there is nothing to unmark.
        client = FakeClient(entries=[entry(1, "first")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client, tagger=RaisingTagger(), store=store, dry_run=True
                )
            finally:
                store.close()

            with contextlib.closing(sqlite3.connect(db)) as conn:
                seen = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
            self.assertEqual(seen, 0)
        self.assertEqual(summary.llm_failed, 1)


class LLMFallbackTest(AutoBase):
    """[ai] fallback_on_fail: the keyword tagger takes over per-article when
    the LLM fails, and the fallback respects the keyword-mode switches."""

    def test_fallback_suggests_and_tags(self):
        # (a) LLM fails, the keyword fallback suggests -> the article is
        # tagged via the normal apply path, counted as llm_fallback, and NOT
        # unmarked (it stays seen like any successfully tagged article).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=RaisingTagger(),
                    store=store,
                    fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
                )
                self.assertTrue(store.is_seen(1))
            finally:
                store.close()

        self.assertEqual(
            (
                summary.presented,
                summary.tagged,
                summary.skipped,
                summary.llm_failed,
                summary.llm_fallback,
            ),
            (1, 1, 0, 0, 1),
        )
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
        self.assertTrue(any("tagged 1" in m for m in messages))
        line = summary_line(summary)
        self.assertIn(", 1 via fallback", line)
        self.assertNotIn("llm failures", line)

    def test_fallback_empty_keeps_llm_failure_path(self):
        # (b) The fallback yields nothing: exactly today's failure path —
        # skipped, llm_failed, and the article unmarked when not dry_run.
        client = FakeClient(entries=[entry(1, "unrelated soup")])
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=RaisingTagger(),
                    store=store,
                    fallback_tagger=make_tagger(),  # no groups/tags -> nothing
                )
                self.assertFalse(store.is_seen(1))
            finally:
                store.close()

        self.assertEqual(
            (
                summary.presented,
                summary.tagged,
                summary.skipped,
                summary.llm_failed,
                summary.llm_fallback,
            ),
            (1, 0, 1, 1, 0),
        )
        self.assertEqual(client.add_calls, [])
        self.assertTrue(any("LLM tagging failed 1" in m for m in messages))
        self.assertEqual(
            summary_line(summary),
            "run: tagged 0 articles (0 tags applied), skipped 1, 1 llm failures",
        )

    def test_fallback_itself_failing_keeps_llm_failure_path(self):
        # A fallback tagger that raises must be swallowed: [] and today's path.
        class FailingFallback:
            def suggest(self, entry):
                raise RuntimeError("fallback exploded")

        client = FakeClient(entries=[entry(1, "first")])
        summary, messages = self.run_auto(
            client, tagger=RaisingTagger(), fallback_tagger=FailingFallback()
        )
        self.assertEqual(
            (summary.skipped, summary.llm_failed, summary.llm_fallback),
            (1, 1, 0),
        )
        self.assertTrue(any("LLM tagging failed 1" in m for m in messages))

    def test_no_fallback_default_unchanged(self):
        # (c) No fallback (default): today's path, byte-identical summary.
        client = FakeClient(entries=[entry(1, "first")])
        summary, _ = self.run_auto(client, tagger=RaisingTagger())
        self.assertEqual((summary.skipped, summary.llm_failed), (1, 1))
        self.assertEqual(summary.llm_fallback, 0)
        self.assertEqual(
            summary_line(summary),
            "run: tagged 0 articles (0 tags applied), skipped 1, 1 llm failures",
        )

    def test_fallback_respects_enable_vocabulary_false(self):
        # (d) A fallback with enable_vocabulary=False emits no source
        # "vocabulary" suggestions: only rule suggestions apply, and the
        # decision log records the "rules" source.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        groups = {
            "a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))
        }
        fallback = KeywordTagger(
            groups,
            max_applied_tags=10,
            tag_policy="all",
            existing_tags=["Pomodoro"],
            enable_vocabulary=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client,
                    tagger=RaisingTagger(),
                    store=store,
                    fallback_tagger=fallback,
                )
            finally:
                store.close()

            self.assertEqual(client.add_calls, [(1, ["productivity"])])
            self.assertEqual(
                decision_rows(db), [(1, "productivity", "accept", "rules")]
            )
        self.assertEqual((summary.tagged, summary.llm_fallback), (1, 1))
        self.assertIn(", 1 via fallback", summary_line(summary))

    def test_fallback_uses_keyword_sources_in_decisions(self):
        # Fallback suggestions carry their real sources into the decision log
        # (vocabulary + rules), exactly like a keyword-mode run.
        client = FakeClient(
            entries=[entry(1, "pomodoro guide")], tags=["Pomodoro"]
        )
        groups = {
            "a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))
        }
        fallback = make_tagger(
            existing_tags=["Pomodoro"], groups=groups, tag_policy="all"
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client,
                    tagger=RaisingTagger(),
                    store=store,
                    fallback_tagger=fallback,
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

    def test_fallback_dry_run_has_no_side_effects(self):
        # Dry-run fallback success: nothing is written and nothing is unmarked
        # (nothing was marked in the first place).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client,
                    tagger=RaisingTagger(),
                    store=store,
                    dry_run=True,
                    fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
                )
            finally:
                store.close()

            with contextlib.closing(sqlite3.connect(db)) as conn:
                seen = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
            self.assertEqual(seen, 0)
        self.assertEqual((summary.tagged, summary.llm_fallback), (1, 1))
        self.assertEqual(client.add_calls, [])

    def test_summary_line_via_fallback_suffix(self):
        # (e) ", N via fallback" is appended after the llm-failures segment.
        summary = AutoSummary(
            presented=1, tagged=1, tags_applied=1, llm_fallback=2
        )
        self.assertEqual(
            summary_line(summary),
            "run: tagged 1 articles (1 tags applied), skipped 0, 2 via fallback",
        )
        self.assertEqual(
            summary_line(summary, dry_run=True),
            "dry run: would tag 1 articles (1 tags), skipped 0, 2 via fallback",
        )
        # Both segments stack, fallback LAST.
        summary = AutoSummary(
            presented=2,
            tagged=1,
            tags_applied=1,
            skipped=1,
            llm_failed=1,
            llm_fallback=1,
        )
        self.assertEqual(
            summary_line(summary),
            "run: tagged 1 articles (1 tags applied), skipped 1, "
            "1 llm failures, 1 via fallback",
        )

    def test_fallback_per_article_scope(self):
        # (g) Per-article scope: the LLM is still tried on subsequent articles.
        # A stub whose suggest() raises once then succeeds proves the LLM keeps
        # being used after a fallback.
        calls = []

        class FlakyTagger:
            def suggest(self, entry):
                calls.append(entry["id"])
                if entry["id"] == 1:
                    raise LLMError("model unavailable", status=500)
                return []

        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "second")]
        )
        summary, _ = self.run_auto(
            client,
            tagger=FlakyTagger(),
            fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
        )
        # Article 1: LLM failed -> fallback suggested Pomodoro -> tagged.
        # Article 2: LLM (still tried) succeeded with no suggestions -> skipped.
        self.assertEqual(calls, [1, 2])
        self.assertEqual((summary.tagged, summary.skipped), (1, 1))
        self.assertEqual((summary.llm_failed, summary.llm_fallback), (0, 1))
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])

    def test_fallback_not_counted_when_add_tags_fails(self):
        # (a) The fallback suggested tags but add_tags raised a DETERMINISTIC
        # 400 (never retried by the engine): the article is seen-but-untagged,
        # so it must NOT be counted "via fallback" — no contradictory
        # "1 via fallback" next to "tagged 0".
        client = FlakyAddTagsClient(
            entries=[entry(1, "pomodoro focus")],
            tags=["Pomodoro"],
            errors=[WallabagError("boom", status=400)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, messages = self.run_auto(
                    client,
                    tagger=RaisingTagger(),
                    store=store,
                    fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
                    add_tags_retry_delay=0,
                )
                # A deterministic failure stays SEEN (deliberately dropped).
                self.assertTrue(store.is_seen(1))
            finally:
                store.close()

        self.assertEqual(
            (
                summary.presented,
                summary.tagged,
                summary.skipped,
                summary.llm_failed,
                summary.llm_fallback,
            ),
            (1, 0, 0, 0, 0),
        )
        self.assertEqual(client.add_calls, [])
        self.assertEqual(len(client.add_attempts), 1)  # no retry on a 400
        self.assertTrue(any("tagging failed 1" in m for m in messages))
        self.assertNotIn("via fallback", summary_line(summary))

    def test_fallback_not_counted_when_deduped_tags_empty(self):
        # (b) A fallback that reports suggestions whose tags dedupe to nothing
        # (a truthy non-list iterable yielding no suggestions bypasses the
        # `if not suggestions` check): the article is skipped and the fallback
        # is NOT counted — no contradictory "skipped 1, 1 via fallback".
        class EmptyTagsFallback:
            def suggest(self, entry):
                return (s for s in ())  # truthy, but yields no suggestions

        client = FakeClient(entries=[entry(1, "first")])
        summary, messages = self.run_auto(
            client,
            tagger=RaisingTagger(),
            fallback_tagger=EmptyTagsFallback(),
        )
        self.assertEqual(
            (
                summary.presented,
                summary.tagged,
                summary.skipped,
                summary.llm_failed,
                summary.llm_fallback,
            ),
            (1, 0, 1, 0, 0),
        )
        self.assertEqual(client.add_calls, [])
        self.assertNotIn("via fallback", summary_line(summary))
        self.assertTrue(any("no suggestions: 1" in m for m in messages))

    def test_fallback_rescued_logs_warning_not_error(self):
        # (c) A rescued fallback is logged at WARNING, not ERROR: the article
        # is still tagged, so ERROR is reserved for when the fallback also
        # fails (or is not configured).
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        with self.assertLogs("wallatag.auto", level="INFO") as cm:
            run_auto(
                client,
                RaisingTagger(),
                Store(None),
                Config(),
                fallback_tagger=make_tagger(existing_tags=["Pomodoro"]),
            )
        levels = {record.levelname for record in cm.records}
        self.assertIn("WARNING", levels)
        self.assertNotIn("ERROR", levels)
        self.assertTrue(
            any(
                record.levelname == "WARNING"
                and "LLM tagging failed 1" in record.getMessage()
                and "keyword fallback applied" in record.getMessage()
                for record in cm.records
            )
        )


class DecisionsTest(AutoBase):
    def test_accept_rows_written_with_sources(self):
        client = FakeClient(
            entries=[entry(1, "pomodoro guide")],
            tags=["Pomodoro"],
        )
        groups = {
            "a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))
        }
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            store = Store(db)
            try:
                summary, _ = self.run_auto(
                    client,
                    tagger=make_tagger(
                        existing_tags=["Pomodoro"],
                        groups=groups,
                        tag_policy="all",
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
        client = FakeClient(
            entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        code, out, _ = self.cmd_run(client=client)
        self.assertEqual(code, 0)
        self.assertTrue(client.closed)
        self.assertIn(
            "run: tagged 1 articles (1 tags applied), skipped 0", out
        )

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
            entries=[
                entry(1, "pomodoro focus"),
                entry(2, "pomodoro again"),
                entry(3, "pomodoro third"),
            ],
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
