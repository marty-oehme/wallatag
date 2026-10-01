"""Tests for flows (skipped when the prefect group isn't installed).

The wallatag_batch flow-run tests exercise the Prefect adapter at the
auto.py port level (style of tests/test_auto.py): the wallabag API is faked
with a FakeClient, the tagger is a real KeywordTagger, and only the flow's
own seams are patched (flows._build_tagger, flows.WallabagClient,
flows.load_config, the env-merge helpers). No subprocess is ever spawned.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
import os
import re
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from wallatag.config import Config, ConfigError, StoreConfig, WallabagConfig
from wallatag.llm import LLMError
from wallatag.tagger import KeywordTagger
from wallatag.wallabag import WallabagError, _should_fetch, _tag_labels


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
    """Minimal wallabag client fake (mirrors tests/test_auto.py)."""

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


def decision_rows(db_path):
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT entry_id, tag, action, source FROM decisions ORDER BY rowid"
        ).fetchall()


class LlmFailTagger:
    """Primary tagger whose suggest() always raises LLMError.

    Drives process_entry's llm_failed path (no fallback) exactly as a real
    LLM tagger error does.
    """

    def suggest(self, entry):
        raise LLMError("model unreachable")


class LlmFailFirstKeywordTagger(KeywordTagger):
    """Keyword tagger that fails like an LLM for entry id 1, then works.

    Lets a batch mix one llm_failed article with a normally tagged one.
    """

    def suggest(self, entry):
        if entry["id"] == 1:
            raise LLMError("model unreachable")
        return super().suggest(entry)


class PrefectFlowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "prefect not installed (uv sync --group prefect)"
            )
        import flows
        from prefect.cache_policies import NO_CACHE

        cls.flows = flows
        cls.NO_CACHE = NO_CACHE

    def setUp(self) -> None:
        # APILogHandler warns (UserWarning) when a run logger emits outside a
        # FlowRunContext. The flow tests call wallatag_batch.fn directly (no
        # flow run context), so every tagged outcome in this module would
        # warn. Silence it for the whole class. (Prefect 3.8.2's Setting has
        # no writable .value attribute, and os.environ is ineffective because
        # settings are cached, so temporary_settings is the correct API.)
        from prefect.settings import (
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW,
            temporary_settings,
        )

        self._missing_flow_ctx = temporary_settings(
            {PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: "ignore"}
        )
        self._missing_flow_ctx.__enter__()

    def tearDown(self) -> None:
        self._missing_flow_ctx.__exit__(None, None, None)

    def make_config(self, store_path=None, max_articles=50):
        """A Config the flow can drive (WallabagClient is patched anyway)."""
        return dataclasses.replace(
            Config(),
            wallabag=WallabagConfig(
                url="https://wallabag.example.com",
                client_id="cid",
                client_secret="secret",
                username="alice",
                password="wonderland",
            ),
            store=StoreConfig(path=store_path),
            max_articles=max_articles,
        )

    def run_batch(
        self,
        client,
        config=None,
        *,
        focus=None,
        max_articles=50,
        store_path=None,
        spy_task=False,
        tagger=None,
    ):
        """Run wallatag_batch with the adapter seams patched to engine fakes.

        ``spy_task=True`` replaces the tag-article task with a spy over the
        real engine function (auto.process_entry) so call counts per
        candidate can be asserted; with ``spy_task=False`` the real Prefect
        task runs end to end. ``tagger`` overrides the primary tagger (e.g.
        an LLM-failing stub) instead of the default KeywordTagger.
        """
        config = (
            config
            if config is not None
            else self.make_config(
                store_path=store_path, max_articles=max_articles
            )
        )
        if tagger is None:
            tagger = KeywordTagger(
                {},
                max_applied_tags=10,
                tag_policy="all",
                existing_tags=list(client.tags),
            )
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("flows.load_config", return_value=config)
            )
            stack.enter_context(
                patch("flows.llm_env_from_block", return_value={})
            )
            stack.enter_context(
                patch("flows.wallabag_env_from_block", return_value={})
            )
            stack.enter_context(patch("flows.Variable.get", return_value=None))
            stack.enter_context(
                patch("flows._build_tagger", return_value=(tagger, None, None))
            )
            stack.enter_context(
                patch("flows.WallabagClient", return_value=client)
            )
            task_mock = None
            if spy_task:
                # Spy over the task's own underlying function: each call is
                # recorded AND still drives the real engine code. Patches
                # tag_article.fn (not the Task object) so the call still runs
                # through the Prefect task engine and a TaskRunContext is
                # active (get_run_logger() requires one).
                task_mock = stack.enter_context(
                    patch(
                        "flows.tag_article.fn",
                        side_effect=self.flows.tag_article.fn,
                    )
                )
            with contextlib.redirect_stdout(out):
                result = self.flows.wallatag_batch.fn(
                    max_articles=max_articles, focus=focus
                )
        return result, out.getvalue(), task_mock

    def test_flow_defined(self) -> None:
        self.assertTrue(hasattr(self.flows, "wallatag_batch"))
        self.assertEqual(self.flows.wallatag_batch.name, "wallatag-batch")

    # -- env-merge helpers -------------------------------------------------

    def test_llm_env_from_block_fail_open(self) -> None:
        with (
            patch(
                "flows.LLMCredentials.load",
                side_effect=Exception("prefect server unreachable"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            env = self.flows.llm_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_wallabag_env_from_block_fail_open(self) -> None:
        with (
            patch(
                "flows.WallabagCredentials.load",
                side_effect=Exception("prefect server unreachable"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            env = self.flows.wallabag_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_variable_names_exact_no_secrets(self) -> None:
        self.assertEqual(
            set(self.flows.WALLATAG_VARIABLES),
            {
                "wallatag_tag_policy",
                "wallatag_max_applied_tags",
                "wallatag_ignore_tags",
                "wallatag_ignore_tags_regex",
                "wallatag_enable_vocabulary",
                "wallatag_enable_rules",
                "wallatag_enable_llm",
                "wallatag_ai_confidence_threshold",
                "wallatag_ai_use_focus_groups",
                "wallatag_ai_max_proposals",
                "wallatag_vocabulary_fields",
                "wallatag_vocabulary_skip_ignored_tags",
            },
        )
        self.assertEqual(len(self.flows.WALLATAG_VARIABLES), 12)
        self.assertEqual(
            len(set(self.flows.WALLATAG_VARIABLES)),
            len(self.flows.WALLATAG_VARIABLES),
        )
        joined = "|".join(self.flows.WALLATAG_VARIABLES).upper()
        for secret in ("CLIENT_SECRET", "PASSWORD", "API_KEY"):
            self.assertNotIn(secret, joined)
        # Prefect requires lowercase variable names; the env-var name is
        # derived mechanically via name.upper(), so the mapping to the
        # WALLATAG_* env names is exact and must never drift.
        self.assertEqual(
            {name.upper() for name in self.flows.WALLATAG_VARIABLES},
            {
                "WALLATAG_TAG_POLICY",
                "WALLATAG_MAX_APPLIED_TAGS",
                "WALLATAG_IGNORE_TAGS",
                "WALLATAG_IGNORE_TAGS_REGEX",
                "WALLATAG_ENABLE_VOCABULARY",
                "WALLATAG_ENABLE_RULES",
                "WALLATAG_ENABLE_LLM",
                "WALLATAG_AI_CONFIDENCE_THRESHOLD",
                "WALLATAG_AI_USE_FOCUS_GROUPS",
                "WALLATAG_AI_MAX_PROPOSALS",
                "WALLATAG_VOCABULARY_FIELDS",
                "WALLATAG_VOCABULARY_SKIP_IGNORED_TAGS",
            },
        )

    def test_variable_env_normalizes_values(self) -> None:
        values = {
            "wallatag_tag_policy": "all",
            "wallatag_max_applied_tags": 7,
            "wallatag_ai_confidence_threshold": 0.8,
            "wallatag_enable_llm": True,
            "wallatag_enable_rules": False,
            "wallatag_ignore_tags": "fix,_frigo",
        }

        def fake_get(name, default=None):
            return values.get(name, default)

        with patch("flows.Variable.get", side_effect=fake_get):
            env = self.flows.variable_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_TAG_POLICY": "all",
                "WALLATAG_MAX_APPLIED_TAGS": "7",
                "WALLATAG_AI_CONFIDENCE_THRESHOLD": "0.8",
                "WALLATAG_ENABLE_LLM": "true",
                "WALLATAG_ENABLE_RULES": "false",
                "WALLATAG_IGNORE_TAGS": "fix,_frigo",
            },
        )

    def test_variable_env_skips_none(self) -> None:
        with patch("flows.Variable.get", return_value=None):
            env = self.flows.variable_env()
        self.assertEqual(env, {})

    def test_variable_env_fail_open(self) -> None:
        with (
            patch(
                "flows.Variable.get",
                side_effect=Exception("prefect server unreachable"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            env = self.flows.variable_env()
        self.assertEqual(env, {})
        self.assertIn("prefect variables not available", out.getvalue())

    def test_variable_env_preserves_partial_set_on_read_failure(self) -> None:
        calls = {"n": 0}

        def fake_get(name, default=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "prefer-existing"
            if calls["n"] == 2:
                return 7
            raise Exception("prefect server unreachable")

        with (
            patch("flows.Variable.get", side_effect=fake_get),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            env = self.flows.variable_env()
        # Values read before the failure are kept; later names fall through
        # to the container env / TOML, and the warning is still printed.
        self.assertEqual(
            env,
            {
                "WALLATAG_TAG_POLICY": "prefer-existing",
                "WALLATAG_MAX_APPLIED_TAGS": "7",
            },
        )
        self.assertEqual(calls["n"], 3)
        self.assertIn("prefect variables not available", out.getvalue())

    def test_focus_groups_env_emits_joined_env_vars(self) -> None:
        raw = {
            "methods": {
                "keywords": ["howto", "tutorial"],
                "tags": ["dev"],
                "fields": ["title", "url"],
            }
        }
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "howto,tutorial",
                "WALLATAG_FOCUS_methods_TAGS": "dev",
                "WALLATAG_FOCUS_methods_FIELDS": "title,url",
            },
        )

    def test_focus_groups_env_omits_empty_keywords_tags(self) -> None:
        raw = {"methods": {"keywords": [], "tags": []}}
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(env, {})

    def test_focus_groups_env_empty_fields_list_disables_group(self) -> None:
        raw = {"methods": {"keywords": ["howto"], "fields": []}}
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "howto",
                "WALLATAG_FOCUS_methods_FIELDS": "",
            },
        )

    def test_focus_groups_env_absent_fields_key_omits_fields_var(self) -> None:
        raw = {"methods": {"keywords": ["howto"]}}
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(env, {"WALLATAG_FOCUS_methods_KEYWORDS": "howto"})

    def test_focus_groups_env_parses_json_string(self) -> None:
        raw = '{"methods": {"keywords": ["a", "b"], "fields": []}}'
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "a,b",
                "WALLATAG_FOCUS_methods_FIELDS": "",
            },
        )

    def test_focus_groups_env_none_returns_empty(self) -> None:
        with patch("flows.Variable.get", return_value=None):
            env = self.flows.focus_groups_env()
        self.assertEqual(env, {})

    def test_focus_groups_env_fail_open_on_read_error(self) -> None:
        with (
            patch(
                "flows.Variable.get",
                side_effect=Exception("prefect server unreachable"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            env = self.flows.focus_groups_env()
        self.assertEqual(env, {})
        self.assertIn("prefect variables not available", out.getvalue())

    def test_focus_groups_env_rejects_malformed_values(self) -> None:
        cases = [
            ["methods"],
            {"methods": ["howto"]},
            {"methods": {"keywords": ["howto", " "]}},
            {"methods": {"keywrods": ["howto"]}},
            {"methods": {"keywords": "howto"}},
            {"methods": {"keywords": [1]}},
            {"methods": {"fields": {}}},
            42,
            "not json at all",
        ]
        for raw in cases:
            with (
                self.subTest(raw=raw),
                patch("flows.Variable.get", return_value=raw),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self.flows.focus_groups_env()
                self.assertIn(
                    self.flows.FOCUS_GROUPS_VARIABLE, str(ctx.exception)
                )

    def test_focus_groups_env_rejects_casefold_duplicate_names(self) -> None:
        raw = {
            "Methods": {"keywords": ["upper"]},
            "methods": {"keywords": ["lower"]},
        }
        with patch("flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn(self.flows.FOCUS_GROUPS_VARIABLE, message)
        self.assertIn("'Methods'", message)
        self.assertIn("'methods'", message)
        self.assertIn("case-insensitively", message)

    def test_focus_groups_env_rejects_non_string_group_name(self) -> None:
        raw = {1: {"keywords": ["howto"]}}
        with patch("flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn(self.flows.FOCUS_GROUPS_VARIABLE, message)
        self.assertIn("group names must be strings", message)

    def test_focus_groups_env_empty_string_value_fails_loud(self) -> None:
        with patch("flows.Variable.get", return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.focus_groups_env()
        self.assertIn(self.flows.FOCUS_GROUPS_VARIABLE, str(ctx.exception))

    def test_focus_groups_env_sorted_order_determinism(self) -> None:
        raw = {
            "zeta": {"tags": ["z"]},
            "alpha": {"keywords": ["a"]},
            "methods": {"keywords": ["m"], "fields": ["title"]},
        }
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(
            list(env.keys()),
            [
                "WALLATAG_FOCUS_alpha_KEYWORDS",
                "WALLATAG_FOCUS_methods_KEYWORDS",
                "WALLATAG_FOCUS_methods_FIELDS",
                "WALLATAG_FOCUS_zeta_TAGS",
            ],
        )

    def test_focus_groups_env_emits_keywords_regex(self) -> None:
        raw = {
            "methods": {
                "keywords": ["howto"],
                "keywords_regex": ["^how.?to", "(?-i:GTD)"],
            }
        }
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "howto",
                "WALLATAG_FOCUS_methods_KEYWORDS_REGEX": "^how.?to,(?-i:GTD)",
            },
        )

    def test_focus_groups_env_omits_empty_keywords_regex(self) -> None:
        raw = {"methods": {"keywords_regex": []}}
        with patch("flows.Variable.get", return_value=raw):
            env = self.flows.focus_groups_env()
        self.assertEqual(env, {})

    def test_focus_groups_env_rejects_invalid_regex(self) -> None:
        raw = {"methods": {"keywords_regex": ["^["]}}
        with patch("flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn(self.flows.FOCUS_GROUPS_VARIABLE, message)
        self.assertIn("'methods'", message)
        self.assertIn("'^['", message)

    def test_focus_groups_env_rejects_comma_containing_regex(self) -> None:
        # A comma-containing pattern is compilable but the comma-separated env
        # translation would silently split it: reject fail-loud here so the
        # user must use TOML keywords_regex instead.
        raw = {"methods": {"keywords_regex": ["^a,b$"]}}
        with patch("flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn(self.flows.FOCUS_GROUPS_VARIABLE, message)
        self.assertIn("'methods'", message)
        self.assertIn("'^a,b$'", message)
        self.assertIn("comma", message)

    # -- flow: env merge ---------------------------------------------------

    def test_flow_env_precedence_merge(self) -> None:
        # merged env order: blocks < container env < variables < focus JSON.
        captured = {}

        def fake_load_config(env=None, **kwargs):
            captured["env"] = env
            return self.make_config()

        def fake_get(name, default=None):
            if name == "wallatag_tag_policy":
                return "all"
            if name == self.flows.FOCUS_GROUPS_VARIABLE:
                return {"methods": {"keywords": ["from-json"]}}
            return default

        client = FakeClient()
        with (
            patch("flows.load_config", side_effect=fake_load_config),
            patch(
                "flows.llm_env_from_block",
                return_value={"WALLATAG_AI_PROVIDER": "openai-compatible"},
            ),
            patch(
                "flows.wallabag_env_from_block",
                return_value={"WALLATAG_URL": "https://block.example"},
            ),
            patch.dict(
                os.environ,
                {
                    "WALLATAG_AI_PROVIDER": "ollama",
                    "WALLATAG_URL": "https://env.example",
                    "WALLATAG_DB": "/data/wallatag.db",
                },
                clear=True,
            ),
            patch("flows.Variable.get", side_effect=fake_get),
            patch("flows._build_tagger", return_value=(object(), None, None)),
            patch("flows.WallabagClient", return_value=client),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.flows.wallatag_batch.fn(max_articles=50)

        env = captured["env"]
        # Container env wins over the blocks for the same var.
        self.assertEqual(env["WALLATAG_AI_PROVIDER"], "ollama")
        self.assertEqual(env["WALLATAG_URL"], "https://env.example")
        # Variables fill gaps / win over container env.
        self.assertEqual(env["WALLATAG_TAG_POLICY"], "all")
        # The focus-groups JSON variable is folded in last.
        self.assertEqual(env["WALLATAG_FOCUS_methods_KEYWORDS"], "from-json")
        # Container env vars pass through untouched.
        self.assertEqual(env["WALLATAG_DB"], "/data/wallatag.db")

    def test_flow_falls_back_to_env_when_blocks_unavailable(self) -> None:
        # Block reads failing (no Prefect server) falls back to container env.
        captured = {}

        def fake_load_config(env=None, **kwargs):
            captured["env"] = env
            return self.make_config()

        with (
            patch("flows.load_config", side_effect=fake_load_config),
            patch(
                "flows.LLMCredentials.load",
                side_effect=Exception("prefect server unreachable"),
            ),
            patch(
                "flows.WallabagCredentials.load",
                side_effect=Exception("prefect server unreachable"),
            ),
            patch.dict(
                os.environ, {"WALLATAG_IGNORE_TAGS": "fix"}, clear=True
            ),
            patch("flows.Variable.get", return_value=None),
            patch("flows._build_tagger", return_value=(object(), None, None)),
            patch("flows.WallabagClient", return_value=FakeClient()),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            self.flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(captured["env"], {"WALLATAG_IGNORE_TAGS": "fix"})
        self.assertIn("falling back to config/env", out.getvalue())

    # -- flow: focus splitting ---------------------------------------------

    def test_flow_focus_param_split_passed_to_apply_run_overrides(
        self,
    ) -> None:
        # The inlined comma-split must behave exactly like the removed
        # _split_focus helper: None and empty/whitespace-only values pass
        # focus=None (no narrowing); segments are stripped and empty ones
        # dropped.
        cases = [
            (None, None),
            ("", None),
            ("   ", None),
            (" , , ", None),
            ("methods, languages", ["methods", "languages"]),
            (" methods , , languages ", ["methods", "languages"]),
        ]
        for focus, expected in cases:
            with self.subTest(focus=focus):
                captured = {}

                def fake_apply(config, **kwargs):
                    captured.update(kwargs)
                    return config

                with (
                    patch(
                        "flows.load_config", return_value=self.make_config()
                    ),
                    patch("flows.apply_run_overrides", side_effect=fake_apply),
                    patch("flows.llm_env_from_block", return_value={}),
                    patch("flows.wallabag_env_from_block", return_value={}),
                    patch.dict(os.environ, {}, clear=True),
                    patch("flows.Variable.get", return_value=None),
                    patch(
                        "flows._build_tagger",
                        return_value=(object(), None, None),
                    ),
                    patch("flows.WallabagClient", return_value=FakeClient()),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.flows.wallatag_batch.fn(max_articles=50, focus=focus)
                self.assertEqual(captured["focus"], expected)
                self.assertEqual(captured["max_articles"], 50)

    def test_flow_unknown_focus_raises(self) -> None:
        # An unknown focus name fails the run loudly (ConfigError propagates
        # from apply_run_overrides instead of silently ignoring the group).
        client = FakeClient(entries=[])
        with self.assertRaises(ConfigError) as ctx:
            self.run_batch(client, focus="nope")
        self.assertIn("unknown focus group 'nope'", str(ctx.exception))

    # -- flow: engine driving ----------------------------------------------

    def test_flow_drives_engine_one_task_per_article(self) -> None:
        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "pomodoro again")],
            tags=["Pomodoro"],
        )
        result, out, task_mock = self.run_batch(client, spy_task=True)
        # One tag-article task call per candidate article, in feed order.
        self.assertEqual(task_mock.call_count, 2)
        self.assertEqual(
            [call.args[0]["id"] for call in task_mock.call_args_list], [1, 2]
        )
        # Tags were applied through the shared engine.
        self.assertEqual(
            client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])]
        )
        # The summary line is returned AND printed.
        self.assertEqual(
            result, "run: tagged 2 articles (2 tags applied), skipped 0"
        )
        self.assertIn(result, out)
        self.assertTrue(client.closed)

    def test_flow_tags_applied_and_decisions_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "s.db")
            client = FakeClient(
                entries=[entry(1, "pomodoro focus")], tags=["Pomodoro"]
            )
            result, _, _ = self.run_batch(client, store_path=db)
            self.assertEqual(
                result, "run: tagged 1 articles (1 tags applied), skipped 0"
            )
            self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])
            self.assertEqual(
                decision_rows(db), [(1, "Pomodoro", "accept", "vocabulary")]
            )

    def test_flow_real_task_runs_end_to_end(self) -> None:
        # No spy: the actual @task runs through the Prefect task engine and
        # still drives the shared engine code per article.
        client = FakeClient(
            entries=[entry(1, "pomodoro focus"), entry(2, "pomodoro again")],
            tags=["Pomodoro"],
        )
        result, out, task_mock = self.run_batch(client)
        self.assertIsNone(task_mock)
        self.assertEqual(
            client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])]
        )
        self.assertEqual(
            result, "run: tagged 2 articles (2 tags applied), skipped 0"
        )
        self.assertIn(result, out)
        self.assertTrue(client.closed)

    def test_flow_tagged_log_line_run_attributed(self) -> None:
        """A tag-article task run logs its tagged line via the run logger.

        get_run_logger() returns a PrefectLogAdapter over
        ``prefect.task_runs`` whose ``extra`` dict (task_run_id,
        task_run_name, task_name, ...) lands on each emitted record, so
        assertLogs on that logger captures the per-article line with the
        article id, title, and applied tags AND the task-run attribution
        extras — the task run id is set and the task name is ``tag-article``.
        (The flow runs via run_batch, which calls wallatag_batch.fn
        directly, so there is no FlowRunContext and flow_run_id is None
        here; production attribution of the line to the task run in the
        UI/DB is done by prefect's APILogHandler.) The engine's own INFO
        lines (wallatag.auto) are dropped in the flow (root handler at
        WARNING, basicConfig a no-op; bug edf2338). The flow's return value
        is unaffected.
        """
        client = FakeClient(
            entries=[entry(42, "fix and todo")], tags=["fix", "todo"]
        )
        with self.assertLogs("prefect.task_runs", level="INFO") as cm:
            result, out, _ = self.run_batch(client)
        joined = "\n".join(cm.output)
        self.assertIn("tagged article 42 (fix and todo): fix, todo", joined)
        tagged = next(
            rec for rec in cm.records if "tagged article" in rec.getMessage()
        )
        self.assertIsNotNone(tagged.task_run_id)
        self.assertEqual(tagged.task_name, "tag-article")
        self.assertEqual(tagged.task_run_name, "tag-article")
        self.assertIsNone(tagged.flow_run_id)
        self.assertEqual(client.add_calls, [(42, ["fix", "todo"])])
        self.assertEqual(
            result, "run: tagged 1 articles (2 tags applied), skipped 0"
        )
        self.assertIn(result, out)
        self.assertTrue(client.closed)

    def test_flow_tagging_failure_logged_run_attributed(self) -> None:
        """A tag-article task run logs its final tagging failure via the run logger.

        The engine's own ERROR line (wallatag.auto) never surfaces in the
        flow (root handler at WARNING, basicConfig a no-op; bug edf2338), so
        the adapter logs the exhausted add_tags failure through
        get_run_logger() on ``prefect.task_runs`` — entry id and title, at
        ERROR. The transient WallabagError (status None) is retried by the
        engine and exhausted, so the task logs the ERROR line, raises
        TaggingFailedError (task run Failed), and the engine defers the
        article (unmark_seen). Because this run's ONLY presented article
        failed, the flow itself also raises RuntimeError (every presented
        article failed to tag) — mirror of cmd_run, a total failure looks like
        a failure to the scheduler. (time.sleep is patched so the engine's
        backoff doesn't slow the test.)
        """
        client = FakeClient(
            entries=[entry(7, "pomodoro doomed")], tags=["Pomodoro"]
        )

        def always_fail_add_tags(entry_id, tags):
            raise WallabagError("boom")

        client.add_tags = always_fail_add_tags
        # side_effect=time.sleep: the patch only RECORDS engine backoff calls;
        # it must not freeze the shared stdlib time module, or Prefect's
        # ephemeral-server readiness loop (which advances its timeout budget
        # per iteration, not per wall second) busy-spins to a startup timeout
        # when this test runs in isolation.
        with (
            patch("wallatag.auto.time.sleep", side_effect=time.sleep),
            self.assertLogs("prefect.task_runs", level="INFO") as cm,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.run_batch(client)
        joined = "\n".join(cm.output)
        self.assertIn("tagging failed article 7 (pomodoro doomed)", joined)
        failed = next(
            rec
            for rec in cm.records
            if "tagging failed article" in rec.getMessage()
        )
        self.assertEqual(failed.levelno, logging.ERROR)
        self.assertIsNotNone(failed.task_run_id)
        self.assertEqual(failed.task_name, "tag-article")
        self.assertEqual(failed.task_run_name, "tag-article")
        self.assertEqual(client.add_calls, [])
        self.assertIn("no article was tagged", str(ctx.exception))
        self.assertIn("1 presented", str(ctx.exception))
        self.assertIn("1 tagging failures", str(ctx.exception))
        self.assertTrue(client.closed)

    def test_tag_article_failed_state_carries_result(self) -> None:
        """A tagging-failed task run ends in state Failed, carrying the result.

        The engine produced suggestions but the final add_tags POST failed
        deterministically (403, no engine retries): the task logs the ERROR
        line and raises TaggingFailedError, so Prefect marks the task run
        Failed. Called with return_state=True, the returned State carries the
        TaggingFailedError whose .result still holds the per-article deltas —
        presented but neither tagged nor skipped (deterministic 4xx keeps the
        article seen; EntryResult docstring, wallatag/auto.py).
        """
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(store_path=os.path.join(tmp, "s.db"))
            tagger = KeywordTagger(
                {},
                max_applied_tags=10,
                tag_policy="all",
                existing_tags=["Pomodoro"],
            )
            client = FakeClient(tags=["Pomodoro"])

            def forbidden_add_tags(entry_id, tags):
                raise WallabagError("forbidden", status=403)

            client.add_tags = forbidden_add_tags
            store = self.flows.Store(config.store.path)
            try:
                state = self.flows.tag_article(
                    entry(7, "pomodoro doomed"),
                    client=client,
                    tagger=tagger,
                    store=store,
                    cfg=config,
                    return_state=True,
                )
            finally:
                store.close()
        self.assertTrue(state.is_failed())
        exc = state.result(raise_on_failure=False)
        self.assertIsInstance(exc, self.flows.TaggingFailedError)
        self.assertEqual(exc.result.outcome, "tagging failed")
        self.assertEqual(exc.result.entry_id, 7)
        self.assertEqual(exc.result.presented, 1)
        self.assertEqual(exc.result.tagged, 0)
        self.assertEqual(exc.result.skipped, 0)

    def test_flow_tagging_failure_task_failed_batch_continues(self) -> None:
        """A tagging-failed task run shows Failed; the batch keeps going.

        Entry 1's add_tags fails deterministically (403, no retries, no
        backoff), so its tag-article task run ends in state Failed (the
        engine logs ``Finished in state Failed(...)`` through
        ``prefect.task_runs``, the same channel the dashboard reads). The
        flow catches TaggingFailedError and CONTINUES: entry 2 is tagged
        through the same engine, the failed article's carried deltas keep
        the summary honest (presented but neither tagged nor skipped), and
        the flow itself still completes with the summary line.
        """
        client = FakeClient(
            entries=[entry(1, "pomodoro doomed"), entry(2, "pomodoro focus")],
            tags=["Pomodoro"],
        )

        def forbidden_add_tags(entry_id, tags):
            if entry_id == 1:
                raise WallabagError("forbidden", status=403)
            client.add_calls.append((entry_id, sorted(tags)))

        client.add_tags = forbidden_add_tags
        with self.assertLogs("prefect.task_runs", level="INFO") as cm:
            result, out, _ = self.run_batch(client)
        joined = "\n".join(cm.output)
        self.assertIn("tagging failed article 1 (pomodoro doomed)", joined)
        self.assertIn("Finished in state Failed(", joined)
        # The batch continued past the failed article.
        self.assertEqual(client.add_calls, [(2, ["Pomodoro"])])
        # presented=1 but neither tagged nor skipped for the failed article.
        self.assertEqual(
            result, "run: tagged 1 articles (1 tags applied), skipped 0"
        )
        self.assertIn(result, out)
        self.assertTrue(client.closed)

    def test_tag_article_llm_failed_state_carries_result(self) -> None:
        """An LLM-failed task run ends in state Failed, carrying the result.

        The tagger raised LLMError and there is no fallback, so the engine
        reports outcome ``"llm_failed"``: the article is presented and DEFERRED
        (skipped=1, llm_failed=1, unmark_seen keeps it in the queue), never
        tagged. The task logs the ERROR line and raises TaggingFailedError, so
        Prefect marks the task run Failed. Called with return_state=True, the
        returned State carries the TaggingFailedError whose .result still holds
        the per-article deltas.
        """
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(store_path=os.path.join(tmp, "s.db"))
            client = FakeClient(tags=["Pomodoro"])
            store = self.flows.Store(config.store.path)
            try:
                state = self.flows.tag_article(
                    entry(7, "pomodoro doomed"),
                    client=client,
                    tagger=LlmFailTagger(),
                    store=store,
                    cfg=config,
                    return_state=True,
                )
            finally:
                store.close()
        self.assertTrue(state.is_failed())
        exc = state.result(raise_on_failure=False)
        self.assertIsInstance(exc, self.flows.TaggingFailedError)
        self.assertEqual(exc.result.outcome, "llm_failed")
        self.assertEqual(exc.result.entry_id, 7)
        self.assertEqual(exc.result.presented, 1)
        self.assertEqual(exc.result.tagged, 0)
        self.assertEqual(exc.result.skipped, 1)
        self.assertEqual(exc.result.llm_failed, 1)

    def test_flow_llm_failure_task_failed_batch_continues(self) -> None:
        """An LLM-failed task run shows Failed; the batch keeps going.

        Entry 1's tagger raises LLMError (outcome ``"llm_failed"``, no
        fallback), so its tag-article task run ends in state Failed and the
        run logger carries the ERROR line (the engine's wallatag.auto ERROR
        never surfaces). The flow catches TaggingFailedError and CONTINUES:
        entry 2 is tagged, the failed article's deltas keep the summary honest
        (presented and skipped, llm_failed counted), and the flow itself still
        completes because it is a partial run.
        """
        client = FakeClient(
            entries=[entry(1, "pomodoro doomed"), entry(2, "pomodoro focus")],
            tags=["Pomodoro"],
        )
        tagger = LlmFailFirstKeywordTagger(
            {},
            max_applied_tags=10,
            tag_policy="all",
            existing_tags=["Pomodoro"],
        )
        with self.assertLogs("prefect.task_runs", level="INFO") as cm:
            result, out, _ = self.run_batch(client, tagger=tagger)
        joined = "\n".join(cm.output)
        self.assertIn("LLM tagging failed article 1 (pomodoro doomed)", joined)
        self.assertIn("Finished in state Failed(", joined)
        # The batch continued past the failed article.
        self.assertEqual(client.add_calls, [(2, ["Pomodoro"])])
        self.assertEqual(
            result,
            "run: tagged 1 articles (1 tags applied), skipped 1, "
            "1 llm failures",
        )
        self.assertIn(result, out)
        self.assertTrue(client.closed)

    def test_flow_all_articles_failed_raises(self) -> None:
        """A run where every presented article failed to tag fails the flow.

        With the tagger down for every candidate, each tag-article task run is
        Failed and the flow itself raises RuntimeError (nothing was tagged), so
        the flow run shows Failed and can notify — unlike before, an all-failed
        run looked byte-for-byte like a clean one. Partial runs (at least one
        article tagged or normally skipped) still return their summary.
        """
        client = FakeClient(
            entries=[
                entry(1, "pomodoro doomed"),
                entry(2, "pomodoro again"),
            ],
            tags=["Pomodoro"],
        )
        with (
            self.assertLogs("prefect.task_runs", level="INFO") as cm,
            self.assertRaises(RuntimeError) as ctx,
        ):
            self.run_batch(client, tagger=LlmFailTagger())
        joined = "\n".join(cm.output)
        self.assertIn("LLM tagging failed article 1 (pomodoro doomed)", joined)
        self.assertIn("LLM tagging failed article 2 (pomodoro again)", joined)
        self.assertEqual(joined.count("Finished in state Failed("), 2)
        self.assertIn("no article was tagged", str(ctx.exception))
        self.assertIn("2 presented", str(ctx.exception))
        self.assertIn("2 llm failures", str(ctx.exception))
        self.assertEqual(client.add_calls, [])
        self.assertTrue(client.closed)

    def test_tag_article_task_disables_result_caching(self) -> None:
        """Per-article results must never be cache-reused (dedupe is the Store's job).

        The default cache policy's input hashing tries to serialize the shared
        runtime objects (client/tagger/store/cfg) and logged a HashError per task
        run (bug 8d10af5); the task pins cache_policy=NO_CACHE.
        """
        self.assertIs(self.flows.tag_article.cache_policy, self.NO_CACHE)

    # -- flow: feed errors -------------------------------------------------

    def test_flow_total_feed_error_raises(self) -> None:
        client = FakeClient(feed_error=WallabagError("cannot connect"))
        with self.assertRaises(RuntimeError) as ctx:
            self.run_batch(client)
        self.assertIn("feed", str(ctx.exception).lower())
        self.assertTrue(client.closed)

    def test_flow_partial_feed_error_returns_summary(self) -> None:
        client = FakeClient(
            entries=[
                entry(1, "pomodoro focus"),
                entry(2, "pomodoro again"),
                entry(3, "pomodoro third"),
            ],
            tags=["Pomodoro"],
            feed_fail_after=2,  # two entries delivered, then the feed dies
        )
        result, out, _ = self.run_batch(client)
        self.assertEqual(
            result,
            "run: tagged 2 articles (2 tags applied), skipped 0, feed error",
        )
        self.assertIn("feed error", out)
        self.assertTrue(client.closed)

    # -- flow: short-circuit ------------------------------------------------

    def test_flow_max_articles_zero_returns_empty_without_client(self) -> None:
        with (
            patch("flows.WallabagClient") as client_mock,
            patch("flows.load_config") as load_mock,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            result = self.flows.wallatag_batch.fn(max_articles=0)
        self.assertEqual(result, "")
        self.assertEqual(out.getvalue(), "")
        client_mock.assert_not_called()
        load_mock.assert_not_called()

    def test_flow_negative_max_articles_raises(self) -> None:
        # -1 passes the == 0 short-circuit, so the shared apply_run_overrides
        # validation must fail the run loudly (ConfigError), matching the
        # CLI's "--max must be a non-negative integer".
        client = FakeClient(entries=[])
        with self.assertRaises(ConfigError) as ctx:
            self.run_batch(client, max_articles=-1)
        self.assertEqual(
            str(ctx.exception), "--max must be a non-negative integer"
        )
        self.assertFalse(client.closed)


if __name__ == "__main__":
    unittest.main()
