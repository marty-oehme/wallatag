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
import os
import re
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from wallatag.config import Config, ConfigError, StoreConfig, WallabagConfig
from wallatag.tagger import KeywordTagger
from wallatag.wallabag import WallabagError, _should_fetch, _tag_labels


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
    """Minimal wallabag client fake (mirrors tests/test_auto.py)."""

    def __init__(self, entries=(), tags=(), fail_first=0, feed_error=None,
                 feed_fail_after=None):
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
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT entry_id, tag, action, source FROM decisions ORDER BY rowid"
        ).fetchall()


class PrefectFlowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import flows
        from prefect.cache_policies import NO_CACHE

        cls.flows = flows
        cls.NO_CACHE = NO_CACHE

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

    def run_batch(self, client, config=None, *, focus=None, max_articles=50,
                  store_path=None, spy_task=False):
        """Run wallatag_batch with the adapter seams patched to engine fakes.

        ``spy_task=True`` replaces the tag-article task with a spy over the
        real engine function (auto.process_entry) so call counts per
        candidate can be asserted; with ``spy_task=False`` the real Prefect
        task runs end to end.
        """
        config = config if config is not None else self.make_config(
            store_path=store_path, max_articles=max_articles
        )
        tagger = KeywordTagger(
            {},
            max_applied_tags=10,
            tag_policy="all",
            existing_tags=list(client.tags),
        )
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("flows.load_config", return_value=config))
            stack.enter_context(patch("flows.llm_env_from_block", return_value={}))
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
                # recorded AND still drives the real engine code.
                task_mock = stack.enter_context(
                    patch(
                        "flows.tag_article",
                        side_effect=self.flows.tag_article.fn,
                    )
                )
            with contextlib.redirect_stdout(out):
                result = self.flows.wallatag_batch.fn(
                    max_articles=max_articles, focus=focus
                )
        return result, out.getvalue(), task_mock

    # -- temporary keep: command builder (removal tracked in bug 262b48c) ----

    def test_command_build_defaults(self) -> None:
        self.assertEqual(
            self.flows.build_wallatag_command(50, None),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_full(self) -> None:
        self.assertEqual(
            self.flows.build_wallatag_command(10, "methods"),
            [
                "wallatag",
                "run",
                "--max",
                "10",
                "--focus",
                "methods",
            ],
        )

    def test_command_build_multiple_focus(self) -> None:
        self.assertEqual(
            self.flows.build_wallatag_command(10, "methods, languages"),
            [
                "wallatag",
                "run",
                "--max",
                "10",
                "--focus",
                "methods",
                "--focus",
                "languages",
            ],
        )

    def test_command_build_focus_empty_string(self) -> None:
        self.assertEqual(
            self.flows.build_wallatag_command(50, ""),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_focus_whitespace_only(self) -> None:
        self.assertEqual(
            self.flows.build_wallatag_command(50, "   "),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_focus_ragged(self) -> None:
        self.assertEqual(
            self.flows.build_wallatag_command(50, " methods , , languages "),
            [
                "wallatag",
                "run",
                "--max",
                "50",
                "--focus",
                "methods",
                "--focus",
                "languages",
            ],
        )

    def test_flow_defined(self) -> None:
        self.assertTrue(hasattr(self.flows, "wallatag_batch"))
        self.assertEqual(self.flows.wallatag_batch.name, "wallatag-batch")

    # -- env-merge helpers -------------------------------------------------

    def test_llm_env_from_block_fail_open(self) -> None:
        with patch(
            "flows.LLMCredentials.load",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.flows.llm_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_wallabag_env_from_block_fail_open(self) -> None:
        with patch(
            "flows.WallabagCredentials.load",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
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
        with patch(
            "flows.Variable.get",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
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

        with patch("flows.Variable.get", side_effect=fake_get), \
             contextlib.redirect_stdout(io.StringIO()) as out:
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
        with patch(
            "flows.Variable.get",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
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
            with self.subTest(raw=raw), patch(
                "flows.Variable.get", return_value=raw
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

    # -- temporary keep: subprocess runner (removal tracked in bug 262b48c) -

    def test_run_wallatag_streams_all_lines_to_flow_log(self) -> None:
        # The child prints a first line, sleeps, then prints a second line.
        # Both lines must land in the STREAMED print output (what becomes the
        # flow log via log_prints=True) AND in the returned captured output.
        # This asserts the output is captured on both channels; the live
        # streaming-before-exit guarantee itself is covered by
        # test_run_wallatag_burst_then_hang_streams_all_lines.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            output = self.flows._run_wallatag(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; print('first', flush=True);"
                    " time.sleep(1); print('second', flush=True)",
                ],
                env=dict(os.environ),
                timeout=30,
            )
        self.assertIn("first", out.getvalue())
        self.assertIn("second", out.getvalue())
        self.assertIn("first", output)
        self.assertIn("second", output)

    def test_run_wallatag_merges_stdout_and_stderr(self) -> None:
        # stderr=STDOUT merges both streams: both lines must be streamed and
        # captured in the returned string.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            output = self.flows._run_wallatag(
                [
                    sys.executable,
                    "-c",
                    "print('a'); import sys; print('b', file=sys.stderr)",
                ],
                env=dict(os.environ),
                timeout=30,
            )
        self.assertIn("a", out.getvalue())
        self.assertIn("b", out.getvalue())
        self.assertIn("a", output)
        self.assertIn("b", output)

    def test_run_wallatag_partial_line_then_hang_times_out(self) -> None:
        # A partial line (no trailing newline) followed by a hang must still
        # hit the deadline: the timeout bounds the read itself, not just idle
        # waits between complete lines. elapsed < 5 proves the old readline()
        # watchdog (which fired at t=10.1s for this scenario) is gone.
        start = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            self.flows._run_wallatag(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; sys.stdout.write('x');"
                    " sys.stdout.flush(); time.sleep(10)",
                ],
                env=dict(os.environ),
                timeout=1,
            )
        elapsed = time.monotonic() - start
        self.assertEqual(
            str(ctx.exception), "wallatag run timed out after 1s"
        )
        self.assertLess(elapsed, 5)

    def test_run_wallatag_burst_then_hang_streams_all_lines(self) -> None:
        # A burst of 50 lines followed by a hang must stream ALL 50 lines
        # before the timeout fires. Regression test for the buffered
        # read-ahead bug where only the first line was streamed and the
        # healthy process was killed at the deadline.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(RuntimeError) as ctx:
                self.flows._run_wallatag(
                    [
                        sys.executable,
                        "-c",
                        "import sys,time; [print(f'line{i}', flush=True)"
                        " for i in range(50)]; time.sleep(10)",
                    ],
                    env=dict(os.environ),
                    timeout=2,
                )
        self.assertEqual(
            str(ctx.exception), "wallatag run timed out after 2s"
        )
        streamed = out.getvalue()
        for i in range(50):
            self.assertIn(f"line{i}", streamed)

    def test_run_wallatag_nonzero_exit_raises_with_detail(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows._run_wallatag(
                    [
                        sys.executable,
                        "-c",
                        "print('boom'); import sys; sys.exit(3)",
                    ],
                    env=dict(os.environ),
                )
        message = str(ctx.exception)
        self.assertIn("exited with code 3", message)
        # The child's own output (the merged capture tail) must be carried in
        # the error detail, not just the exit code.
        self.assertIn("boom", message)

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
        with patch("flows.load_config", side_effect=fake_load_config), \
             patch(
                 "flows.llm_env_from_block",
                 return_value={"WALLATAG_AI_PROVIDER": "openai-compatible"},
             ), \
             patch(
                 "flows.wallabag_env_from_block",
                 return_value={"WALLATAG_URL": "https://block.example"},
             ), \
             patch.dict(
                 os.environ,
                 {
                     "WALLATAG_AI_PROVIDER": "ollama",
                     "WALLATAG_URL": "https://env.example",
                     "WALLATAG_DB": "/data/wallatag.db",
                 },
                 clear=True,
             ), \
             patch("flows.Variable.get", side_effect=fake_get), \
             patch(
                 "flows._build_tagger", return_value=(object(), None, None)
             ), \
             patch("flows.WallabagClient", return_value=client), \
             contextlib.redirect_stdout(io.StringIO()):
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

        with patch("flows.load_config", side_effect=fake_load_config), \
             patch(
                 "flows.LLMCredentials.load",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch(
                 "flows.WallabagCredentials.load",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch.dict(
                 os.environ, {"WALLATAG_IGNORE_TAGS": "fix"}, clear=True
             ), \
             patch("flows.Variable.get", return_value=None), \
             patch(
                 "flows._build_tagger", return_value=(object(), None, None)
             ), \
             patch("flows.WallabagClient", return_value=FakeClient()), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(captured["env"], {"WALLATAG_IGNORE_TAGS": "fix"})
        self.assertIn("falling back to config/env", out.getvalue())

    # -- flow: focus splitting ---------------------------------------------

    def test_split_focus_variants(self) -> None:
        self.assertIsNone(self.flows._split_focus(None))
        self.assertIsNone(self.flows._split_focus(""))
        self.assertIsNone(self.flows._split_focus("   "))
        self.assertIsNone(self.flows._split_focus(" , , "))
        self.assertEqual(self.flows._split_focus("methods"), ["methods"])
        self.assertEqual(
            self.flows._split_focus("methods, languages"),
            ["methods", "languages"],
        )
        self.assertEqual(
            self.flows._split_focus(" methods , , languages "),
            ["methods", "languages"],
        )

    def test_flow_focus_param_split_passed_to_apply_run_overrides(self) -> None:
        captured = {}

        def fake_apply(config, **kwargs):
            captured.update(kwargs)
            return config

        with patch("flows.load_config", return_value=self.make_config()), \
             patch("flows.apply_run_overrides", side_effect=fake_apply), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch("flows.wallabag_env_from_block", return_value={}), \
             patch.dict(os.environ, {}, clear=True), \
             patch("flows.Variable.get", return_value=None), \
             patch(
                 "flows._build_tagger", return_value=(object(), None, None)
             ), \
             patch("flows.WallabagClient", return_value=FakeClient()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(
                max_articles=50, focus="methods, languages"
            )
        self.assertEqual(captured["focus"], ["methods", "languages"])
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
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])])
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
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"]), (2, ["Pomodoro"])])
        self.assertEqual(
            result, "run: tagged 2 articles (2 tags applied), skipped 0"
        )
        self.assertIn(result, out)
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
        with patch("flows.WallabagClient") as client_mock, \
             patch("flows.load_config") as load_mock, \
             contextlib.redirect_stdout(io.StringIO()) as out:
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
