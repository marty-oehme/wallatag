"""Tests for flows (skipped when the prefect group isn't installed)."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import unittest
from unittest.mock import patch


class PrefectFlowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import flows

        cls.flows = flows

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

    def test_llm_env_from_block_fail_open(self) -> None:
        with patch(
            "flows.LLMCredentials.load",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.flows.llm_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_flow_env_overrides_block_env_for_same_var(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows.llm_env_from_block",
                 return_value={"WALLATAG_AI_PROVIDER": "openai-compatible"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_AI_PROVIDER": "ollama"},
                 clear=True,
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # When both the block and os.environ define the same var, os.environ
        # (container env) wins over the block value.
        self.assertEqual(
            captured["env"], {"WALLATAG_AI_PROVIDER": "ollama"}
        )

    def test_flow_block_fills_env_gaps(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows.llm_env_from_block",
                 return_value={
                     "WALLATAG_AI_PROVIDER": "openai-compatible",
                     "WALLATAG_AI_MODEL": "gpt-4o-mini",
                     "WALLATAG_AI_BASE_URL": "https://api.example.com/v1",
                 },
             ), \
             patch.dict(os.environ, {}, clear=True), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # Block values fill in WALLATAG_AI_* vars that os.environ does not set.
        self.assertEqual(
            captured["env"],
            {
                "WALLATAG_AI_PROVIDER": "openai-compatible",
                "WALLATAG_AI_MODEL": "gpt-4o-mini",
                "WALLATAG_AI_BASE_URL": "https://api.example.com/v1",
            },
        )

    def test_flow_passes_present_but_empty_env_var_through(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows.llm_env_from_block",
                 return_value={"WALLATAG_AI_MODEL": "gpt-4o-mini"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_AI_MODEL": ""},
                 clear=True,
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # A present-but-empty env var overrides the block with "" verbatim
        # (documents the merge: wallatag's config validation then rejects it,
        # which is why the docs say to `dokku config:unset` a block value
        # rather than `config:set` it to an empty string).
        self.assertEqual(captured["env"], {"WALLATAG_AI_MODEL": ""})

    def test_flow_falls_back_to_os_environ_when_block_unavailable(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows.LLMCredentials.load",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(captured["env"], {**os.environ})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_wallabag_env_from_block_fail_open(self) -> None:
        with patch(
            "flows.WallabagCredentials.load",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.flows.wallabag_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_flow_wallabag_block_fills_env_gaps(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch(
                 "flows.wallabag_env_from_block",
                 return_value={
                     "WALLATAG_URL": "https://wallabag.example.com",
                     "WALLATAG_CLIENT_ID": "client-id-123",
                     "WALLATAG_CLIENT_SECRET": "client-secret-456",
                     "WALLATAG_USERNAME": "reader@example.com",
                     "WALLATAG_PASSWORD": "hunter2",
                 },
             ), \
             patch.dict(os.environ, {}, clear=True), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # Block values fill in WALLATAG_* vars that os.environ does not set.
        self.assertEqual(
            captured["env"],
            {
                "WALLATAG_URL": "https://wallabag.example.com",
                "WALLATAG_CLIENT_ID": "client-id-123",
                "WALLATAG_CLIENT_SECRET": "client-secret-456",
                "WALLATAG_USERNAME": "reader@example.com",
                "WALLATAG_PASSWORD": "hunter2",
            },
        )

    def test_flow_wallabag_env_overrides_block_for_same_var(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch(
                 "flows.wallabag_env_from_block",
                 return_value={"WALLATAG_URL": "https://block.example"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_URL": "https://env.example"},
                 clear=True,
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # When both the block and os.environ define the same var, os.environ
        # (container env) wins over the block value.
        self.assertEqual(
            captured["env"], {"WALLATAG_URL": "https://env.example"}
        )

    def test_flow_passes_present_but_empty_wallabag_env_var_through(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch(
                 "flows.wallabag_env_from_block",
                 return_value={"WALLATAG_URL": "https://block.example"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_URL": ""},
                 clear=True,
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # A present-but-empty env var overrides the block with "" verbatim
        # (wallatag's config validation then rejects it, which is why the docs
        # say to `dokku config:unset` a block value rather than `config:set`
        # it to an empty string).
        self.assertEqual(captured["env"], {"WALLATAG_URL": ""})

    def test_flow_falls_back_to_os_environ_when_wallabag_block_unavailable(
        self,
    ) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows.WallabagCredentials.load",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(captured["env"], {**os.environ})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_missing_console_script_raises(self) -> None:
        with patch("flows.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                self.flows.wallatag_batch.fn(max_articles=50)

    def test_nonzero_exit_raises(self) -> None:
        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows._run_wallatag",
                 side_effect=RuntimeError("wallatag run exited with code 1: boom"),
             ), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.wallatag_batch.fn(max_articles=50)
        message = str(ctx.exception)
        self.assertIn("wallatag run exited with code 1", message)
        # The CLI output detail is carried in the raised message.
        self.assertIn("boom", message)

    def test_success_returns_output(self) -> None:
        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows._run_wallatag", return_value="wallatag run output"), \
             contextlib.redirect_stdout(io.StringIO()):
            result = self.flows.wallatag_batch.fn(max_articles=50)
        self.assertEqual(result, "wallatag run output")

    def test_timeout_raises(self) -> None:
        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "flows._run_wallatag",
                 side_effect=RuntimeError("wallatag run timed out after 1800s"),
             ), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as ctx:
                self.flows.wallatag_batch.fn(max_articles=50)
        self.assertIn("timed out after 1800s", str(ctx.exception))

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

    def test_flow_variable_env_beats_container_env(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        def fake_get(name, default=None):
            if name == "wallatag_ignore_tags":
                return "from-var"
            return default

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch("flows.wallabag_env_from_block", return_value={}), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_IGNORE_TAGS": "from-env"},
                 clear=True,
             ), \
             patch("flows.Variable.get", side_effect=fake_get), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        # Prefect Variables override container env vars (os.environ) for the
        # same setting; the winning value is what reaches the subprocess.
        self.assertEqual(captured["env"]["WALLATAG_IGNORE_TAGS"], "from-var")

    def test_flow_focus_groups_json_beats_per_group_env_var(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        def fake_get(name, default=None):
            if name == self.flows.FOCUS_GROUPS_VARIABLE:
                return {"methods": {"keywords": ["from-json"]}}
            return default

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch("flows.wallabag_env_from_block", return_value={}), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_FOCUS_methods_KEYWORDS": "from-env"},
                 clear=True,
             ), \
             patch("flows.Variable.get", side_effect=fake_get), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(
            captured["env"]["WALLATAG_FOCUS_methods_KEYWORDS"], "from-json"
        )

    def test_flow_focus_param_emits_repeated_flags(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["cmd"] = cmd
            captured["env"] = env
            return "wallatag run output"

        def fake_get(name, default=None):
            if name == "wallatag_tag_policy":
                return "prefer-existing"
            return default

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch("flows.wallabag_env_from_block", return_value={}), \
             patch.dict(os.environ, {}, clear=True), \
             patch("flows.Variable.get", side_effect=fake_get), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.flows.wallatag_batch.fn(
                max_articles=50, focus="methods, languages"
            )

        # The comma-separated focus param is split in the flow layer and each
        # name becomes its own --focus flag (order preserved).
        self.assertEqual(
            captured["cmd"],
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

    def test_flow_unchanged_when_variable_get_raises(self) -> None:
        captured = {}

        def fake_run(cmd, env, timeout=1800):
            captured["env"] = env
            return "wallatag run output"

        with patch("flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("flows.llm_env_from_block", return_value={}), \
             patch("flows.wallabag_env_from_block", return_value={}), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_IGNORE_TAGS": "from-env"},
                 clear=True,
             ), \
             patch(
                 "flows.Variable.get",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch("flows._run_wallatag", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.flows.wallatag_batch.fn(max_articles=50)

        # With Variable.get failing (no server), the flow behaves exactly as
        # before this feature: container env wins and warnings are printed.
        self.assertEqual(captured["env"], {"WALLATAG_IGNORE_TAGS": "from-env"})
        self.assertIn("prefect variables not available", out.getvalue())


if __name__ == "__main__":
    unittest.main()
