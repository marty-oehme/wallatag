"""Tests for prefect_flows (skipped when the prefect group isn't installed)."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import unittest
from unittest.mock import patch


class PrefectFlowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import prefect_flows

        cls.prefect_flows = prefect_flows

    def test_command_build_defaults(self) -> None:
        self.assertEqual(
            self.prefect_flows.build_wallatag_command(50, None),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_full(self) -> None:
        self.assertEqual(
            self.prefect_flows.build_wallatag_command(10, "methods"),
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
            self.prefect_flows.build_wallatag_command(10, "methods, languages"),
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
            self.prefect_flows.build_wallatag_command(50, ""),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_focus_whitespace_only(self) -> None:
        self.assertEqual(
            self.prefect_flows.build_wallatag_command(50, "   "),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_focus_ragged(self) -> None:
        self.assertEqual(
            self.prefect_flows.build_wallatag_command(50, " methods , , languages "),
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
        self.assertTrue(hasattr(self.prefect_flows, "wallatag_batch"))
        self.assertEqual(self.prefect_flows.wallatag_batch.name, "wallatag-batch")

    def test_llm_env_from_block_fail_open(self) -> None:
        with patch(
            "prefect_flows.LLMCredentials.load",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.prefect_flows.llm_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_flow_env_overrides_block_env_for_same_var(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "prefect_flows.llm_env_from_block",
                 return_value={"WALLATAG_AI_PROVIDER": "openai-compatible"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_AI_PROVIDER": "ollama"},
                 clear=True,
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        # When both the block and os.environ define the same var, os.environ
        # (container env) wins over the block value.
        self.assertEqual(
            captured["env"], {"WALLATAG_AI_PROVIDER": "ollama"}
        )

    def test_flow_block_fills_env_gaps(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "prefect_flows.llm_env_from_block",
                 return_value={
                     "WALLATAG_AI_PROVIDER": "openai-compatible",
                     "WALLATAG_AI_MODEL": "gpt-4o-mini",
                     "WALLATAG_AI_BASE_URL": "https://api.example.com/v1",
                 },
             ), \
             patch.dict(os.environ, {}, clear=True), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

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
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "prefect_flows.llm_env_from_block",
                 return_value={"WALLATAG_AI_MODEL": "gpt-4o-mini"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_AI_MODEL": ""},
                 clear=True,
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        # A present-but-empty env var overrides the block with "" verbatim
        # (documents the merge: wallatag's config validation then rejects it,
        # which is why the docs say to `dokku config:unset` a block value
        # rather than `config:set` it to an empty string).
        self.assertEqual(captured["env"], {"WALLATAG_AI_MODEL": ""})

    def test_flow_falls_back_to_os_environ_when_block_unavailable(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "prefect_flows.LLMCredentials.load",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(captured["env"], {**os.environ})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_wallabag_env_from_block_fail_open(self) -> None:
        with patch(
            "prefect_flows.WallabagCredentials.load",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.prefect_flows.wallabag_env_from_block()
        self.assertEqual(env, {})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_flow_wallabag_block_fills_env_gaps(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch(
                 "prefect_flows.wallabag_env_from_block",
                 return_value={
                     "WALLATAG_URL": "https://wallabag.example.com",
                     "WALLATAG_CLIENT_ID": "client-id-123",
                     "WALLATAG_CLIENT_SECRET": "client-secret-456",
                     "WALLATAG_USERNAME": "reader@example.com",
                     "WALLATAG_PASSWORD": "hunter2",
                 },
             ), \
             patch.dict(os.environ, {}, clear=True), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

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
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch(
                 "prefect_flows.wallabag_env_from_block",
                 return_value={"WALLATAG_URL": "https://block.example"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_URL": "https://env.example"},
                 clear=True,
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        # When both the block and os.environ define the same var, os.environ
        # (container env) wins over the block value.
        self.assertEqual(
            captured["env"], {"WALLATAG_URL": "https://env.example"}
        )

    def test_flow_passes_present_but_empty_wallabag_env_var_through(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch(
                 "prefect_flows.wallabag_env_from_block",
                 return_value={"WALLATAG_URL": "https://block.example"},
             ), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_URL": ""},
                 clear=True,
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        # A present-but-empty env var overrides the block with "" verbatim
        # (wallatag's config validation then rejects it, which is why the docs
        # say to `dokku config:unset` a block value rather than `config:set`
        # it to an empty string).
        self.assertEqual(captured["env"], {"WALLATAG_URL": ""})

    def test_flow_falls_back_to_os_environ_when_wallabag_block_unavailable(
        self,
    ) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch(
                 "prefect_flows.WallabagCredentials.load",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(captured["env"], {**os.environ})
        self.assertIn("falling back to config/env", out.getvalue())

    def test_missing_console_script_raises(self) -> None:
        with patch("prefect_flows.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                self.prefect_flows.wallatag_batch.fn(max_articles=50)

    def test_nonzero_exit_raises_with_stderr(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=2, stdout="", stderr="boom"
        )
        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.subprocess.run", return_value=completed):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.wallatag_batch.fn(max_articles=50)
        self.assertIn("boom", str(ctx.exception))

    def test_success_returns_stdout(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.subprocess.run", return_value=completed), \
             contextlib.redirect_stdout(io.StringIO()):
            result = self.prefect_flows.wallatag_batch.fn(max_articles=50)
        self.assertEqual(result, "tagged 3 articles")

    def test_timeout_raises(self) -> None:
        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="wallatag", timeout=1800)):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.wallatag_batch.fn(max_articles=50)
        self.assertIn("timed out after 1800s", str(ctx.exception))

    def test_variable_names_exact_no_secrets(self) -> None:
        self.assertEqual(
            set(self.prefect_flows.WALLATAG_VARIABLES),
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
        self.assertEqual(len(self.prefect_flows.WALLATAG_VARIABLES), 12)
        self.assertEqual(
            len(set(self.prefect_flows.WALLATAG_VARIABLES)),
            len(self.prefect_flows.WALLATAG_VARIABLES),
        )
        joined = "|".join(self.prefect_flows.WALLATAG_VARIABLES)
        for secret in ("CLIENT_SECRET", "PASSWORD", "API_KEY"):
            self.assertNotIn(secret, joined)

    def test_variable_env_normalizes_values(self) -> None:
        values = {
            "WALLATAG_TAG_POLICY": "all",
            "WALLATAG_MAX_APPLIED_TAGS": 7,
            "WALLATAG_AI_CONFIDENCE_THRESHOLD": 0.8,
            "WALLATAG_ENABLE_LLM": True,
            "WALLATAG_ENABLE_RULES": False,
            "WALLATAG_IGNORE_TAGS": "fix,_frigo",
        }

        def fake_get(name, default=None):
            return values.get(name, default)

        with patch("prefect_flows.Variable.get", side_effect=fake_get):
            env = self.prefect_flows.variable_env()
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
        with patch("prefect_flows.Variable.get", return_value=None):
            env = self.prefect_flows.variable_env()
        self.assertEqual(env, {})

    def test_variable_env_fail_open(self) -> None:
        with patch(
            "prefect_flows.Variable.get",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.prefect_flows.variable_env()
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

        with patch("prefect_flows.Variable.get", side_effect=fake_get), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.prefect_flows.variable_env()
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
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
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
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(env, {})

    def test_focus_groups_env_empty_fields_list_disables_group(self) -> None:
        raw = {"methods": {"keywords": ["howto"], "fields": []}}
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "howto",
                "WALLATAG_FOCUS_methods_FIELDS": "",
            },
        )

    def test_focus_groups_env_absent_fields_key_omits_fields_var(self) -> None:
        raw = {"methods": {"keywords": ["howto"]}}
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(env, {"WALLATAG_FOCUS_methods_KEYWORDS": "howto"})

    def test_focus_groups_env_parses_json_string(self) -> None:
        raw = '{"methods": {"keywords": ["a", "b"], "fields": []}}'
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "a,b",
                "WALLATAG_FOCUS_methods_FIELDS": "",
            },
        )

    def test_focus_groups_env_none_returns_empty(self) -> None:
        with patch("prefect_flows.Variable.get", return_value=None):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(env, {})

    def test_focus_groups_env_fail_open_on_read_error(self) -> None:
        with patch(
            "prefect_flows.Variable.get",
            side_effect=Exception("prefect server unreachable"),
        ), contextlib.redirect_stdout(io.StringIO()) as out:
            env = self.prefect_flows.focus_groups_env()
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
                "prefect_flows.Variable.get", return_value=raw
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self.prefect_flows.focus_groups_env()
                self.assertIn("WALLATAG_FOCUS_GROUPS", str(ctx.exception))

    def test_focus_groups_env_rejects_casefold_duplicate_names(self) -> None:
        raw = {
            "Methods": {"keywords": ["upper"]},
            "methods": {"keywords": ["lower"]},
        }
        with patch("prefect_flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn("WALLATAG_FOCUS_GROUPS", message)
        self.assertIn("'Methods'", message)
        self.assertIn("'methods'", message)
        self.assertIn("case-insensitively", message)

    def test_focus_groups_env_rejects_non_string_group_name(self) -> None:
        raw = {1: {"keywords": ["howto"]}}
        with patch("prefect_flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn("WALLATAG_FOCUS_GROUPS", message)
        self.assertIn("group names must be strings", message)

    def test_focus_groups_env_empty_string_value_fails_loud(self) -> None:
        with patch("prefect_flows.Variable.get", return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.focus_groups_env()
        self.assertIn("WALLATAG_FOCUS_GROUPS", str(ctx.exception))

    def test_focus_groups_env_sorted_order_determinism(self) -> None:
        raw = {
            "zeta": {"tags": ["z"]},
            "alpha": {"keywords": ["a"]},
            "methods": {"keywords": ["m"], "fields": ["title"]},
        }
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
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
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(
            env,
            {
                "WALLATAG_FOCUS_methods_KEYWORDS": "howto",
                "WALLATAG_FOCUS_methods_KEYWORDS_REGEX": "^how.?to,(?-i:GTD)",
            },
        )

    def test_focus_groups_env_omits_empty_keywords_regex(self) -> None:
        raw = {"methods": {"keywords_regex": []}}
        with patch("prefect_flows.Variable.get", return_value=raw):
            env = self.prefect_flows.focus_groups_env()
        self.assertEqual(env, {})

    def test_focus_groups_env_rejects_invalid_regex(self) -> None:
        raw = {"methods": {"keywords_regex": ["^["]}}
        with patch("prefect_flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn("WALLATAG_FOCUS_GROUPS", message)
        self.assertIn("'methods'", message)
        self.assertIn("'^['", message)

    def test_focus_groups_env_rejects_comma_containing_regex(self) -> None:
        # A comma-containing pattern is compilable but the comma-separated env
        # translation would silently split it: reject fail-loud here so the
        # user must use TOML keywords_regex instead.
        raw = {"methods": {"keywords_regex": ["^a,b$"]}}
        with patch("prefect_flows.Variable.get", return_value=raw):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.focus_groups_env()
        message = str(ctx.exception)
        self.assertIn("WALLATAG_FOCUS_GROUPS", message)
        self.assertIn("'methods'", message)
        self.assertIn("'^a,b$'", message)
        self.assertIn("comma", message)

    def test_flow_variable_env_beats_container_env(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        def fake_get(name, default=None):
            if name == "WALLATAG_IGNORE_TAGS":
                return "from-var"
            return default

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch("prefect_flows.wallabag_env_from_block", return_value={}), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_IGNORE_TAGS": "from-env"},
                 clear=True,
             ), \
             patch("prefect_flows.Variable.get", side_effect=fake_get), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        # Prefect Variables override container env vars (os.environ) for the
        # same setting; the winning value is what reaches the subprocess.
        self.assertEqual(captured["env"]["WALLATAG_IGNORE_TAGS"], "from-var")

    def test_flow_focus_groups_json_beats_per_group_env_var(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        def fake_get(name, default=None):
            if name == "WALLATAG_FOCUS_GROUPS":
                return {"methods": {"keywords": ["from-json"]}}
            return default

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch("prefect_flows.wallabag_env_from_block", return_value={}), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_FOCUS_methods_KEYWORDS": "from-env"},
                 clear=True,
             ), \
             patch("prefect_flows.Variable.get", side_effect=fake_get), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        self.assertEqual(
            captured["env"]["WALLATAG_FOCUS_methods_KEYWORDS"], "from-json"
        )

    def test_flow_focus_param_emits_repeated_flags(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["args"] = args[0]
            captured["env"] = kwargs["env"]
            return completed

        def fake_get(name, default=None):
            if name == "WALLATAG_TAG_POLICY":
                return "prefer-existing"
            return default

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch("prefect_flows.wallabag_env_from_block", return_value={}), \
             patch.dict(os.environ, {}, clear=True), \
             patch("prefect_flows.Variable.get", side_effect=fake_get), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.prefect_flows.wallatag_batch.fn(
                max_articles=50, focus="methods, languages"
            )

        # The comma-separated focus param is split in the flow layer and each
        # name becomes its own --focus flag (order preserved).
        self.assertEqual(
            captured["args"],
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
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return completed

        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.llm_env_from_block", return_value={}), \
             patch("prefect_flows.wallabag_env_from_block", return_value={}), \
             patch.dict(
                 os.environ,
                 {"WALLATAG_IGNORE_TAGS": "from-env"},
                 clear=True,
             ), \
             patch(
                 "prefect_flows.Variable.get",
                 side_effect=Exception("prefect server unreachable"),
             ), \
             patch("prefect_flows.subprocess.run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.prefect_flows.wallatag_batch.fn(max_articles=50)

        # With Variable.get failing (no server), the flow behaves exactly as
        # before this feature: container env wins and warnings are printed.
        self.assertEqual(captured["env"], {"WALLATAG_IGNORE_TAGS": "from-env"})
        self.assertIn("prefect variables not available", out.getvalue())


if __name__ == "__main__":
    unittest.main()
