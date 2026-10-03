"""Tests for wallatag.cli: parser, flag overrides, stubs and status output."""

import argparse
import contextlib
import dataclasses
import io
import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wallatag.cli import (
    _build_tagger,
    _reset_seen,
    apply_flag_overrides,
    build_parser,
    cmd_config_show,
    cmd_run,
    main,
)
from wallatag.config import (
    AiConfig,
    Config,
    ConfigError,
    FocusGroup,
    TaggerConfig,
    WallabagConfig,
    load_config,
)
from wallatag.tagger import KeywordTagger, LLMTagger
from wallatag.wallabag import _should_fetch, _tag_labels


def _args(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace shaped like a parsed subcommand invocation."""
    defaults = {
        "config": None,
        "max": None,
        "focus": None,
        "tag_policy": None,
        "no_history": False,
        "no_apply": False,
        "reset_seen": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class HelpTest(unittest.TestCase):
    """(a) --help exits 0 and mentions all three subcommands."""

    def test_help(self):
        parser = build_parser()
        out = io.StringIO()
        with (
            self.assertRaises(SystemExit) as ctx,
            contextlib.redirect_stdout(out),
        ):
            parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        for sub in ("manual", "run", "config", "status"):
            self.assertIn(sub, text)


class VersionTest(unittest.TestCase):
    """(b) --version exits 0 and prints the version."""

    def test_version(self):
        out = io.StringIO()
        with (
            self.assertRaises(SystemExit) as ctx,
            contextlib.redirect_stdout(out),
        ):
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("0.1.0", out.getvalue())


class RunStubTest(unittest.TestCase):
    """(c) run is headless batch tagging; an empty config fails cleanly."""

    def test_run_without_config_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "wallatag.toml"
            config_path.write_text("", encoding="utf-8")
            out = io.StringIO()
            err = io.StringIO()
            try:
                with (
                    contextlib.redirect_stdout(out),
                    contextlib.redirect_stderr(err),
                ):
                    code = main(
                        [
                            "run",
                            "--config",
                            str(config_path),
                            "--max",
                            "10",
                            "--tag-policy",
                            "all",
                            "--no-history",
                            "--no-apply",
                            "--verbose",
                        ]
                    )
            finally:
                # cmd_run configures root logging against the redirected
                # stdout; drop those handlers so other tests are unaffected.
                for handler in list(logging.getLogger().handlers):
                    logging.getLogger().removeHandler(handler)
        self.assertEqual(code, 2)
        self.assertIn("url is not configured", err.getvalue())


class FlagOverrideTest(unittest.TestCase):
    """(d) CLI flags win over TOML values via apply_flag_overrides."""

    def _load(self, tmp: str) -> Config:
        path = Path(tmp) / "wallatag.toml"
        path.write_text(
            '[tagger]\ntag_policy = "prefer-existing"\n', encoding="utf-8"
        )
        return load_config(config_path=str(path), env={})

    def test_flag_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._load(tmp)
            overridden = apply_flag_overrides(config, _args(tag_policy="all"))
        self.assertEqual(overridden.tagger.tag_policy, "all")

    def test_default_preserved_when_flag_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._load(tmp)
            overridden = apply_flag_overrides(config, _args())
        self.assertEqual(overridden.tagger.tag_policy, "prefer-existing")

    def test_no_history_and_verbose(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text('[store]\npath = "/data/x.db"\n', encoding="utf-8")
            config = load_config(config_path=str(path), env={})
            overridden = apply_flag_overrides(
                config, _args(no_history=True, verbose=True)
            )
        self.assertIsNone(overridden.store.path)
        self.assertTrue(overridden.verbose)


class MaxArticlesTest(unittest.TestCase):
    """--max is a runtime run limit; it must not touch max_applied_tags."""

    def test_max_does_not_alter_max_applied_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                "[tagger]\nmax_applied_tags = 9\n", encoding="utf-8"
            )
            config = load_config(config_path=str(path), env={})
            overridden = apply_flag_overrides(config, _args(max=100))
            self.assertEqual(overridden.tagger.max_applied_tags, 9)
            self.assertEqual(overridden.max_articles, 100)

    def test_status_shows_toml_max_applied_tags_and_run_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                "[tagger]\nmax_applied_tags = 9\n", encoding="utf-8"
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path), "--max", "100"])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("max_applied_tags=9", text)
        self.assertNotIn("max_applied_tags=100", text)
        self.assertIn("run limit: 100 articles", text)

    def test_status_shows_max_proposals_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                "[tagger]\nmax_applied_tags = 9\n\n[ai]\nmax_proposals = 4\n",
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("max_applied_tags=9", text)
        self.assertIn("max_proposals=4", text)

    def test_status_omits_max_proposals_when_unset(self):
        # No [ai] max_proposals: the tagger line stays unchanged (the keyword
        # tagger has no such knob).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                "[tagger]\nmax_applied_tags = 9\n", encoding="utf-8"
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("max_applied_tags=9", text)
        self.assertNotIn("max_proposals", text)


class FocusTest(unittest.TestCase):
    """--focus validates the name and narrows the active focus groups."""

    FOCUS_TOML = (
        "[focus.methods]\n"
        'keywords = ["pomodoro"]\n'
        'tags = ["productivity"]\n'
        "\n"
        "[focus.languages]\n"
        'keywords = ["python"]\n'
        'tags = ["programming"]\n'
    )

    def _load_with_focus_groups(self, tmp: str) -> Config:
        path = Path(tmp) / "wallatag.toml"
        path.write_text(self.FOCUS_TOML, encoding="utf-8")
        return load_config(config_path=str(path), env={})

    def test_unknown_focus_exits_2_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(self.FOCUS_TOML, encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(
                    ["status", "--config", str(path), "--focus", "bogus"]
                )
            message = err.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("unknown focus group 'bogus'", message)
        self.assertIn("methods", message)
        self.assertIn("languages", message)

    def test_valid_focus_narrows_to_single_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._load_with_focus_groups(tmp)
            narrowed = apply_flag_overrides(config, _args(focus="methods"))
        self.assertEqual(list(narrowed.tagger.focus_groups), ["methods"])
        self.assertEqual(
            narrowed.tagger.focus_groups["methods"].keywords, ("pomodoro",)
        )
        self.assertEqual(
            narrowed.tagger.focus_groups["methods"].tags, ("productivity",)
        )

    def test_valid_focus_via_main_returns_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(self.FOCUS_TOML, encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(
                    ["status", "--config", str(path), "--focus", "methods"]
                )
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("focus groups: methods", text)

    def test_multiple_focus_narrows_to_both_groups_in_flag_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._load_with_focus_groups(tmp)
            narrowed = apply_flag_overrides(
                config, _args(focus=["languages", "methods"])
            )
        self.assertEqual(
            list(narrowed.tagger.focus_groups), ["languages", "methods"]
        )
        self.assertEqual(
            narrowed.tagger.focus_groups["languages"].keywords, ("python",)
        )
        self.assertEqual(
            narrowed.tagger.focus_groups["languages"].tags, ("programming",)
        )
        self.assertEqual(
            narrowed.tagger.focus_groups["methods"].keywords, ("pomodoro",)
        )
        self.assertEqual(
            narrowed.tagger.focus_groups["methods"].tags, ("productivity",)
        )

    def test_unknown_focus_among_valid_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(self.FOCUS_TOML, encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(
                    [
                        "status",
                        "--config",
                        str(path),
                        "--focus",
                        "methods",
                        "--focus",
                        "bogus",
                    ]
                )
            message = err.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("unknown focus group 'bogus'", message)
        self.assertIn("methods", message)
        self.assertIn("languages", message)

    def test_valid_multiple_focus_via_main_returns_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(self.FOCUS_TOML, encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(
                    [
                        "status",
                        "--config",
                        str(path),
                        "--focus",
                        "methods",
                        "--focus",
                        "languages",
                    ]
                )
            text = out.getvalue()
        self.assertEqual(code, 0)
        # status prints one line "focus groups: methods (1 keywords, 1 tags),
        # languages (1 keywords, 1 tags)"; both selected names must appear.
        self.assertIn("focus groups: methods", text)
        self.assertIn("languages", text)

    def test_repeated_focus_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._load_with_focus_groups(tmp)
            narrowed = apply_flag_overrides(
                config, _args(focus=["methods", "methods"])
            )
        self.assertEqual(list(narrowed.tagger.focus_groups), ["methods"])


class MaxNegativeTest(unittest.TestCase):
    """--max -1 must fail loudly with a non-zero exit."""

    def test_negative_max_exits_2_via_parser(self):
        # argparse prints the usage/error banner to stderr itself; capture it
        # so the expected failure does not pollute the CI log.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["run", "--max", "-1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_negative_max_raises_configerror_in_overrides(self):
        config = Config()
        with self.assertRaises(ConfigError):
            apply_flag_overrides(config, _args(max=-1))


class StatusOutputTest(unittest.TestCase):
    """(e) status shows config but never the client_secret."""

    def test_status_never_leaks_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[wallabag]\n"
                    'url = "https://wallabag.example.com"\n'
                    'client_id = "cid"\n'
                    'client_secret = "super-secret-value"\n'
                    'username = "alice"\n'
                    'password = "super-secret-password"\n'
                ),
                encoding="utf-8",
            )
            config = load_config(config_path=str(path), env={})
            args = argparse.Namespace(
                config=str(path),
                max=None,
                focus=None,
                tag_policy=None,
                no_history=False,
                no_apply=False,
                verbose=False,
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cmd_config_show(config, args)
            text = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("wallatag", text)
        self.assertIn("https://wallabag.example.com", text)
        # The username is shown (non-secret)...
        self.assertIn("auth: username=alice", text)
        # ...but neither the client_secret nor the password ever appear.
        self.assertNotIn("super-secret-value", text)
        self.assertNotIn("super-secret-password", text)
        self.assertNotIn("cid", text)

    def test_status_username_not_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                '[wallabag]\nurl = "https://wallabag.example.com"\n',
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("wallabag: https://wallabag.example.com", text)
        self.assertIn("auth: username=(not configured)", text)

    def test_status_via_main_shows_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                '[wallabag]\nurl = "https://wallabag.example.com"\n',
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("wallabag: https://wallabag.example.com", text)
        self.assertIn("store: history-less", text)


class ConfigCommandTest(unittest.TestCase):
    """`wallatag config`/`config show` print the config; `status` aliases it."""

    def test_config_bare_shows_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                '[wallabag]\nurl = "https://wallabag.example.com"\n',
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["config", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("wallabag: https://wallabag.example.com", text)
        self.assertIn("store: history-less", text)

    def test_config_show_matches_bare_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                '[wallabag]\nurl = "https://wallabag.example.com"\n',
                encoding="utf-8",
            )
            out_bare = io.StringIO()
            out_show = io.StringIO()
            with contextlib.redirect_stdout(out_bare):
                code_bare = main(["config", "--config", str(path)])
            with contextlib.redirect_stdout(out_show):
                code_show = main(["config", "show", "--config", str(path)])
        self.assertEqual(code_bare, 0)
        self.assertEqual(code_show, 0)
        self.assertEqual(out_bare.getvalue(), out_show.getvalue())

    def test_status_alias_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                '[wallabag]\nurl = "https://wallabag.example.com"\n',
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("wallabag: https://wallabag.example.com", out.getvalue())

    def test_config_help_lists_subcommands(self):
        parser = build_parser()
        out = io.StringIO()
        with (
            self.assertRaises(SystemExit) as ctx,
            contextlib.redirect_stdout(out),
        ):
            parser.parse_args(["config", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        self.assertIn("show", text)


class FakeClient:
    """Minimal wallabag client fake for the cmd_run integration test."""

    def __init__(self, entries=(), tags=()):
        self.entries = list(entries)
        self.tags = list(tags)
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
        self.add_calls.append((entry_id, sorted(tags)))

    def close(self):
        self.closed = True


def _entry(eid, title):
    return {
        "id": eid,
        "title": title,
        "url": "https://example.com/x",
        "domain_name": "example.com",
        "content": "",
        "reading_time": 5,
        "language": "en",
        "tags": [],
    }


def _ai_cfg(
    provider="ollama",
    base_url="http://localhost:11434",
    model="qwen2.5:3b",
    threshold=0.7,
):
    # The [ai] trio AND the opt-in enable_llm switch: this helper builds the
    # fully LLM-enabled config (provider trio alone is no longer enough).
    return dataclasses.replace(
        Config(),
        wallabag=WallabagConfig(
            url="https://wallabag.example.com",
            client_id="cid",
            client_secret="secret",
            username="alice",
            password="wonderland",
        ),
        ai=AiConfig(
            provider=provider,
            base_url=base_url,
            model=model,
            confidence_threshold=threshold,
        ),
        tagger=TaggerConfig(enable_llm=True),
    )


def config_focus_group():
    return FocusGroup(keywords=("python",), tags=("programming",))


class BuildTaggerTest(unittest.TestCase):
    """_build_tagger picks LLMTagger iff [ai] provider AND enable_llm are set."""

    def test_ai_configured_selects_llm_tagger(self):
        config = dataclasses.replace(
            _ai_cfg(threshold=0.8),
            tagger=TaggerConfig(
                max_applied_tags=3,
                tag_policy="only-existing",
                focus_groups={"a": config_focus_group()},
                enable_llm=True,
            ),
        )
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(
                config, ["python", "rust"]
            )

        client_cls.assert_called_once_with(
            "ollama", "http://localhost:11434", "qwen2.5:3b", api_key=""
        )
        self.assertIs(llm_client, client_cls.return_value)
        self.assertIsInstance(tagger, LLMTagger)
        self.assertIsNone(fallback)
        self.assertEqual(tagger.confidence_threshold, 0.8)
        self.assertEqual(tagger.max_applied_tags, 3)
        self.assertEqual(tagger.tag_policy, "only-existing")
        self.assertEqual(tagger.existing_tags, ["python", "rust"])

    def test_ai_configured_passes_api_key_to_client(self):
        config = dataclasses.replace(
            _ai_cfg(
                provider="openai-compatible",
                base_url="https://api.example.com/v1",
                model="gpt-4o-mini",
            ),
            ai=AiConfig(
                provider="openai-compatible",
                base_url="https://api.example.com/v1",
                model="gpt-4o-mini",
                api_key="sk-test-key",
            ),
        )
        with patch("wallatag.cli.LLMClient") as client_cls:
            _build_tagger(config, ["python"])
        client_cls.assert_called_once_with(
            "openai-compatible",
            "https://api.example.com/v1",
            "gpt-4o-mini",
            api_key="sk-test-key",
        )

    def test_no_ai_selects_keyword_tagger(self):
        config = Config()
        tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        self.assertIsNone(fallback)

    def test_ai_use_focus_groups_false_wired_to_tagger(self):
        # TOML `use_focus_groups = false` flows through load_config ->
        # _build_tagger into LLMTagger.use_focus_groups.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                    "use_focus_groups = false\n"
                    "\n"
                    "[tagger]\n"
                    "enable_llm = true\n"
                ),
                encoding="utf-8",
            )
            config = load_config(config_path=str(path), env={})
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, LLMTagger)
        self.assertFalse(tagger.use_focus_groups)
        self.assertIs(llm_client, client_cls.return_value)

    def test_ai_use_focus_groups_default_true_wired_to_tagger(self):
        config = _ai_cfg()
        with patch("wallatag.cli.LLMClient"):
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, LLMTagger)
        self.assertTrue(tagger.use_focus_groups)

    def test_ai_provider_empty_with_partial_ai_still_keyword(self):
        # Threshold-only [ai]: provider is empty -> keyword tagger, no client.
        config = dataclasses.replace(
            Config(), ai=AiConfig(confidence_threshold=0.9)
        )
        tagger, llm_client, fallback = _build_tagger(config, [])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)

    def test_keyword_tagger_receives_vocabulary_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                '[vocabulary]\nfields = ["title", "url"]\n', encoding="utf-8"
            )
            config = load_config(config_path=str(path), env={})
            tagger, llm_client, fallback = _build_tagger(config, [])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        self.assertEqual(tagger.vocabulary_fields, ("title", "url"))

    def test_keyword_tagger_default_vocabulary_fields_all_four(self):
        config = Config()
        tagger, _, fallback = _build_tagger(config, [])
        self.assertEqual(
            tagger.vocabulary_fields,
            ("title", "url", "domain_name", "content"),
        )

    def test_empty_fields_disable_both_sources_end_to_end(self):
        # End-to-end disable: a TOML with `fields = []` on BOTH the
        # vocabulary and a focus group flows through load_config ->
        # _build_tagger -> suggest(). The entry content contains the
        # existing-tag label AND the keyword (both would match under the
        # all-four-fields default), yet neither source fires: suggest() is [].
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[vocabulary]\n"
                    "fields = []\n"
                    "\n"
                    "[focus.x]\n"
                    'keywords = ["k"]\n'
                    'tags = ["t"]\n'
                    "fields = []\n"
                ),
                encoding="utf-8",
            )
            config = load_config(config_path=str(path), env={})
            tagger, llm_client, fallback = _build_tagger(config, ["label"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        self.assertIsNone(fallback)
        self.assertEqual(tagger.vocabulary_fields, ())
        self.assertEqual(tagger.focus_groups["x"].fields, ())
        entry = {
            "title": "unrelated",
            "url": "https://example.com/unrelated",
            "domain_name": "example.com",
            "content": "label k",
        }
        self.assertEqual(tagger.suggest(entry), [])

    # --- per-source enable switches (issue ba1332b) ---

    def _load_toml(self, text: str) -> Config:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(config_path=str(path), env={})

    AI_TOML = (
        "[ai]\n"
        'provider = "ollama"\n'
        'base_url = "http://localhost:11434"\n'
        'model = "qwen2.5:3b"\n'
    )

    def test_ai_plus_enable_llm_true_selects_llm_tagger(self):
        # [ai] fully configured AND enable_llm = true -> LLMTagger.
        config = self._load_toml(
            self.AI_TOML + "\n[tagger]\nenable_llm = true\n"
        )
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, LLMTagger)
        self.assertIs(llm_client, client_cls.return_value)

    def test_ai_without_enable_llm_selects_keyword_tagger(self):
        # NEW opt-in behavior: [ai] alone (enable_llm absent, default false)
        # no longer activates the LLM tagger.
        config = self._load_toml(self.AI_TOML)
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        client_cls.assert_not_called()

    def test_ai_with_enable_llm_false_selects_keyword_tagger(self):
        config = self._load_toml(
            self.AI_TOML + "\n[tagger]\nenable_llm = false\n"
        )
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        client_cls.assert_not_called()

    def test_enable_llm_true_without_ai_selects_keyword_tagger(self):
        # The switch alone cannot enable the LLM: the provider trio is still
        # required.
        config = self._load_toml("[tagger]\nenable_llm = true\n")
        tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        self.assertIsNone(fallback)

    def test_keyword_tagger_receives_enable_switches_from_config(self):
        config = self._load_toml(
            "[tagger]\nenable_vocabulary = false\nenable_rules = false\n"
        )
        tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        self.assertIsNone(fallback)
        self.assertFalse(tagger.enable_vocabulary)
        self.assertFalse(tagger.enable_rules)
        # The switch flags land on the tagger AND actually gate suggest().
        self.assertEqual(
            tagger.suggest({"title": "python", "content": "x"}), []
        )

    def test_keyword_tagger_default_switches_enabled(self):
        config = Config()
        tagger, _, fallback = _build_tagger(config, ["python"])
        self.assertTrue(tagger.enable_vocabulary)
        self.assertTrue(tagger.enable_rules)

    # --- [ai] fallback_on_fail (issue d5535d0) ---

    def test_fallback_on_fail_builds_keyword_fallback(self):
        # [ai] trio + enable_llm + fallback_on_fail -> 3-tuple whose third
        # element is a KeywordTagger mirroring the keyword-mode construction.
        config = dataclasses.replace(
            _ai_cfg(threshold=0.8),
            tagger=TaggerConfig(
                max_applied_tags=3,
                tag_policy="only-existing",
                focus_groups={"a": config_focus_group()},
                enable_llm=True,
            ),
            ai=AiConfig(
                provider="ollama",
                base_url="http://localhost:11434",
                model="qwen2.5:3b",
                confidence_threshold=0.8,
                fallback_on_fail=True,
            ),
        )
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(
                config, ["python", "rust"]
            )

        self.assertIsInstance(tagger, LLMTagger)
        self.assertIs(llm_client, client_cls.return_value)
        self.assertIsInstance(fallback, KeywordTagger)
        # Same construction params as a normal keyword-mode run: same focus
        # groups, max_applied_tags, tag_policy, existing tags, and the
        # enable_vocabulary/enable_rules switches.
        self.assertEqual(fallback.focus_groups, tagger.focus_groups)
        self.assertEqual(fallback.max_applied_tags, 3)
        self.assertEqual(fallback.tag_policy, "only-existing")
        self.assertEqual(fallback.existing_tags, ["python", "rust"])
        self.assertTrue(fallback.enable_vocabulary)
        self.assertTrue(fallback.enable_rules)

    def test_fallback_on_fail_default_false_no_fallback(self):
        # Same config minus fallback_on_fail (default false): 3rd element None.
        config = _ai_cfg()
        with patch("wallatag.cli.LLMClient") as client_cls:
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, LLMTagger)
        self.assertIs(llm_client, client_cls.return_value)
        self.assertIsNone(fallback)

    def test_fallback_on_fail_explicit_false_no_fallback(self):
        config = dataclasses.replace(
            _ai_cfg(),
            ai=dataclasses.replace(_ai_cfg().ai, fallback_on_fail=False),
        )
        with patch("wallatag.cli.LLMClient"):
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, LLMTagger)
        self.assertIsNone(fallback)

    def test_fallback_on_fail_without_enable_llm_ignored(self):
        # No enable_llm -> KeywordTagger as the tagger and NO fallback at all
        # (a keyword fallback behind a keyword tagger would be a no-op).
        config = dataclasses.replace(
            Config(),
            ai=AiConfig(
                provider="ollama",
                base_url="http://localhost:11434",
                model="qwen2.5:3b",
                fallback_on_fail=True,
            ),
        )
        tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, KeywordTagger)
        self.assertIsNone(llm_client)
        self.assertIsNone(fallback)

    def test_fallback_respects_enable_switches_and_vocabulary_fields(self):
        # The fallback KeywordTagger receives enable_vocabulary/enable_rules
        # and vocabulary_fields from config, exactly like the non-LLM branch.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                    "fallback_on_fail = true\n"
                    "\n"
                    "[tagger]\n"
                    "enable_llm = true\n"
                    "enable_vocabulary = false\n"
                    "enable_rules = false\n"
                    "\n"
                    "[vocabulary]\n"
                    'fields = ["title"]\n'
                ),
                encoding="utf-8",
            )
            config = load_config(config_path=str(path), env={})
        with patch("wallatag.cli.LLMClient"):
            tagger, llm_client, fallback = _build_tagger(config, ["python"])
        self.assertIsInstance(tagger, LLMTagger)
        self.assertIsNotNone(llm_client)
        self.assertIsInstance(fallback, KeywordTagger)
        self.assertFalse(fallback.enable_vocabulary)
        self.assertFalse(fallback.enable_rules)
        self.assertEqual(fallback.vocabulary_fields, ("title",))
        # The switches actually gate the fallback's suggest(): nothing matches.
        self.assertEqual(
            fallback.suggest({"title": "python", "content": "x"}), []
        )


class CmdRunAiSelectionTest(unittest.TestCase):
    """cmd_run wires the LLM tagger when [ai] is configured (and not otherwise)."""

    def setUp(self):
        # cmd_run configures root logging against the redirected stdout; drop
        # those handlers so other tests are unaffected.
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    def test_cmd_run_with_ai_uses_llm_tagger(self):
        client = FakeClient(
            entries=[_entry(1, "python article")], tags=["python"]
        )
        created = []

        class FakeLLM:
            def __init__(self, provider, base_url, model, api_key=""):
                self.provider = provider
                self.base_url = base_url
                self.model = model
                self.api_key = api_key
                self.closed = False
                created.append(self)

            def complete(self, system_prompt, user_prompt):
                return '[{"tag": "python", "confidence": 0.95}]'

            def close(self):
                self.closed = True

        out = io.StringIO()
        err = io.StringIO()
        with (
            patch("wallatag.cli.WallabagClient", return_value=client),
            patch("wallatag.cli.LLMClient", side_effect=FakeLLM),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = cmd_run(_ai_cfg(), _args())

        self.assertEqual(code, 0)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].provider, "ollama")
        self.assertEqual(created[0].base_url, "http://localhost:11434")
        self.assertEqual(created[0].model, "qwen2.5:3b")
        self.assertTrue(created[0].closed)  # closed in cmd_run's finally
        # The LLM tagger's suggestion was applied: proves LLMTagger was used.
        self.assertEqual(client.add_calls, [(1, ["python"])])

    def test_cmd_run_without_ai_skips_llm_client(self):
        client = FakeClient(
            entries=[_entry(1, "pomodoro focus")], tags=["Pomodoro"]
        )
        out = io.StringIO()
        err = io.StringIO()
        cfg = dataclasses.replace(
            Config(),
            wallabag=WallabagConfig(
                url="https://wallabag.example.com",
                client_id="cid",
                client_secret="secret",
                username="alice",
                password="wonderland",
            ),
        )
        with (
            patch("wallatag.cli.WallabagClient", return_value=client),
            patch("wallatag.cli.LLMClient") as llm_cls,
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = cmd_run(cfg, _args())

        self.assertEqual(code, 0)
        llm_cls.assert_not_called()
        # KeywordTagger's vocabulary suggestion was applied.
        self.assertEqual(client.add_calls, [(1, ["Pomodoro"])])


class StatusAiLineTest(unittest.TestCase):
    def test_status_shows_ai_line_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                    "confidence_threshold = 0.8\n"
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("ai: provider=ollama model=qwen2.5:3b", text)
        self.assertIn("(confidence_threshold=0.8)", text)

    def test_status_marks_llm_inactive_when_enable_llm_absent(self):
        # [ai] trio configured but enable_llm absent (default false): status
        # must make explicit that the LLM tagger is NOT active.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("(llm enabled=no; set [tagger] enable_llm = true)", text)

    def test_status_marks_fallback_when_fallback_on_fail(self):
        # [ai] fallback_on_fail=true with the LLM enabled: the ai line gains a
        # compact "(llm fallback on)" marker appended after the confidence
        # marker; the rest of the enabled-line format is unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                    "confidence_threshold = 0.8\n"
                    "fallback_on_fail = true\n"
                    "\n"
                    "[tagger]\n"
                    "enable_llm = true\n"
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn(
            "ai: provider=ollama model=qwen2.5:3b "
            "(confidence_threshold=0.8) (llm fallback on)",
            text,
        )

    def test_status_marks_fallback_even_when_llm_disabled(self):
        # fallback_on_fail=true with enable_llm absent (default false): the
        # fallback marker is appended after the enable_llm note.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                    "fallback_on_fail = true\n"
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn(
            "(llm enabled=no; set [tagger] enable_llm = true) (llm fallback on)",
            text,
        )

    def test_status_no_fallback_marker_when_false(self):
        # fallback_on_fail absent (default false): the enabled ai line is
        # byte-identical to the pre-change output (no fallback marker).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "ollama"\n'
                    'base_url = "http://localhost:11434"\n'
                    'model = "qwen2.5:3b"\n'
                    "confidence_threshold = 0.8\n"
                    "\n"
                    "[tagger]\n"
                    "enable_llm = true\n"
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn(
            "ai: provider=ollama model=qwen2.5:3b "
            "(confidence_threshold=0.8)\n",
            text,
        )
        self.assertNotIn("llm fallback", text)

    def test_status_ai_not_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text("", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("ai: not configured", text)

    def test_status_never_leaks_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text(
                (
                    "[ai]\n"
                    'provider = "openai-compatible"\n'
                    'base_url = "https://api.example.com/v1"\n'
                    'model = "gpt-4o-mini"\n'
                    'api_key = "sk-super-secret-api-key"\n'
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path)])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("ai: provider=openai-compatible", text)
        self.assertNotIn("sk-super-secret-api-key", text)

    def test_never_leaks_api_key_via_cmd_config_show(self):
        # Direct cmd_config_show with an AiConfig that has api_key set.
        config = dataclasses.replace(
            Config(),
            ai=AiConfig(
                provider="openai-compatible",
                base_url="https://api.example.com/v1",
                model="gpt-4o-mini",
                api_key="sk-direct-secret",
            ),
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cmd_config_show(config, _args())
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("ai: provider=openai-compatible", text)
        self.assertNotIn("sk-direct-secret", text)


class ResetSeenTest(unittest.TestCase):
    """`--reset-seen` clears cooldowns; dry run only reports the count."""

    def test_dry_run_reports_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            from wallatag.store import Store

            store = Store(Path(tmp) / "s.db")
            self.addCleanup(store.close)
            store.mark_seen(1)
            store.mark_seen(2)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                _reset_seen(store, dry_run=True)
            self.assertIn("would reset 2 cooldown entries", out.getvalue())
            # Nothing changed: both rows are still in cooldown.
            self.assertEqual(store.count_cooldowns(), 2)

    def test_reset_clears_cooldowns_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            from wallatag.store import Store

            store = Store(Path(tmp) / "s.db")
            self.addCleanup(store.close)
            store.mark_seen(1)
            store.claim(2)
            store.record_decision(1, "python", "accept", "rules")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                _reset_seen(store, dry_run=False)
            self.assertIn("reset 1 cooldown entries", out.getvalue())
            self.assertEqual(store.count_cooldowns(), 0)
            # The in-progress lease and the decision history survive.
            self.assertTrue(store.is_seen(2))
            with contextlib.closing(
                __import__("sqlite3").connect(store.path)
            ) as conn:
                decisions = conn.execute(
                    "SELECT COUNT(*) FROM decisions"
                ).fetchone()[0]
            self.assertEqual(decisions, 1)

    def test_flag_is_parsed_for_manual_and_run(self):
        parser = build_parser()
        for command in ("manual", "run"):
            args = parser.parse_args([command, "--reset-seen"])
            self.assertTrue(args.reset_seen)


if __name__ == "__main__":
    unittest.main()
