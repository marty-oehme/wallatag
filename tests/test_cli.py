"""Tests for wallatag.cli: parser, flag overrides, stubs and status output."""

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from wallatag.cli import (
    apply_flag_overrides,
    build_parser,
    cmd_status,
    main,
)
from wallatag.config import Config, ConfigError, load_config


def _args(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace shaped like a parsed subcommand invocation."""
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


class HelpTest(unittest.TestCase):
    """(a) --help exits 0 and mentions all three subcommands."""

    def test_help(self):
        parser = build_parser()
        out = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(out):
            parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        for sub in ("manual", "run", "status"):
            self.assertIn(sub, text)


class VersionTest(unittest.TestCase):
    """(b) --version exits 0 and prints the version."""

    def test_version(self):
        out = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(out):
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("0.1.0", out.getvalue())


class RunStubTest(unittest.TestCase):
    """(c) run with all flags returns 0 and does not raise."""

    def test_run_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "wallatag.toml"
            config_path.write_text("", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
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
        self.assertEqual(code, 0)
        self.assertIn("not implemented yet", out.getvalue())


class FlagOverrideTest(unittest.TestCase):
    """(d) CLI flags win over TOML values via apply_flag_overrides."""

    def _load(self, tmp: str) -> Config:
        path = Path(tmp) / "wallatag.toml"
        path.write_text('[tagger]\ntag_policy = "prefer-existing"\n', encoding="utf-8")
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
    """--max is a runtime run limit; it must not touch max_suggestions."""

    def test_max_does_not_alter_max_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text("[tagger]\nmax_suggestions = 9\n", encoding="utf-8")
            config = load_config(config_path=str(path), env={})
            overridden = apply_flag_overrides(config, _args(max=100))
            self.assertEqual(overridden.tagger.max_suggestions, 9)
            self.assertEqual(overridden.max_articles, 100)

    def test_status_shows_toml_max_suggestions_and_run_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallatag.toml"
            path.write_text("[tagger]\nmax_suggestions = 9\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["status", "--config", str(path), "--max", "100"])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("max_suggestions=9", text)
        self.assertNotIn("max_suggestions=100", text)
        self.assertIn("run limit: 100 articles", text)


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
                code = main(["status", "--config", str(path), "--focus", "bogus"])
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
                code = main(["status", "--config", str(path), "--focus", "methods"])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("focus groups: methods", text)


class MaxNegativeTest(unittest.TestCase):
    """--max -1 must fail loudly with a non-zero exit."""

    def test_negative_max_exits_2_via_parser(self):
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
                    '[wallabag]\n'
                    'url = "https://wallabag.example.com"\n'
                    'client_id = "cid"\n'
                    'client_secret = "super-secret-value"\n'
                ),
                encoding="utf-8",
            )
            config = load_config(config_path=str(path), env={})
            args = argparse.Namespace(config=str(path), max=None, focus=None,
                                     tag_policy=None, no_history=False,
                                     no_apply=False, verbose=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cmd_status(config, args)
            text = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("wallatag", text)
        self.assertIn("https://wallabag.example.com", text)
        self.assertNotIn("super-secret-value", text)
        self.assertNotIn("cid", text)

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


if __name__ == "__main__":
    unittest.main()
