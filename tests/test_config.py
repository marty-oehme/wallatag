"""Tests for wallatag.config: TOML + env + defaults layering and validation."""

import os
import tempfile
import unittest
from pathlib import Path

from wallatag.config import (
    ConfigError,
    find_config_file,
    load_config,
)


def write_toml(tmp: Path, text: str) -> Path:
    path = tmp / "wallatag.toml"
    path.write_text(text, encoding="utf-8")
    return path


class NoConfigFileDefaultsTest(unittest.TestCase):
    """(a) No config file anywhere -> all defaults."""

    def test_defaults_with_empty_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                config = load_config(env={})
            finally:
                os.chdir(old_cwd)
        self.assertEqual(config.wallabag.url, "")
        self.assertEqual(config.wallabag.client_id, "")
        self.assertEqual(config.wallabag.client_secret, "")
        self.assertEqual(config.wallabag.username, "")
        self.assertEqual(config.wallabag.password, "")
        self.assertIsNone(config.store.path)
        self.assertEqual(config.tagger.max_suggestions, 5)
        self.assertEqual(config.tagger.tag_policy, "prefer-existing")
        self.assertEqual(config.tagger.focus_groups, {})
        self.assertFalse(config.verbose)


class TomlParsingTest(unittest.TestCase):
    """(b) TOML values are parsed, [ai] is ignored, focus groups tolerate gaps."""

    def test_toml_values_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[wallabag]
url = "https://wallabag.example.com"
client_id = "cid"
client_secret = "secret"
username = "alice"
password = "wonderland"

[store]
path = "/data/wallatag.db"

[tagger]
max_suggestions = 9
tag_policy = "all"

[focus.methods]
keywords = ["pomodoro", "gtd"]
tags = ["productivity"]

[focus.languages]
keywords = ["python"]
tags = ["programming"]

[ai]
provider = "ollama"
model = "qwen2.5:3b"
""",
            )
            config = load_config(config_path=str(tmp / "wallatag.toml"), env={})

        self.assertEqual(config.wallabag.url, "https://wallabag.example.com")
        self.assertEqual(config.wallabag.client_id, "cid")
        self.assertEqual(config.wallabag.client_secret, "secret")
        self.assertEqual(config.wallabag.username, "alice")
        self.assertEqual(config.wallabag.password, "wonderland")
        self.assertEqual(config.store.path, "/data/wallatag.db")
        self.assertEqual(config.tagger.max_suggestions, 9)
        self.assertEqual(config.tagger.tag_policy, "all")
        self.assertEqual(
            config.tagger.focus_groups["methods"].keywords, ("pomodoro", "gtd")
        )
        self.assertEqual(config.tagger.focus_groups["methods"].tags, ("productivity",))
        self.assertEqual(config.tagger.focus_groups["languages"].tags, ("programming",))

    def test_missing_keywords_tolerated_and_ai_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
tags = ["productivity"]

[ai]
provider = "ollama"
""",
            )
            config = load_config(config_path=str(tmp / "wallatag.toml"), env={})

        group = config.tagger.focus_groups["methods"]
        self.assertEqual(group.keywords, ())
        self.assertEqual(group.tags, ("productivity",))
        # [ai] silently ignored: no config surface for it.
        self.assertFalse(hasattr(config, "ai"))

    def test_ai_non_table_ignored_without_error(self):
        # The [ai] section is reserved for phase 2 and must be ignored
        # unconditionally, even when it is present but not a table.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
ai = "foo"

[wallabag]
url = "https://wallabag.example.com"
""",
            )
            config = load_config(config_path=str(tmp / "wallatag.toml"), env={})

        self.assertEqual(config.wallabag.url, "https://wallabag.example.com")
        self.assertFalse(hasattr(config, "ai"))


class FocusGroupValidationTest(unittest.TestCase):
    """Malformed [focus.<name>] keywords/tags must fail with ConfigError."""

    def test_non_iterable_keywords_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
keywords = 5
tags = ["productivity"]
""",
            )
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        message = str(ctx.exception)
        self.assertIn("methods", message)
        self.assertIn("keywords", message)
        self.assertIn("list of strings", message)

    def test_plain_string_keywords_raises_not_split(self):
        # A plain string must be rejected outright: never silently split
        # into individual characters.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
keywords = "abc"
""",
            )
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        message = str(ctx.exception)
        self.assertIn("keywords", message)
        self.assertIn("list of strings", message)

    def test_non_string_keywords_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
keywords = [1, 2]
""",
            )
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        message = str(ctx.exception)
        self.assertIn("methods", message)
        self.assertIn("keywords", message)
        self.assertIn("non-string", message)

    def test_valid_string_list_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
keywords = ["pomodoro", "gtd"]
tags = ["productivity"]
""",
            )
            config = load_config(config_path=str(tmp / "wallatag.toml"), env={})

        group = config.tagger.focus_groups["methods"]
        self.assertEqual(group.keywords, ("pomodoro", "gtd"))
        self.assertEqual(group.tags, ("productivity",))


class EnvOverridesTest(unittest.TestCase):
    """(c)+(d) Env overrides TOML; WALLATAG_DB="" means history-less."""

    def test_env_overrides_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[wallabag]
url = "https://from-toml.example.com"
client_id = "toml-cid"
client_secret = "toml-secret"
username = "toml-user"
password = "toml-pass"

[store]
path = "/data/from-toml.db"
""",
            )
            config = load_config(
                config_path=str(tmp / "wallatag.toml"),
                env={
                    "WALLATAG_URL": "https://from-env.example.com",
                    "WALLATAG_CLIENT_ID": "env-cid",
                    "WALLATAG_DB": "/x.db",
                    "WALLATAG_USERNAME": "env-user",
                    "WALLATAG_PASSWORD": "env-pass",
                },
            )

        self.assertEqual(config.wallabag.url, "https://from-env.example.com")
        self.assertEqual(config.wallabag.client_id, "env-cid")
        # Not overridden -> falls back to TOML value.
        self.assertEqual(config.wallabag.client_secret, "toml-secret")
        # Overridden by WALLATAG_USERNAME / WALLATAG_PASSWORD.
        self.assertEqual(config.wallabag.username, "env-user")
        self.assertEqual(config.wallabag.password, "env-pass")
        self.assertEqual(config.store.path, "/x.db")

    def test_env_empty_db_is_history_less(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                '[store]\npath = "/data/from-toml.db"\n',
            )
            config = load_config(
                config_path=str(tmp / "wallatag.toml"),
                env={"WALLATAG_DB": ""},
            )

        self.assertIsNone(config.store.path)


class FindConfigFileTest(unittest.TestCase):
    """(e) explicit config path that doesn't exist -> ConfigError."""

    def test_explicit_missing_path_raises(self):
        with self.assertRaises(ConfigError):
            find_config_file(explicit="/nonexistent/config.toml", env={})

    def test_env_missing_path_raises(self):
        with self.assertRaises(ConfigError):
            find_config_file(env={"WALLATAG_CONFIG": "/nonexistent/config.toml"})

    def test_missing_default_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertIsNone(find_config_file(env={}))
            finally:
                os.chdir(old_cwd)

    def test_cwd_default_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, "")
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                found = find_config_file(env={})
            finally:
                os.chdir(old_cwd)
        self.assertEqual(found, tmp / "wallatag.toml")


class ValidationTest(unittest.TestCase):
    """(f) invalid tag_policy -> ConfigError; (g) negative max -> ConfigError."""

    def test_invalid_tag_policy_in_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, '[tagger]\ntag_policy = "nonsense"\n')
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        self.assertIn("only-existing", str(ctx.exception))
        self.assertIn("prefer-existing", str(ctx.exception))
        self.assertIn("all", str(ctx.exception))

    def test_negative_max_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, "[tagger]\nmax_suggestions = -3\n")
            with self.assertRaises(ConfigError):
                load_config(config_path=str(tmp / "wallatag.toml"), env={})


if __name__ == "__main__":
    unittest.main()
