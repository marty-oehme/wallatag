"""Tests for wallatag.config: TOML + env + defaults layering and validation."""

import os
import tempfile
import unittest
from pathlib import Path

from wallatag.config import (
    Config,
    ConfigError,
    FocusGroup,
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
        self.assertEqual(config.tagger.ignore_tags, ())
        self.assertFalse(config.verbose)


class TomlParsingTest(unittest.TestCase):
    """(b) TOML values are parsed, [ai] is parsed, focus groups tolerate gaps."""

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
base_url = "http://localhost:11434"
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
        self.assertEqual(config.ai.provider, "ollama")
        self.assertEqual(config.ai.base_url, "http://localhost:11434")
        self.assertEqual(config.ai.model, "qwen2.5:3b")

    def test_missing_keywords_tolerated_and_ai_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
tags = ["productivity"]

[ai]
""",
            )
            config = load_config(config_path=str(tmp / "wallatag.toml"), env={})

        group = config.tagger.focus_groups["methods"]
        self.assertEqual(group.keywords, ())
        self.assertEqual(group.tags, ("productivity",))
        # An empty [ai] table stays at the defaults: LLM tagger disabled.
        self.assertEqual(config.ai.provider, "")
        self.assertEqual(config.ai.base_url, "")
        self.assertEqual(config.ai.model, "")
        self.assertEqual(config.ai.confidence_threshold, 0.7)

    def test_ai_non_table_raises(self):
        # [ai] present but not a table (ai = "foo") must fail loudly now that
        # the section is actually parsed.
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
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})

        self.assertIn("[ai]", str(ctx.exception))
        self.assertIn("table", str(ctx.exception))

    def test_ai_empty_string_non_table_raises(self):
        # Falsy non-tables must NOT be masked by `or {}` normalization.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, 'ai = ""\n')
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        self.assertIn("section [ai] must be a table", str(ctx.exception))

    def test_ai_empty_array_non_table_raises(self):
        # A falsy [] is still a non-table: it must raise, not parse as empty.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, "ai = []\n")
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        self.assertIn("section [ai] must be a table", str(ctx.exception))


class IgnoreTagsTomlTest(unittest.TestCase):
    """[tagger] ignore_tags: list of tags treated as untagged."""

    def _load(self, toml_text: str) -> Config:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, toml_text)
            return load_config(config_path=str(tmp / "wallatag.toml"), env={})

    def test_ignore_tags_parsed(self):
        config = self._load('[tagger]\nignore_tags = ["fix", "_frigo"]\n')
        self.assertEqual(config.tagger.ignore_tags, ("fix", "_frigo"))

    def test_ignore_tags_absent_defaults_empty(self):
        # No ignore_tags key -> () with no error (a no-op).
        config = self._load('[tagger]\ntag_policy = "all"\n')
        self.assertEqual(config.tagger.ignore_tags, ())

    def test_ignore_tags_empty_array_defaults_empty(self):
        config = self._load("[tagger]\nignore_tags = []\n")
        self.assertEqual(config.tagger.ignore_tags, ())

    def test_ignore_tags_string_raises(self):
        # A plain string must be rejected, never split into characters.
        with self.assertRaises(ConfigError) as ctx:
            self._load('[tagger]\nignore_tags = "fix"\n')
        message = str(ctx.exception)
        self.assertIn("ignore_tags", message)
        self.assertIn("list of strings", message)

    def test_ignore_tags_non_string_element_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._load("[tagger]\nignore_tags = [1, 2]\n")
        message = str(ctx.exception)
        self.assertIn("ignore_tags", message)
        self.assertIn("non-string", message)


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

    def test_casefold_duplicate_group_names_raise(self):
        # [focus.methods] + [focus.Methods] differ only in case and would merge
        # ambiguously at env-override time: reject the TOML up front.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(
                tmp,
                """
[focus.methods]
keywords = ["pomodoro"]

[focus.Methods]
keywords = ["gtd"]
""",
            )
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path=str(tmp / "wallatag.toml"), env={})
        message = str(ctx.exception)
        self.assertIn("case-insensitively", message)
        self.assertIn("'methods'", message)
        self.assertIn("'Methods'", message)


class AiConfigTomlTest(unittest.TestCase):
    """[ai] is parsed into config.ai; the trio is atomic; threshold validated."""

    def _load(self, toml_text: str):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, toml_text)
            return load_config(config_path=str(tmp / "wallatag.toml"), env={})

    def test_full_ai_block_parsed(self):
        config = self._load(
            '[ai]\n'
            'provider = "ollama"\n'
            'base_url = "http://localhost:11434"\n'
            'model = "qwen2.5:3b"\n'
            'confidence_threshold = 0.8\n'
        )
        self.assertEqual(config.ai.provider, "ollama")
        self.assertEqual(config.ai.base_url, "http://localhost:11434")
        self.assertEqual(config.ai.model, "qwen2.5:3b")
        self.assertEqual(config.ai.confidence_threshold, 0.8)

    def test_default_confidence_threshold(self):
        config = self._load(
            '[ai]\n'
            'provider = "ollama"\n'
            'base_url = "http://localhost:11434"\n'
            'model = "qwen2.5:3b"\n'
        )
        self.assertEqual(config.ai.confidence_threshold, 0.7)

    def test_api_key_parsed(self):
        config = self._load(
            '[ai]\n'
            'provider = "ollama"\n'
            'base_url = "http://localhost:11434"\n'
            'model = "qwen2.5:3b"\n'
            'api_key = "sk-abc-123"\n'
        )
        self.assertEqual(config.ai.api_key, "sk-abc-123")

    def test_api_key_optional_not_part_of_trio(self):
        # provider/base_url/model without api_key stays valid.
        config = self._load(
            '[ai]\n'
            'provider = "openai-compatible"\n'
            'base_url = "https://api.example.com/v1"\n'
            'model = "gpt-4o-mini"\n'
        )
        self.assertEqual(config.ai.provider, "openai-compatible")
        self.assertEqual(config.ai.api_key, "")

    def test_api_key_only_is_valid(self):
        # api_key alone, without the trio, is a valid (LLM-disabled) config.
        config = self._load('[ai]\napi_key = "sk-alone"\n')
        self.assertEqual(config.ai.api_key, "sk-alone")
        self.assertEqual(config.ai.provider, "")

    def test_api_key_default_is_empty(self):
        config = self._load("[ai]\nprovider = \"ollama\"\nbase_url = \"http://x\"\nmodel = \"m\"\n")
        self.assertEqual(config.ai.api_key, "")

    def test_api_key_missing_section_disabled(self):
        config = self._load("")
        self.assertEqual(config.ai.api_key, "")

    def test_non_string_api_key_is_coerced_not_rejected(self):
        # A non-string api_key (e.g. from a TOML number) is str()-coerced like
        # the other string fields: it must NOT raise.
        config = self._load("[ai]\napi_key = 42\n")
        self.assertEqual(config.ai.api_key, "42")

    def test_confidence_threshold_only_disables_llm(self):
        # Only a threshold set: valid config, but no provider -> LLM disabled.
        config = self._load("[ai]\nconfidence_threshold = 0.9\n")
        self.assertEqual(config.ai.confidence_threshold, 0.9)
        self.assertEqual(config.ai.provider, "")
        self.assertEqual(config.ai.base_url, "")
        self.assertEqual(config.ai.model, "")

    def test_missing_model_with_provider_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._load(
                '[ai]\n'
                'provider = "ollama"\n'
                'base_url = "http://localhost:11434"\n'
            )
        self.assertIn("provider", str(ctx.exception))
        self.assertIn("model", str(ctx.exception))

    def test_missing_base_url_with_provider_raises(self):
        with self.assertRaises(ConfigError):
            self._load(
                '[ai]\nprovider = "ollama"\nmodel = "qwen2.5:3b"\n'
            )

    def test_bad_provider_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._load(
                '[ai]\n'
                'provider = "openai"\n'
                'base_url = "http://localhost:11434"\n'
                'model = "gpt-4o"\n'
            )
        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("openai-compatible", str(ctx.exception))

    def test_confidence_threshold_bad_values_raise(self):
        for bad in ("0", "-0.5", "1.5"):
            with self.assertRaises(ConfigError):
                self._load(f"[ai]\nconfidence_threshold = {bad}\n")

    def test_confidence_threshold_bool_raises(self):
        # bool is a float subclass; it must not be accepted as a threshold.
        with self.assertRaises(ConfigError):
            self._load("[ai]\nconfidence_threshold = true\n")

    def test_confidence_threshold_non_number_raises(self):
        with self.assertRaises(ConfigError):
            self._load('[ai]\nconfidence_threshold = "high"\n')

    def test_absent_ai_section_disabled(self):
        config = self._load("[tagger]\nmax_suggestions = 3\n")
        self.assertEqual(config.ai.provider, "")
        self.assertEqual(config.ai.confidence_threshold, 0.7)


class AiConfigEnvTest(unittest.TestCase):
    """WALLATAG_AI_* env vars overlay TOML with the same atomic trio rule."""

    def _env_load(self, toml_text: str, env: dict):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, toml_text)
            return load_config(config_path=str(tmp / "wallatag.toml"), env=env)

    def test_env_overrides_all_ai_fields(self):
        config = self._env_load(
            '[ai]\n'
            'provider = "ollama"\n'
            'base_url = "http://localhost:11434"\n'
            'model = "qwen2.5:3b"\n'
            'confidence_threshold = 0.7\n',
            {
                "WALLATAG_AI_PROVIDER": "openai-compatible",
                "WALLATAG_AI_BASE_URL": "https://api.example.com/v1",
                "WALLATAG_AI_MODEL": "gpt-4o-mini",
                "WALLATAG_AI_CONFIDENCE_THRESHOLD": "0.95",
            },
        )
        self.assertEqual(config.ai.provider, "openai-compatible")
        self.assertEqual(config.ai.base_url, "https://api.example.com/v1")
        self.assertEqual(config.ai.model, "gpt-4o-mini")
        self.assertEqual(config.ai.confidence_threshold, 0.95)

    def test_env_partial_trio_completes_from_toml(self):
        # Env sets only the provider; base_url/model come from TOML.
        config = self._env_load(
            '[ai]\n'
            'provider = "ollama"\n'
            'base_url = "http://localhost:11434"\n'
            'model = "qwen2.5:3b"\n',
            {"WALLATAG_AI_PROVIDER": "openai-compatible"},
        )
        self.assertEqual(config.ai.provider, "openai-compatible")
        self.assertEqual(config.ai.base_url, "http://localhost:11434")
        self.assertEqual(config.ai.model, "qwen2.5:3b")

    def test_env_provider_alone_without_toml_raises(self):
        with self.assertRaises(ConfigError):
            self._env_load(
                "[tagger]\nmax_suggestions = 1\n",
                {"WALLATAG_AI_PROVIDER": "ollama"},
            )

    def test_env_bad_provider_raises(self):
        with self.assertRaises(ConfigError):
            self._env_load(
                "",
                {
                    "WALLATAG_AI_PROVIDER": "openai",
                    "WALLATAG_AI_BASE_URL": "http://x",
                    "WALLATAG_AI_MODEL": "gpt-4o",
                },
            )

    def test_env_invalid_confidence_threshold_raises(self):
        with self.assertRaises(ConfigError):
            self._env_load("", {"WALLATAG_AI_CONFIDENCE_THRESHOLD": "abc"})
        with self.assertRaises(ConfigError):
            self._env_load("", {"WALLATAG_AI_CONFIDENCE_THRESHOLD": "2"})

    def test_env_threshold_only_disables_llm(self):
        config = self._env_load("", {"WALLATAG_AI_CONFIDENCE_THRESHOLD": "0.8"})
        self.assertEqual(config.ai.confidence_threshold, 0.8)
        self.assertEqual(config.ai.provider, "")

    def test_env_api_key_overrides_toml(self):
        config = self._env_load(
            '[ai]\n'
            'provider = "openai-compatible"\n'
            'base_url = "https://api.example.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "sk-from-toml"\n',
            {"WALLATAG_AI_API_KEY": "sk-from-env"},
        )
        self.assertEqual(config.ai.api_key, "sk-from-env")

    def test_env_api_key_empty_clears_toml(self):
        # A present-but-empty WALLATAG_AI_API_KEY clears the TOML value,
        # matching the WALLATAG_DB="" pattern.
        config = self._env_load(
            '[ai]\n'
            'provider = "openai-compatible"\n'
            'base_url = "https://api.example.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "sk-from-toml"\n',
            {"WALLATAG_AI_API_KEY": ""},
        )
        self.assertEqual(config.ai.api_key, "")

    def test_env_api_key_only_without_toml(self):
        # api_key alone via env (no trio) is valid: LLM stays disabled.
        config = self._env_load("", {"WALLATAG_AI_API_KEY": "sk-env-only"})
        self.assertEqual(config.ai.api_key, "sk-env-only")
        self.assertEqual(config.ai.provider, "")

    def test_env_trio_rule_unaffected_by_api_key(self):
        # Env trio incomplete still raises even when api_key is present.
        with self.assertRaises(ConfigError):
            self._env_load(
                "",
                {
                    "WALLATAG_AI_PROVIDER": "ollama",
                    "WALLATAG_AI_API_KEY": "sk-x",
                },
            )


class TaggerEnvTest(unittest.TestCase):
    """WALLATAG_* env vars overlay [tagger] settings (env wins over TOML)."""

    def _env_load(self, toml_text: str, env: dict):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, toml_text)
            return load_config(config_path=str(tmp / "wallatag.toml"), env=env)

    def test_env_overrides_all_tagger_fields(self):
        config = self._env_load(
            "",
            {
                "WALLATAG_TAG_POLICY": "all",
                "WALLATAG_MAX_SUGGESTIONS": "7",
                "WALLATAG_IGNORE_TAGS": "fix,_frigo",
            },
        )
        self.assertEqual(config.tagger.tag_policy, "all")
        self.assertEqual(config.tagger.max_suggestions, 7)
        self.assertEqual(config.tagger.ignore_tags, ("fix", "_frigo"))

    def test_env_ignore_tags_strips_whitespace_and_drops_empties(self):
        config = self._env_load(
            "", {"WALLATAG_IGNORE_TAGS": " fix , _frigo "}
        )
        self.assertEqual(config.tagger.ignore_tags, ("fix", "_frigo"))
        config = self._env_load("", {"WALLATAG_IGNORE_TAGS": "fix,"})
        self.assertEqual(config.tagger.ignore_tags, ("fix",))

    def test_env_ignore_tags_empty_clears_toml(self):
        # A present-but-empty WALLATAG_IGNORE_TAGS clears the TOML list,
        # matching the WALLATAG_DB="" pattern.
        config = self._env_load(
            '[tagger]\nignore_tags = ["fix"]\n',
            {"WALLATAG_IGNORE_TAGS": ""},
        )
        self.assertEqual(config.tagger.ignore_tags, ())

    def test_env_wins_over_toml(self):
        config = self._env_load(
            '[tagger]\nmax_suggestions = 3\ntag_policy = "only-existing"\n'
            'ignore_tags = ["fix"]\n',
            {
                "WALLATAG_MAX_SUGGESTIONS": "7",
                "WALLATAG_TAG_POLICY": "all",
            },
        )
        self.assertEqual(config.tagger.max_suggestions, 7)
        self.assertEqual(config.tagger.tag_policy, "all")
        # Not overridden -> falls back to the TOML value.
        self.assertEqual(config.tagger.ignore_tags, ("fix",))

    def test_env_invalid_tag_policy_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_TAG_POLICY": "nonsense"})
        message = str(ctx.exception)
        self.assertIn("WALLATAG_TAG_POLICY", message)
        self.assertIn("only-existing", message)
        self.assertIn("prefer-existing", message)
        self.assertIn("all", message)

    def test_env_negative_max_suggestions_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_MAX_SUGGESTIONS": "-3"})
        self.assertIn("WALLATAG_MAX_SUGGESTIONS", str(ctx.exception))

    def test_env_non_integer_max_suggestions_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_MAX_SUGGESTIONS": "abc"})
        self.assertIn("WALLATAG_MAX_SUGGESTIONS", str(ctx.exception))

    def test_env_max_suggestions_strips_whitespace(self):
        config = self._env_load("", {"WALLATAG_MAX_SUGGESTIONS": " 7 "})
        self.assertEqual(config.tagger.max_suggestions, 7)

    def test_env_max_suggestions_leading_zeros(self):
        config = self._env_load("", {"WALLATAG_MAX_SUGGESTIONS": "007"})
        self.assertEqual(config.tagger.max_suggestions, 7)

    def test_env_max_suggestions_bool_string_raises(self):
        # "True" is not an integer: the case-sensitive int() parse must fail,
        # exactly like "abc".
        with self.assertRaises(ConfigError):
            self._env_load("", {"WALLATAG_MAX_SUGGESTIONS": "True"})

    def test_env_tag_policy_is_case_sensitive(self):
        # Only lowercase "all" is valid, matching the TOML path: "ALL" must
        # raise, not be case-folded.
        with self.assertRaises(ConfigError):
            self._env_load("", {"WALLATAG_TAG_POLICY": "ALL"})

    def test_env_ignore_tags_drops_inner_empties(self):
        config = self._env_load("", {"WALLATAG_IGNORE_TAGS": "a,,b"})
        self.assertEqual(config.tagger.ignore_tags, ("a", "b"))

    def test_env_whitespace_only_ignore_tags_raises(self):
        # " " parses to () which would silently clear the list: reject it.
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_IGNORE_TAGS": " "})
        self.assertIn("WALLATAG_IGNORE_TAGS", str(ctx.exception))

    def test_env_comma_only_ignore_tags_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_IGNORE_TAGS": ","})
        self.assertIn("WALLATAG_IGNORE_TAGS", str(ctx.exception))

    def test_env_empty_tag_policy_raises(self):
        # A present-but-empty WALLATAG_TAG_POLICY is invalid, not a clear:
        # unset the var (e.g. `dokku config:unset`) instead.
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_TAG_POLICY": ""})
        self.assertIn("WALLATAG_TAG_POLICY", str(ctx.exception))

    def test_env_empty_max_suggestions_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_MAX_SUGGESTIONS": ""})
        self.assertIn("WALLATAG_MAX_SUGGESTIONS", str(ctx.exception))


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


class FocusEnvTest(unittest.TestCase):
    """WALLATAG_FOCUS_<NAME>_{KEYWORDS,TAGS} env vars merge focus groups by name."""

    def _env_load(self, toml_text: str, env: dict):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_toml(tmp, toml_text)
            return load_config(config_path=str(tmp / "wallatag.toml"), env=env)

    def test_env_overrides_toml_group_per_field(self):
        # Only the keywords field is overridden; tags fall back to TOML.
        config = self._env_load(
            """
[focus.methods]
keywords = ["pomodoro"]
tags = ["productivity"]
""",
            {"WALLATAG_FOCUS_METHODS_KEYWORDS": "gtd,zettelkasten"},
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"],
            FocusGroup(keywords=("gtd", "zettelkasten"), tags=("productivity",)),
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"].keywords, ("gtd", "zettelkasten")
        )
        self.assertEqual(config.tagger.focus_groups["methods"].tags, ("productivity",))

    def test_env_only_group_both_fields(self):
        config = self._env_load(
            "",
            {
                "WALLATAG_FOCUS_LANGUAGES_KEYWORDS": "python,rust",
                "WALLATAG_FOCUS_LANGUAGES_TAGS": "programming",
            },
        )
        self.assertEqual(
            config.tagger.focus_groups["languages"],
            FocusGroup(keywords=("python", "rust"), tags=("programming",)),
        )

    def test_case_insensitive_override_preserves_toml_casing(self):
        # Env name matches the TOML group case-insensitively; the TOML key's
        # original casing ("Methods") is kept, not replaced by "methods".
        config = self._env_load(
            """
[focus.Methods]
keywords = ["pomodoro"]
tags = ["productivity"]
""",
            {"WALLATAG_FOCUS_METHODS_TAGS": "gtd"},
        )
        self.assertIn("Methods", config.tagger.focus_groups)
        self.assertNotIn("methods", config.tagger.focus_groups)
        group = config.tagger.focus_groups["Methods"]
        self.assertEqual(group.keywords, ("pomodoro",))
        self.assertEqual(group.tags, ("gtd",))

    def test_env_empty_clears_field(self):
        config = self._env_load(
            """
[focus.methods]
keywords = ["pomodoro"]
tags = ["productivity"]
""",
            {"WALLATAG_FOCUS_METHODS_KEYWORDS": ""},
        )
        self.assertEqual(config.tagger.focus_groups["methods"].keywords, ())
        self.assertEqual(config.tagger.focus_groups["methods"].tags, ("productivity",))

    def test_env_only_group_keywords_only_defaults_tags(self):
        config = self._env_load("", {"WALLATAG_FOCUS_LANGUAGES_KEYWORDS": "python"})
        self.assertEqual(
            config.tagger.focus_groups["languages"],
            FocusGroup(keywords=("python",), tags=()),
        )

    def test_env_only_group_tags_only_defaults_keywords(self):
        config = self._env_load("", {"WALLATAG_FOCUS_LANGUAGES_TAGS": "programming"})
        self.assertEqual(
            config.tagger.focus_groups["languages"],
            FocusGroup(keywords=(), tags=("programming",)),
        )

    def test_env_insertion_order_does_not_matter(self):
        # _apply_env iterates sorted(env), so listing the group's _TAGS var
        # BEFORE its _KEYWORDS var in the dict still lands both fields on the
        # one merged group (deterministic regardless of insertion order).
        config = self._env_load(
            "",
            {
                "WALLATAG_FOCUS_METHODS_TAGS": "productivity",
                "WALLATAG_FOCUS_METHODS_KEYWORDS": "gtd",
            },
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"],
            FocusGroup(keywords=("gtd",), tags=("productivity",)),
        )

    def test_env_whitespace_only_focus_value_raises(self):
        # " " parses to () which would silently clear the field: reject it.
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_FOCUS_METHODS_KEYWORDS": " "})
        self.assertIn("WALLATAG_FOCUS_METHODS_KEYWORDS", str(ctx.exception))

    def test_env_strips_whitespace_and_drops_empties(self):
        config = self._env_load(
            "", {"WALLATAG_FOCUS_METHODS_KEYWORDS": " fix , _frigo "}
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"].keywords, ("fix", "_frigo")
        )
        config = self._env_load("", {"WALLATAG_FOCUS_METHODS_KEYWORDS": "fix,"})
        self.assertEqual(config.tagger.focus_groups["methods"].keywords, ("fix",))

    def test_toml_groups_without_env_counterpart_survive(self):
        # Unrelated env vars (WALLATAG_URL, FOO) must not touch focus groups.
        config = self._env_load(
            """
[focus.methods]
keywords = ["pomodoro"]
[focus.languages]
keywords = ["python"]
""",
            {"WALLATAG_URL": "https://x.example", "FOO": "bar"},
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"].keywords, ("pomodoro",)
        )
        self.assertEqual(
            config.tagger.focus_groups["languages"].keywords, ("python",)
        )

    def test_env_empty_name_raises(self):
        # Exactly WALLATAG_FOCUS_KEYWORDS: prefix + suffix but an empty name.
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_FOCUS_KEYWORDS": "x"})
        self.assertIn("WALLATAG_FOCUS_KEYWORDS", str(ctx.exception))
        with self.assertRaises(ConfigError) as ctx:
            self._env_load("", {"WALLATAG_FOCUS_TAGS": "x"})
        self.assertIn("WALLATAG_FOCUS_TAGS", str(ctx.exception))

    def test_env_no_suffix_alone_is_ignored(self):
        # WALLATAG_FOCUS_<NAME> without a suffix, and the bare prefix, are
        # ignored; the TOML group is untouched.
        config = self._env_load(
            """
[focus.methods]
keywords = ["pomodoro"]
""",
            {"WALLATAG_FOCUS_METHODS": "x", "WALLATAG_FOCUS_": "y"},
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"].keywords, ("pomodoro",)
        )

    def test_env_empty_matches_toml_only(self):
        # No focus env vars -> focus groups come from TOML unchanged.
        config = self._env_load(
            """
[focus.methods]
keywords = ["pomodoro"]
tags = ["productivity"]
""",
            {},
        )
        self.assertEqual(
            config.tagger.focus_groups["methods"],
            FocusGroup(keywords=("pomodoro",), tags=("productivity",)),
        )


if __name__ == "__main__":
    unittest.main()
