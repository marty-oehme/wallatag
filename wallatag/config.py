"""Configuration loading and layering for wallatag.

Layering order (lowest to highest precedence): defaults < TOML file
(``wallatag.toml`` or ``--config``) < environment variables < CLI flags.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

VALID_TAG_POLICIES = ("only-existing", "prefer-existing", "all")
VALID_AI_PROVIDERS = ("ollama", "openai-compatible")

# Environment variable -> config field mapping.
_ENV_URL = "WALLATAG_URL"
_ENV_CLIENT_ID = "WALLATAG_CLIENT_ID"
_ENV_CLIENT_SECRET = "WALLATAG_CLIENT_SECRET"
_ENV_USERNAME = "WALLATAG_USERNAME"
_ENV_PASSWORD = "WALLATAG_PASSWORD"
_ENV_DB = "WALLATAG_DB"
_ENV_CONFIG = "WALLATAG_CONFIG"
_ENV_AI_PROVIDER = "WALLATAG_AI_PROVIDER"
_ENV_AI_BASE_URL = "WALLATAG_AI_BASE_URL"
_ENV_AI_MODEL = "WALLATAG_AI_MODEL"
_ENV_AI_CONFIDENCE_THRESHOLD = "WALLATAG_AI_CONFIDENCE_THRESHOLD"
_ENV_AI_API_KEY = "WALLATAG_AI_API_KEY"
_ENV_IGNORE_TAGS = "WALLATAG_IGNORE_TAGS"
_ENV_TAG_POLICY = "WALLATAG_TAG_POLICY"
_ENV_MAX_SUGGESTIONS = "WALLATAG_MAX_SUGGESTIONS"
_ENV_FOCUS_PREFIX = "WALLATAG_FOCUS_"
_ENV_FOCUS_KEYWORDS_SUFFIX = "_KEYWORDS"
_ENV_FOCUS_TAGS_SUFFIX = "_TAGS"

_DEFAULT_CONFIG_NAME = "wallatag.toml"


class ConfigError(Exception):
    """Raised when wallatag configuration is invalid or unusable."""


@dataclass(frozen=True)
class FocusGroup:
    keywords: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class WallabagConfig:
    url: str = ""
    client_id: str = ""
    client_secret: str = ""
    username: str = ""
    password: str = ""


@dataclass(frozen=True)
class StoreConfig:
    path: str | None = None  # None = history-less (no DB at all)


@dataclass(frozen=True)
class TaggerConfig:
    max_suggestions: int = 5
    tag_policy: str = "prefer-existing"
    focus_groups: dict[str, FocusGroup] = field(default_factory=dict)
    # Tags treated as untagged: articles carrying ONLY these tags are still
    # fetched. Empty default = only fully untagged articles are fetched.
    ignore_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AiConfig:
    """LLM tagger configuration.

    The LLM tagger is active iff ``provider`` is non-empty; a fully default
    ``AiConfig()`` (or a config that leaves all three of provider/base_url/
    model unset) means the LLM tagger is disabled and KeywordTagger is used.

    ``api_key`` is optional: when set (non-empty) the LLM client sends it as a
    bearer token (``Authorization: Bearer <api_key>``) on every request for
    openai-compatible providers that require auth; empty/unset means no
    Authorization header. It is NOT part of the atomic provider/base_url/model
    trio, so a config with only the trio (or only api_key) is valid. Like all
    config, the key must never be printed or logged.
    """

    provider: str = ""
    base_url: str = ""
    model: str = ""
    confidence_threshold: float = 0.7
    api_key: str = ""


@dataclass(frozen=True)
class Config:
    wallabag: WallabagConfig = field(default_factory=WallabagConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    verbose: bool = False
    # Runtime-only: sourced solely from the --max flag (max articles per run).
    # Never read from TOML: that is `[tagger] max_suggestions` instead.
    max_articles: int | None = None


def find_config_file(
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Locate the configuration file, or ``None`` for defaults-only.

    Candidate order: ``explicit`` (--config) > env ``WALLATAG_CONFIG`` >
    ``./wallatag.toml`` in the current working directory.

    An explicitly-given path (flag or env) that is not a file raises
    ``ConfigError``; a missing default file simply returns ``None``.
    """
    if env is None:
        env = os.environ

    candidate: str | None = None
    if explicit is not None:
        candidate = explicit
    elif env.get(_ENV_CONFIG):
        candidate = env[_ENV_CONFIG]

    if candidate is not None:
        path = Path(candidate)
        if not path.is_file():
            raise ConfigError(f"config file not found: {candidate}")
        return path

    default = Path.cwd() / _DEFAULT_CONFIG_NAME
    if default.is_file():
        return default
    return None


def _parse_toml(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc


def _parse_string_list(context: str, key: str, value: object) -> tuple[str, ...]:
    """Validate one list-of-strings key (list/tuple of str); return a tuple.

    ``context`` names the owning section for error messages (e.g. a focus
    group or ``[tagger]``). Raises ConfigError on a non-list value or any
    non-string element.
    """
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{context}: {key} must be a list of strings")
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(
                f"{context}: {key} contains a non-string value {item!r}"
            )
    return tuple(value)


def _parse_comma_separated(value: str) -> tuple[str, ...]:
    """Parse a comma-separated env-var value into a tuple of non-empty items.

    Items are stripped of surrounding whitespace and empty items are dropped,
    so ``""`` yields ``()`` (used to clear a TOML list). Shared by
    ``WALLATAG_IGNORE_TAGS`` and the ``WALLATAG_FOCUS_<NAME>_*`` vars.
    """
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_focus_groups(raw: dict) -> dict[str, FocusGroup]:
    groups: dict[str, FocusGroup] = {}
    # Two group names that differ only in case (e.g. `[focus.methods]` and
    # `[focus.Methods]`) would merge ambiguously when env vars override by
    # case-insensitive name, so they are rejected up front.
    seen_casefold: dict[str, str] = {}
    for name, section in raw.items():
        folded = name.casefold()
        previous = seen_casefold.get(folded)
        if previous is not None:
            raise ConfigError(
                "focus group names must be unique case-insensitively: "
                f"{previous!r} vs {name!r}"
            )
        seen_casefold[folded] = name
        if not isinstance(section, dict):
            raise ConfigError(
                f"focus group {name!r} must be a table with keywords and tags"
            )
        keywords = (
            _parse_string_list(f"focus group {name!r}", "keywords", section["keywords"])
            if "keywords" in section
            else ()
        )
        tags = (
            _parse_string_list(f"focus group {name!r}", "tags", section["tags"])
            if "tags" in section
            else ()
        )
        groups[name] = FocusGroup(keywords=keywords, tags=tags)
    return groups


def _parse_ai(raw: dict) -> AiConfig:
    """Parse and validate the [ai] section into an AiConfig.

    The LLM tagger is active iff ``provider`` is non-empty, so an absent or
    empty [ai] table yields defaults (LLM disabled). provider, base_url and
    model are an atomic trio: setting ANY one of them requires all three to be
    non-empty, and the provider must be a known value. confidence_threshold
    defaults to 0.7 and must be a number in (0, 1]; the bool-is-number trap is
    explicitly rejected (``confidence_threshold = true`` is not a threshold).
    ``api_key`` is optional and independent of the trio: it is str()-coerced
    like the other string fields and empty/unset simply means no auth header.
    """
    # The isinstance check runs on the RAW value BEFORE any `or {}`
    # normalization: falsy non-tables (`ai = ""`, `ai = []`) must raise, not
    # silently parse as an empty table. Only a dict value is normalized.
    ai_value = raw.get("ai", {})
    if not isinstance(ai_value, dict):
        raise ConfigError("section [ai] must be a table")
    ai_raw = ai_value or {}

    provider = ai_raw.get("provider", "") or ""
    base_url = ai_raw.get("base_url", "") or ""
    model = ai_raw.get("model", "") or ""
    api_key = ai_raw.get("api_key", "") or ""

    confidence = ai_raw.get("confidence_threshold", 0.7)
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 < float(confidence) <= 1
    ):
        raise ConfigError("confidence_threshold must be a number in the range (0, 1]")
    confidence = float(confidence)

    if provider or base_url or model:
        if not provider or not base_url or not model:
            raise ConfigError(
                "section [ai]: provider, base_url and model must all be set together"
            )
        if provider not in VALID_AI_PROVIDERS:
            raise ConfigError(
                "invalid ai provider %r; valid choices: %s"
                % (provider, ", ".join(VALID_AI_PROVIDERS))
            )

    return AiConfig(
        provider=str(provider),
        base_url=str(base_url),
        model=str(model),
        confidence_threshold=confidence,
        api_key=str(api_key),
    )


def _parse_toml_config(raw: dict) -> Config:
    wallabag_raw = raw.get("wallabag", {}) or {}
    store_raw = raw.get("store", {}) or {}
    tagger_raw = raw.get("tagger", {}) or {}

    if not isinstance(wallabag_raw, dict) or not isinstance(store_raw, dict) or not isinstance(tagger_raw, dict):
        raise ConfigError("sections [wallabag], [store], [tagger] must be tables")

    tag_policy = tagger_raw.get("tag_policy", "prefer-existing")
    if tag_policy not in VALID_TAG_POLICIES:
        raise ConfigError(
            f"invalid tag_policy {tag_policy!r}; valid choices: {', '.join(VALID_TAG_POLICIES)}"
        )

    max_suggestions = tagger_raw.get("max_suggestions", 5)
    if not isinstance(max_suggestions, int) or isinstance(max_suggestions, bool) or max_suggestions < 0:
        raise ConfigError("max_suggestions must be a non-negative integer")

    # Absent key -> () (no-op); empty array -> () too. Any present value goes
    # through the same list-of-strings validation as focus-group keywords/tags.
    ignore_tags_raw = tagger_raw.get("ignore_tags")
    ignore_tags = (
        _parse_string_list("tagger", "ignore_tags", ignore_tags_raw)
        if ignore_tags_raw is not None
        else ()
    )

    focus_raw = raw.get("focus", {}) or {}
    if not isinstance(focus_raw, dict):
        raise ConfigError("section [focus] must be a table")

    return Config(
        wallabag=WallabagConfig(
            url=str(wallabag_raw.get("url", "") or ""),
            client_id=str(wallabag_raw.get("client_id", "") or ""),
            client_secret=str(wallabag_raw.get("client_secret", "") or ""),
            username=str(wallabag_raw.get("username", "") or ""),
            password=str(wallabag_raw.get("password", "") or ""),
        ),
        store=StoreConfig(path=str(store_raw.get("path") or "") or None),
        tagger=TaggerConfig(
            max_suggestions=max_suggestions,
            tag_policy=tag_policy,
            focus_groups=_parse_focus_groups(focus_raw),
            ignore_tags=ignore_tags,
        ),
        ai=_parse_ai(raw),
        verbose=False,
    )


def _apply_env(config: Config, env: Mapping[str, str]) -> Config:
    """Overlay environment variables on top of a parsed config."""
    wallabag = config.wallabag
    url = env.get(_ENV_URL)
    client_id = env.get(_ENV_CLIENT_ID)
    client_secret = env.get(_ENV_CLIENT_SECRET)
    username = env.get(_ENV_USERNAME)
    password = env.get(_ENV_PASSWORD)
    if url is not None:
        wallabag = replace(wallabag, url=url)
    if client_id is not None:
        wallabag = replace(wallabag, client_id=client_id)
    if client_secret is not None:
        wallabag = replace(wallabag, client_secret=client_secret)
    if username is not None:
        wallabag = replace(wallabag, username=username)
    if password is not None:
        wallabag = replace(wallabag, password=password)

    store = config.store
    db = env.get(_ENV_DB)
    if db is not None:
        store = replace(store, path=db or None)  # empty string -> history-less

    ai = config.ai
    provider = env.get(_ENV_AI_PROVIDER)
    base_url = env.get(_ENV_AI_BASE_URL)
    model = env.get(_ENV_AI_MODEL)
    threshold_raw = env.get(_ENV_AI_CONFIDENCE_THRESHOLD)
    if threshold_raw is not None:
        try:
            confidence = float(threshold_raw)
        except ValueError:
            raise ConfigError(
                f"WALLATAG_AI_CONFIDENCE_THRESHOLD must be a number in the "
                f"range (0, 1], got {threshold_raw!r}"
            ) from None
        if not 0 < confidence <= 1:
            raise ConfigError(
                f"WALLATAG_AI_CONFIDENCE_THRESHOLD must be in the range "
                f"(0, 1], got {threshold_raw!r}"
            )
        ai = replace(ai, confidence_threshold=confidence)
    if provider is not None or base_url is not None or model is not None:
        # Each set env var wins over the TOML value, but the trio must still
        # end up complete and the provider valid (same rule as the TOML
        # parser): WALLATAG_AI_PROVIDER alone with no [ai] config is an error.
        provider = provider if provider is not None else ai.provider
        base_url = base_url if base_url is not None else ai.base_url
        model = model if model is not None else ai.model
        if not provider or not base_url or not model:
            raise ConfigError(
                "WALLATAG_AI_PROVIDER, WALLATAG_AI_BASE_URL and WALLATAG_AI_MODEL "
                "must all be set together"
            )
        if provider not in VALID_AI_PROVIDERS:
            raise ConfigError(
                "invalid ai provider %r; valid choices: %s"
                % (provider, ", ".join(VALID_AI_PROVIDERS))
            )
        ai = replace(ai, provider=provider, base_url=base_url, model=model)

    api_key = env.get(_ENV_AI_API_KEY)
    if api_key is not None:
        # A present-but-empty WALLATAG_AI_API_KEY clears the TOML value,
        # matching the WALLATAG_DB="" pattern.
        ai = replace(ai, api_key=api_key)

    tagger = config.tagger
    ignore_tags_raw = env.get(_ENV_IGNORE_TAGS)
    if ignore_tags_raw is not None:
        # Comma-separated string; empty string clears the TOML value (matching
        # the WALLATAG_DB="" pattern). Items are stripped of whitespace and
        # empty items are dropped.
        tagger = replace(
            tagger,
            ignore_tags=_parse_comma_separated(
                "tagger", _ENV_IGNORE_TAGS, ignore_tags_raw
            ),
        )
    tag_policy = env.get(_ENV_TAG_POLICY)
    if tag_policy is not None:
        if tag_policy not in VALID_TAG_POLICIES:
            raise ConfigError(
                f"invalid tag_policy {tag_policy!r} (from WALLATAG_TAG_POLICY); "
                f"valid choices: {', '.join(VALID_TAG_POLICIES)}"
            )
        tagger = replace(tagger, tag_policy=tag_policy)
    max_suggestions_raw = env.get(_ENV_MAX_SUGGESTIONS)
    if max_suggestions_raw is not None:
        try:
            max_suggestions = int(max_suggestions_raw)
        except ValueError:
            raise ConfigError(
                f"WALLATAG_MAX_SUGGESTIONS: max_suggestions must be a "
                f"non-negative integer, got {max_suggestions_raw!r}"
            ) from None
        if (
            not isinstance(max_suggestions, int)
            or isinstance(max_suggestions, bool)
            or max_suggestions < 0
        ):
            raise ConfigError(
                f"WALLATAG_MAX_SUGGESTIONS: max_suggestions must be a "
                f"non-negative integer, got {max_suggestions_raw!r}"
            )
        tagger = replace(tagger, max_suggestions=max_suggestions)

    # Focus groups: WALLATAG_FOCUS_<NAME>_KEYWORDS / WALLATAG_FOCUS_<NAME>_TAGS
    # (comma-separated, same parsing as WALLATAG_IGNORE_TAGS; "" clears that
    # field). Matching is by name, case-insensitively: an env var overrides the
    # same-named TOML group per-field (only the fields it sets), an env-only
    # group is created with the missing field defaulting to (), and TOML groups
    # without an env counterpart survive unchanged. Overrides of an existing
    # TOML group keep the TOML key's original casing; env-only groups are
    # stored lowercased. A var with the prefix but an empty name (e.g.
    # WALLATAG_FOCUS_KEYWORDS) is a ConfigError; WALLATAG_FOCUS_<NAME> without
    # a suffix is ignored.
    focus = dict(config.tagger.focus_groups)
    focus_by_casefold = {name.casefold(): name for name in focus}
    focus_changed = False
    for key in sorted(env):
        if not key.startswith(_ENV_FOCUS_PREFIX):
            continue
        if key.endswith(_ENV_FOCUS_KEYWORDS_SUFFIX):
            field = "keywords"
            name = key[
                len(_ENV_FOCUS_PREFIX) : len(key) - len(_ENV_FOCUS_KEYWORDS_SUFFIX)
            ]
        elif key.endswith(_ENV_FOCUS_TAGS_SUFFIX):
            field = "tags"
            name = key[len(_ENV_FOCUS_PREFIX) : len(key) - len(_ENV_FOCUS_TAGS_SUFFIX)]
        else:
            # No suffix: WALLATAG_FOCUS_ (bare prefix) and
            # WALLATAG_FOCUS_<NAME> alone are ignored.
            continue
        if not name:
            raise ConfigError(
                f"invalid focus-group environment variable {key}: the focus "
                f"group name is empty; expected WALLATAG_FOCUS_<NAME>_{field.upper()}"
            )
        storage_key = focus_by_casefold.get(name.casefold())
        if storage_key is None:
            storage_key = name.lower()
            focus_by_casefold[storage_key.casefold()] = storage_key
            focus[storage_key] = FocusGroup(keywords=(), tags=())
        group = focus[storage_key]
        if field == "keywords":
            group = replace(
                group,
                keywords=_parse_comma_separated("focus group", key, env[key]),
            )
        else:
            group = replace(
                group,
                tags=_parse_comma_separated("focus group", key, env[key]),
            )
        focus[storage_key] = group
        focus_changed = True
    if focus_changed:
        tagger = replace(tagger, focus_groups=focus)

    if (
        wallabag is config.wallabag
        and store is config.store
        and ai is config.ai
        and tagger is config.tagger
    ):
        return config
    return replace(
        config, wallabag=wallabag, store=store, ai=ai, tagger=tagger
    )


def load_config(
    config_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load and merge configuration: defaults < TOML < environment.

    ``env=None`` means ``os.environ``.
    """
    if env is None:
        env = os.environ

    path = find_config_file(explicit=config_path, env=env)
    config = _parse_toml_config(_parse_toml(path)) if path is not None else Config()
    config = _apply_env(config, env)
    return config
