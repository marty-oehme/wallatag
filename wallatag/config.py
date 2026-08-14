"""Configuration loading and layering for wallatag.

Layering order (lowest to highest precedence): defaults < TOML file
(``wallatag.toml`` or ``--config``) < environment variables < CLI flags.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

VALID_TAG_POLICIES = ("only-existing", "prefer-existing", "all")
VALID_AI_PROVIDERS = ("ollama", "openai-compatible")
# The article dict keys the keyword tagger can match against (also the keys
# _field_needles reads). Single source of truth: tagger.py imports it.
VALID_MATCH_FIELDS = ("title", "url", "domain_name", "content")

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
_ENV_AI_USE_FOCUS_GROUPS = "WALLATAG_AI_USE_FOCUS_GROUPS"
_ENV_AI_FALLBACK_ON_FAIL = "WALLATAG_AI_FALLBACK_ON_FAIL"
_ENV_AI_MAX_PROPOSALS = "WALLATAG_AI_MAX_PROPOSALS"
_ENV_IGNORE_TAGS = "WALLATAG_IGNORE_TAGS"
_ENV_IGNORE_TAGS_REGEX = "WALLATAG_IGNORE_TAGS_REGEX"
_ENV_TAG_POLICY = "WALLATAG_TAG_POLICY"
_ENV_MAX_APPLIED_TAGS = "WALLATAG_MAX_APPLIED_TAGS"
_ENV_ENABLE_VOCABULARY = "WALLATAG_ENABLE_VOCABULARY"
_ENV_ENABLE_RULES = "WALLATAG_ENABLE_RULES"
_ENV_ENABLE_LLM = "WALLATAG_ENABLE_LLM"
_ENV_FOCUS_PREFIX = "WALLATAG_FOCUS_"
_ENV_FOCUS_KEYWORDS_SUFFIX = "_KEYWORDS"
_ENV_FOCUS_KEYWORDS_REGEX_SUFFIX = "_KEYWORDS_REGEX"
_ENV_FOCUS_TAGS_SUFFIX = "_TAGS"
_ENV_FOCUS_FIELDS_SUFFIX = "_FIELDS"
_ENV_VOCABULARY_FIELDS = "WALLATAG_VOCABULARY_FIELDS"

_DEFAULT_CONFIG_NAME = "wallatag.toml"


class ConfigError(Exception):
    """Raised when wallatag configuration is invalid or unusable."""


@dataclass(frozen=True)
class FocusGroup:
    keywords: tuple[str, ...]
    tags: tuple[str, ...]
    # Article fields this group's keywords match against. None = the keyword
    # tagger's default (all four fields); an explicitly empty tuple disables
    # the group entirely. None distinguishes "absent" from "explicitly empty".
    fields: tuple[str, ...] | None = None
    # Optional regex patterns for this group, parallel to keywords: the group
    # fires when ANY literal keyword matches OR ANY regex matches (per-field,
    # never across fields). Patterns are stored raw and validated at parse
    # time (invalid and empty/whitespace-only patterns are ConfigErrors);
    # matching is case-insensitive by default (re.IGNORECASE) and reads the
    # RAW article field values, so inline flags like ``(?-i:...)`` still work.
    # Empty default: regex matching is off.
    keywords_regex: tuple[str, ...] = ()


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
class VocabularyConfig:
    """Which article fields the keyword tagger's vocabulary matcher checks.

    ``fields`` is a tuple of article dict keys (from ``VALID_MATCH_FIELDS``)
    that existing-tag labels are matched against. The default is all four
    fields; an explicitly empty tuple disables vocabulary matching entirely
    (absent config falls back to the all-fields default at parse time).
    """

    fields: tuple[str, ...] = VALID_MATCH_FIELDS


@dataclass(frozen=True)
class TaggerConfig:
    max_applied_tags: int = 5
    tag_policy: str = "prefer-existing"
    focus_groups: dict[str, FocusGroup] = field(default_factory=dict)
    # Tags treated as untagged: articles carrying ONLY these tags are still
    # fetched. Empty default = only fully untagged articles are fetched.
    ignore_tags: tuple[str, ...] = ()
    # Regex patterns for tags treated as untagged, parallel to ignore_tags: an
    # article carrying ONLY tags that are literal-ignored or match a pattern is
    # still fetched. Empty default.
    ignore_tags_regex: tuple[str, ...] = ()
    # Per-source enable switches for the keyword tagger. enable_vocabulary
    # gates the existing-tag vocabulary matcher; enable_rules gates focus-group
    # rule matching. Both default to True (existing behavior).
    enable_vocabulary: bool = True
    enable_rules: bool = True
    # enable_llm gates LLM classification. DELIBERATELY opt-in (default
    # False): a configured [ai] provider trio alone no longer activates the
    # LLM tagger; it requires enable_llm = true (or
    # WALLATAG_ENABLE_LLM=true) as well.
    enable_llm: bool = False


@dataclass(frozen=True)
class AiConfig:
    """LLM tagger configuration.

    The LLM tagger is active iff ``provider`` is non-empty AND
    ``[tagger] enable_llm`` is true (LLM tagging is opt-in; ``enable_llm``
    defaults to false). A fully default ``AiConfig()`` (or a config that
    leaves all three of provider/base_url/model unset) means the LLM tagger
    is disabled and KeywordTagger is used.

    ``api_key`` is optional: when set (non-empty) the LLM client sends it as a
    bearer token (``Authorization: Bearer <api_key>``) on every request for
    openai-compatible providers that require auth; empty/unset means no
    Authorization header. It is NOT part of the atomic provider/base_url/model
    trio, so a config with only the trio (or only api_key) is valid. Like all
    config, the key must never be printed or logged.

    ``use_focus_groups`` toggles whether focus groups influence LLM tagging;
    False = keyword-only mode for focus groups (the LLM prompt omits the
    "Focus areas" line entirely).

    ``fallback_on_fail`` (default False) toggles the per-article keyword
    fallback: when True and the LLM tagger fails for an article (LLMError
    propagates from ``suggest`` after retries are exhausted), the keyword
    tagger takes over for THAT article. The fallback behaves exactly like a
    normal keyword-mode run (enable_vocabulary/enable_rules/tag_policy all
    apply); the LLM is still tried on subsequent articles.

    ``max_proposals`` (default None) sets how many tags the LLM is ASKED to
    propose (the "Return at most N tags." line in the system prompt). None
    follows ``[tagger] max_applied_tags`` (backward-compatible default). It
    only affects the prompt: the applied list is always capped by
    ``max_applied_tags`` regardless.
    """

    provider: str = ""
    base_url: str = ""
    model: str = ""
    confidence_threshold: float = 0.7
    max_proposals: int | None = None
    api_key: str = ""
    use_focus_groups: bool = True
    fallback_on_fail: bool = False


@dataclass(frozen=True)
class Config:
    wallabag: WallabagConfig = field(default_factory=WallabagConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
    verbose: bool = False
    # Runtime-only: sourced solely from the --max flag (max articles per run).
    # Never read from TOML: that is `[tagger] max_applied_tags` instead.
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


def _parse_comma_separated(context: str, key: str, value: str) -> tuple[str, ...]:
    """Parse a comma-separated env-var value into a tuple of non-empty items.

    Items are stripped of surrounding whitespace and empty items are dropped,
    so ``""`` yields ``()`` (used to clear a TOML list). A non-empty value
    that parses to ``()``, i.e. only separators/whitespace, is rejected
    instead of silently clearing the list, since that is almost certainly a
    typo. Shared by ``WALLATAG_IGNORE_TAGS``, ``WALLATAG_IGNORE_TAGS_REGEX``,
    ``WALLATAG_VOCABULARY_FIELDS`` and the ``WALLATAG_FOCUS_<NAME>_*`` vars.
    ``context``/``key`` name the source in error messages, mirroring the
    ``_parse_string_list`` convention.
    """
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if value != "" and not parsed:
        raise ConfigError(
            f"{context}: {key} must be a comma-separated list; "
            "got only separators/whitespace"
        )
    return parsed


def _parse_bool_env(context: str, key: str, value: str) -> bool:
    """Parse an env-var boolean, case-insensitively, with strict validation.

    Accepts ``true``/``1``/``yes`` -> True and ``false``/``0``/``no`` -> False
    (any casing); surrounding whitespace is trimmed before matching, so padded
    values like ``" yes "`` or ``"true "`` are accepted (consistent with
    ``_parse_comma_separated``). Anything else, including ``""``, raises
    ConfigError naming the key, mirroring the existing env error styles.
    ``context``/``key`` name the source in error messages (e.g. the [ai]
    section and the env var).
    """
    folded = value.strip().casefold()
    if folded in ("true", "1", "yes"):
        return True
    if folded in ("false", "0", "no"):
        return False
    raise ConfigError(f"{context}: {key} must be true or false, got {value!r}")


def _validate_match_fields(
    context: str, key: str, fields: tuple[str, ...]
) -> tuple[str, ...]:
    """Validate a tuple of article-field names against ``VALID_MATCH_FIELDS``.

    Strict, case-sensitive, exact membership: an unknown member (e.g. a typo
    like ``"title "`` or ``"Title"``) raises ConfigError naming the offending
    member and listing the valid choices. Returns the tuple unchanged on
    success. Both the TOML path and the env path call it, so a bad field name
    fails loudly instead of silently narrowing the matcher.
    """
    for field_name in fields:
        if field_name not in VALID_MATCH_FIELDS:
            raise ConfigError(
                f"{context}: {key} contains unknown field {field_name!r}; "
                f"valid choices: {', '.join(VALID_MATCH_FIELDS)}"
            )
    return fields


def _validate_regexes(
    context: str, key: str, patterns: tuple[str, ...]
) -> tuple[str, ...]:
    """Validate a tuple of regex patterns; return it unchanged.

    Generic across config contexts: focus-group ``keywords_regex`` and
    ``[tagger] ignore_tags_regex`` both call it (via the TOML path and the env
    path) so a bad pattern fails loudly instead of silently never/always
    matching. Patterns are stored raw (matching reads the raw article
    fields/tags), so only compilability is validated here, with no flags:
    matching applies ``re.IGNORECASE`` itself and inline flags like
    ``(?-i:...)`` must survive intact. Empty or whitespace-only patterns are
    rejected before compiling: an empty regex matches everything, which is
    almost certainly a mistake. ``context``/``key`` name the source in error
    messages (e.g. ``focus group 'methods'``/``keywords_regex`` or
    ``tagger``/``ignore_tags_regex``), mirroring the ``_validate_match_fields``
    convention.
    """
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ConfigError(
                f"{context}: {key} contains an empty or whitespace-only regex "
                f"pattern {pattern!r}; an empty regex matches everything"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(
                f"{context}: {key} contains invalid regex {pattern!r}: {exc}"
            ) from exc
    return patterns


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
                f"focus group {name!r} must be a table with "
                "keywords/tags/fields/keywords_regex"
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
        # Optional per-group match fields. Absent -> None (the tagger's
        # default: all four fields); present -> strict validation, and an
        # explicitly empty list () disables the group entirely.
        fields = (
            _validate_match_fields(
                f"focus group {name!r}",
                "fields",
                _parse_string_list(f"focus group {name!r}", "fields", section["fields"]),
            )
            if "fields" in section
            else None
        )
        # Optional per-group regex patterns, parallel to keywords. Parsed like
        # keywords (list of strings), then validated: invalid or
        # empty/whitespace-only patterns are ConfigErrors at parse time.
        keywords_regex = (
            _validate_regexes(
                f"focus group {name!r}",
                "keywords_regex",
                _parse_string_list(
                    f"focus group {name!r}",
                    "keywords_regex",
                    section["keywords_regex"],
                ),
            )
            if "keywords_regex" in section
            else ()
        )
        groups[name] = FocusGroup(
            keywords=keywords,
            tags=tags,
            fields=fields,
            keywords_regex=keywords_regex,
        )
    return groups


def _parse_ai(raw: dict) -> AiConfig:
    """Parse and validate the [ai] section into an AiConfig.

    The LLM tagger is active iff ``provider`` is non-empty AND
    ``[tagger] enable_llm`` is true, so an absent or empty [ai] table yields
    defaults (LLM disabled regardless). provider, base_url and model are an
    atomic trio: setting ANY one of them requires all three to be non-empty,
    and the provider must be a known value. confidence_threshold
    defaults to 0.7 and must be a number in (0, 1]; the bool-is-number trap is
    explicitly rejected (``confidence_threshold = true`` is not a threshold).
    ``api_key`` is optional and independent of the trio: it is str()-coerced
    like the other string fields and empty/unset simply means no auth header.
    ``use_focus_groups`` defaults to True and must be a strict boolean; False
    makes focus groups keyword-only (the LLM prompt omits the focus areas).
    ``fallback_on_fail`` defaults to False and must be a strict boolean too;
    True enables the per-article keyword fallback when the LLM tagger fails.
    ``max_proposals`` (default None) optionally overrides how many tags the
    LLM is asked to propose; when set it must be a non-negative integer (not a
    bool), mirroring the max_applied_tags validation. None follows
    ``[tagger] max_applied_tags``.
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

    use_focus_groups = ai_raw.get("use_focus_groups", True)
    if not isinstance(use_focus_groups, bool):
        raise ConfigError("use_focus_groups must be a boolean (true or false)")

    fallback_on_fail = ai_raw.get("fallback_on_fail", False)
    if not isinstance(fallback_on_fail, bool):
        raise ConfigError("fallback_on_fail must be a boolean (true or false)")

    confidence = ai_raw.get("confidence_threshold", 0.7)
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 < float(confidence) <= 1
    ):
        raise ConfigError("confidence_threshold must be a number in the range (0, 1]")
    confidence = float(confidence)

    # Optional LLM proposal bound: None follows [tagger] max_applied_tags.
    # Mirrors the max_applied_tags validation (non-negative int, not bool).
    max_proposals = ai_raw.get("max_proposals")
    if max_proposals is not None and (
        not isinstance(max_proposals, int)
        or isinstance(max_proposals, bool)
        or max_proposals < 0
    ):
        raise ConfigError("max_proposals must be a non-negative integer")

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
        max_proposals=max_proposals,
        api_key=str(api_key),
        use_focus_groups=use_focus_groups,
        fallback_on_fail=fallback_on_fail,
    )


def _parse_toml_config(raw: dict) -> Config:
    wallabag_raw = raw.get("wallabag", {}) or {}
    store_raw = raw.get("store", {}) or {}
    tagger_raw = raw.get("tagger", {}) or {}

    if not isinstance(wallabag_raw, dict) or not isinstance(store_raw, dict) or not isinstance(tagger_raw, dict):
        raise ConfigError("sections [wallabag], [store], [tagger] must be tables")
    # The isinstance check runs on the RAW value BEFORE any `or {}`
    # normalization (mirrors _parse_ai): falsy non-tables (`vocabulary = ""`,
    # `vocabulary = false`, `vocabulary = []`) must raise, not silently parse
    # as an empty table. Only a dict value is normalized.
    vocabulary_value = raw.get("vocabulary", {})
    if not isinstance(vocabulary_value, dict):
        raise ConfigError("section [vocabulary] must be a table")
    vocabulary_raw = vocabulary_value or {}

    tag_policy = tagger_raw.get("tag_policy", "prefer-existing")
    if tag_policy not in VALID_TAG_POLICIES:
        raise ConfigError(
            f"invalid tag_policy {tag_policy!r}; valid choices: {', '.join(VALID_TAG_POLICIES)}"
        )

    max_applied_tags = tagger_raw.get("max_applied_tags", 5)
    if not isinstance(max_applied_tags, int) or isinstance(max_applied_tags, bool) or max_applied_tags < 0:
        raise ConfigError("max_applied_tags must be a non-negative integer")

    # Absent key -> () (no-op); empty array -> () too. Any present value goes
    # through the same list-of-strings validation as focus-group keywords/tags.
    ignore_tags_raw = tagger_raw.get("ignore_tags")
    ignore_tags = (
        _parse_string_list("tagger", "ignore_tags", ignore_tags_raw)
        if ignore_tags_raw is not None
        else ()
    )

    # Regex patterns for ignored tags, parallel to ignore_tags. Parsed like
    # keywords_regex (list of strings), then validated: invalid or
    # empty/whitespace-only patterns are ConfigErrors at load time, so a bad
    # pattern never reaches the client.
    ignore_tags_regex_raw = tagger_raw.get("ignore_tags_regex")
    ignore_tags_regex = (
        _validate_regexes(
            "tagger",
            "ignore_tags_regex",
            _parse_string_list("tagger", "ignore_tags_regex", ignore_tags_regex_raw),
        )
        if ignore_tags_regex_raw is not None
        else ()
    )

    # Per-source enable switches, strict booleans (mirrors _parse_ai's
    # use_focus_groups). enable_vocabulary/enable_rules default to True
    # (existing behavior); enable_llm is opt-in and defaults to False.
    enable_vocabulary = tagger_raw.get("enable_vocabulary", True)
    if not isinstance(enable_vocabulary, bool):
        raise ConfigError("enable_vocabulary must be a boolean (true or false)")
    enable_rules = tagger_raw.get("enable_rules", True)
    if not isinstance(enable_rules, bool):
        raise ConfigError("enable_rules must be a boolean (true or false)")
    enable_llm = tagger_raw.get("enable_llm", False)
    if not isinstance(enable_llm, bool):
        raise ConfigError("enable_llm must be a boolean (true or false)")

    focus_raw = raw.get("focus", {}) or {}
    if not isinstance(focus_raw, dict):
        raise ConfigError("section [focus] must be a table")

    # [vocabulary] fields: which article fields the vocabulary matcher checks.
    # Absent -> all four fields (the default); present -> list-of-strings
    # parsing plus strict field-name validation. An empty list () disables
    # vocabulary matching entirely.
    vocabulary_fields = (
        _validate_match_fields(
            "tagger",
            "vocabulary_fields",
            _parse_string_list("tagger", "fields", vocabulary_raw["fields"]),
        )
        if "fields" in vocabulary_raw
        else VALID_MATCH_FIELDS
    )

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
            max_applied_tags=max_applied_tags,
            tag_policy=tag_policy,
            focus_groups=_parse_focus_groups(focus_raw),
            ignore_tags=ignore_tags,
            ignore_tags_regex=ignore_tags_regex,
            enable_vocabulary=enable_vocabulary,
            enable_rules=enable_rules,
            enable_llm=enable_llm,
        ),
        ai=_parse_ai(raw),
        vocabulary=VocabularyConfig(fields=vocabulary_fields),
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

    use_focus_groups_raw = env.get(_ENV_AI_USE_FOCUS_GROUPS)
    if use_focus_groups_raw is not None:
        ai = replace(
            ai,
            use_focus_groups=_parse_bool_env(
                "ai", _ENV_AI_USE_FOCUS_GROUPS, use_focus_groups_raw
            ),
        )

    fallback_on_fail_raw = env.get(_ENV_AI_FALLBACK_ON_FAIL)
    if fallback_on_fail_raw is not None:
        ai = replace(
            ai,
            fallback_on_fail=_parse_bool_env(
                "ai", _ENV_AI_FALLBACK_ON_FAIL, fallback_on_fail_raw
            ),
        )

    max_proposals_raw = env.get(_ENV_AI_MAX_PROPOSALS)
    if max_proposals_raw is not None:
        if max_proposals_raw == "":
            # A present-but-empty WALLATAG_AI_MAX_PROPOSALS clears the TOML
            # value back to None (the prompt bound follows
            # [tagger] max_applied_tags), matching the WALLATAG_DB="" pattern.
            ai = replace(ai, max_proposals=None)
        else:
            try:
                max_proposals = int(max_proposals_raw)
            except ValueError:
                raise ConfigError(
                    f"WALLATAG_AI_MAX_PROPOSALS: max_proposals must be a "
                    f"non-negative integer, got {max_proposals_raw!r}"
                ) from None
            # The isinstance guards mirror the TOML path, where tomllib can
            # parse `max_proposals = true` as a bool; they are unreachable
            # here because int() of a str can never return a bool or a
            # non-int. Kept for symmetry so both paths validate identically.
            if (
                not isinstance(max_proposals, int)
                or isinstance(max_proposals, bool)
                or max_proposals < 0
            ):
                raise ConfigError(
                    f"WALLATAG_AI_MAX_PROPOSALS: max_proposals must be a "
                    f"non-negative integer, got {max_proposals_raw!r}"
                )
            ai = replace(ai, max_proposals=max_proposals)

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
    ignore_tags_regex_raw = env.get(_ENV_IGNORE_TAGS_REGEX)
    if ignore_tags_regex_raw is not None:
        # Comma-separated string; empty string clears the TOML value. Items
        # are stripped and empties dropped, then compilability is validated.
        tagger = replace(
            tagger,
            ignore_tags_regex=_validate_regexes(
                "tagger",
                _ENV_IGNORE_TAGS_REGEX,
                _parse_comma_separated(
                    "tagger", _ENV_IGNORE_TAGS_REGEX, ignore_tags_regex_raw
                ),
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
    max_applied_tags_raw = env.get(_ENV_MAX_APPLIED_TAGS)
    if max_applied_tags_raw is not None:
        try:
            max_applied_tags = int(max_applied_tags_raw)
        except ValueError:
            raise ConfigError(
                f"WALLATAG_MAX_APPLIED_TAGS: max_applied_tags must be a "
                f"non-negative integer, got {max_applied_tags_raw!r}"
            ) from None
        # The isinstance guards mirror the TOML path, where tomllib can parse
        # `max_applied_tags = true` as a bool; they are unreachable here because
        # int() of a str can never return a bool or a non-int. Kept for
        # symmetry so both paths validate identically.
        if (
            not isinstance(max_applied_tags, int)
            or isinstance(max_applied_tags, bool)
            or max_applied_tags < 0
        ):
            raise ConfigError(
                f"WALLATAG_MAX_APPLIED_TAGS: max_applied_tags must be a "
                f"non-negative integer, got {max_applied_tags_raw!r}"
            )
        tagger = replace(tagger, max_applied_tags=max_applied_tags)

    # Per-source enable switches, parsed strictly as booleans (same accepted
    # values as WALLATAG_AI_USE_FOCUS_GROUPS: true/1/yes, false/0/no,
    # case-insensitive).
    enable_vocabulary_raw = env.get(_ENV_ENABLE_VOCABULARY)
    if enable_vocabulary_raw is not None:
        tagger = replace(
            tagger,
            enable_vocabulary=_parse_bool_env(
                "tagger", _ENV_ENABLE_VOCABULARY, enable_vocabulary_raw
            ),
        )
    enable_rules_raw = env.get(_ENV_ENABLE_RULES)
    if enable_rules_raw is not None:
        tagger = replace(
            tagger,
            enable_rules=_parse_bool_env(
                "tagger", _ENV_ENABLE_RULES, enable_rules_raw
            ),
        )
    enable_llm_raw = env.get(_ENV_ENABLE_LLM)
    if enable_llm_raw is not None:
        tagger = replace(
            tagger,
            enable_llm=_parse_bool_env("tagger", _ENV_ENABLE_LLM, enable_llm_raw),
        )

    # Vocabulary match fields: WALLATAG_VOCABULARY_FIELDS (comma-separated,
    # same parsing as WALLATAG_IGNORE_TAGS). "" clears the TOML list -> the
    # vocabulary matcher is disabled (matches nothing); whitespace-only or
    # comma-only values are rejected.
    vocabulary = config.vocabulary
    vocabulary_fields_raw = env.get(_ENV_VOCABULARY_FIELDS)
    if vocabulary_fields_raw is not None:
        vocabulary = replace(
            vocabulary,
            fields=_validate_match_fields(
                "tagger",
                _ENV_VOCABULARY_FIELDS,
                _parse_comma_separated(
                    "tagger", _ENV_VOCABULARY_FIELDS, vocabulary_fields_raw
                ),
            ),
        )

    # Focus groups: WALLATAG_FOCUS_<NAME>_KEYWORDS / WALLATAG_FOCUS_<NAME>_TAGS
    # / WALLATAG_FOCUS_<NAME>_FIELDS / WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX
    # (comma-separated, same parsing as WALLATAG_IGNORE_TAGS; "" clears that
    # field). Matching is by name,
    # case-insensitively: an env var overrides the same-named TOML group
    # per-field (only the fields it sets), an env-only group is created with
    # the missing field defaulting to (), and TOML groups without an env
    # counterpart survive unchanged. Overrides of an existing TOML group keep
    # the TOML key's original casing; env-only groups are stored lowercased. A
    # var with the prefix but an empty name (e.g. WALLATAG_FOCUS_KEYWORDS) is a
    # ConfigError; WALLATAG_FOCUS_<NAME> without a suffix is ignored.
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
        elif key.endswith(_ENV_FOCUS_KEYWORDS_REGEX_SUFFIX):
            field = "keywords_regex"
            name = key[
                len(_ENV_FOCUS_PREFIX) : len(key)
                - len(_ENV_FOCUS_KEYWORDS_REGEX_SUFFIX)
            ]
        elif key.endswith(_ENV_FOCUS_TAGS_SUFFIX):
            field = "tags"
            name = key[len(_ENV_FOCUS_PREFIX) : len(key) - len(_ENV_FOCUS_TAGS_SUFFIX)]
        elif key.endswith(_ENV_FOCUS_FIELDS_SUFFIX):
            field = "fields"
            name = key[len(_ENV_FOCUS_PREFIX) : len(key) - len(_ENV_FOCUS_FIELDS_SUFFIX)]
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
        elif field == "keywords_regex":
            group = replace(
                group,
                keywords_regex=_validate_regexes(
                    "focus group",
                    key,
                    _parse_comma_separated("focus group", key, env[key]),
                ),
            )
        elif field == "tags":
            group = replace(
                group,
                tags=_parse_comma_separated("focus group", key, env[key]),
            )
        else:  # field == "fields"
            group = replace(
                group,
                fields=_validate_match_fields(
                    "focus group",
                    key,
                    _parse_comma_separated("focus group", key, env[key]),
                ),
            )
        focus[storage_key] = group
        focus_changed = True
    if focus_changed:
        tagger = replace(tagger, focus_groups=focus)

    if (
        wallabag is config.wallabag
        and store is config.store
        and ai is config.ai
        and vocabulary is config.vocabulary
        and tagger is config.tagger
    ):
        return config
    return replace(
        config,
        wallabag=wallabag,
        store=store,
        ai=ai,
        vocabulary=vocabulary,
        tagger=tagger,
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
