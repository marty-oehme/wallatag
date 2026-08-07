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

# Environment variable -> config field mapping.
_ENV_URL = "WALLATAG_URL"
_ENV_CLIENT_ID = "WALLATAG_CLIENT_ID"
_ENV_CLIENT_SECRET = "WALLATAG_CLIENT_SECRET"
_ENV_DB = "WALLATAG_DB"
_ENV_CONFIG = "WALLATAG_CONFIG"

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


@dataclass(frozen=True)
class StoreConfig:
    path: str | None = None  # None = history-less (no DB at all)


@dataclass(frozen=True)
class TaggerConfig:
    max_suggestions: int = 5
    tag_policy: str = "prefer-existing"
    focus_groups: dict[str, FocusGroup] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    wallabag: WallabagConfig = field(default_factory=WallabagConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
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


def _parse_focus_string_list(group_name: str, key: str, value: object) -> tuple[str, ...]:
    """Validate one focus-group keyword/tag list (list/tuple of strings)."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"focus group {group_name!r}: {key} must be a list of strings"
        )
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(
                f"focus group {group_name!r}: {key} contains a non-string value {item!r}"
            )
    return tuple(value)


def _parse_focus_groups(raw: dict) -> dict[str, FocusGroup]:
    groups: dict[str, FocusGroup] = {}
    for name, section in raw.items():
        if not isinstance(section, dict):
            raise ConfigError(
                f"focus group {name!r} must be a table with keywords and tags"
            )
        keywords = (
            _parse_focus_string_list(name, "keywords", section["keywords"])
            if "keywords" in section
            else ()
        )
        tags = (
            _parse_focus_string_list(name, "tags", section["tags"])
            if "tags" in section
            else ()
        )
        groups[name] = FocusGroup(keywords=keywords, tags=tags)
    return groups


def _parse_toml_config(raw: dict) -> Config:
    wallabag_raw = raw.get("wallabag", {}) or {}
    store_raw = raw.get("store", {}) or {}
    tagger_raw = raw.get("tagger", {}) or {}

    if not isinstance(wallabag_raw, dict) or not isinstance(store_raw, dict) or not isinstance(tagger_raw, dict):
        raise ConfigError("sections [wallabag], [store], [tagger] must be tables")

    # [ai] is reserved for phase 2 and is ignored unconditionally: even if it
    # is present but not a table, it never raises.

    tag_policy = tagger_raw.get("tag_policy", "prefer-existing")
    if tag_policy not in VALID_TAG_POLICIES:
        raise ConfigError(
            f"invalid tag_policy {tag_policy!r}; valid choices: {', '.join(VALID_TAG_POLICIES)}"
        )

    max_suggestions = tagger_raw.get("max_suggestions", 5)
    if not isinstance(max_suggestions, int) or isinstance(max_suggestions, bool) or max_suggestions < 0:
        raise ConfigError("max_suggestions must be a non-negative integer")

    focus_raw = raw.get("focus", {}) or {}
    if not isinstance(focus_raw, dict):
        raise ConfigError("section [focus] must be a table")

    return Config(
        wallabag=WallabagConfig(
            url=str(wallabag_raw.get("url", "") or ""),
            client_id=str(wallabag_raw.get("client_id", "") or ""),
            client_secret=str(wallabag_raw.get("client_secret", "") or ""),
        ),
        store=StoreConfig(path=str(store_raw.get("path") or "") or None),
        tagger=TaggerConfig(
            max_suggestions=max_suggestions,
            tag_policy=tag_policy,
            focus_groups=_parse_focus_groups(focus_raw),
        ),
        verbose=False,
    )


def _apply_env(config: Config, env: Mapping[str, str]) -> Config:
    """Overlay environment variables on top of a parsed config."""
    wallabag = config.wallabag
    url = env.get(_ENV_URL)
    client_id = env.get(_ENV_CLIENT_ID)
    client_secret = env.get(_ENV_CLIENT_SECRET)
    if url is not None:
        wallabag = replace(wallabag, url=url)
    if client_id is not None:
        wallabag = replace(wallabag, client_id=client_id)
    if client_secret is not None:
        wallabag = replace(wallabag, client_secret=client_secret)

    store = config.store
    db = env.get(_ENV_DB)
    if db is not None:
        store = replace(store, path=db or None)  # empty string -> history-less

    if wallabag is config.wallabag and store is config.store:
        return config
    return replace(config, wallabag=wallabag, store=store)


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
