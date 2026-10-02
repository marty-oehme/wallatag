"""Prefect block definitions for wallatag (optional, Prefect-only surface).

Nothing in this module is imported by the wallatag package, which keeps zero
Prefect dependency and reads its config from TOML + env vars. The blocks below
are an ADDITIONAL configuration source for Prefect-scheduled runs: they
provide WALLATAG_AI_* / WALLATAG_* DEFAULTS, and container env vars (dokku
config:set) take precedence over block fields for scheduled runs too.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable

from prefect.blocks.core import Block
from pydantic import Field, SecretStr

BLOCK_NAME = "wallatag-llm"
WALLABAG_BLOCK_NAME = "wallabag"


class LLMCredentials(Block):
    """LLM URL/credentials for Prefect-scheduled wallatag runs.

    Seeded from the ``WALLATAG_AI_*`` env vars on first release (see
    deploy/release.py); an empty block is valid so it can be filled in later
    from the Prefect UI. The wallatag CLI keeps reading config/env; non-empty
    block fields are used as DEFAULTS for scheduled runs, and WALLATAG_AI_*
    environment variables override them. The confidence threshold is NOT a
    block field: it is owned by the ``wallatag_ai_confidence_threshold``
    Prefect Variable (config vs secrets vs credentials split, git-bug
    db3f009).
    """

    _block_type_name = "Wallatag LLM Credentials"
    _block_type_slug = "wallatag-llm-credentials"

    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key: SecretStr = Field(default_factory=partial(SecretStr, ""))

    def llm_env(self) -> dict[str, str]:
        """Map non-empty fields to WALLATAG_AI_* env vars (defaults).

        An empty block yields {} (no defaults provided), so scheduled runs
        fall back to the TOML config / container env. Never logs or prints the
        api key.
        """
        env: dict[str, str] = {}
        if self.provider:
            env["WALLATAG_AI_PROVIDER"] = self.provider
        if self.base_url:
            env["WALLATAG_AI_BASE_URL"] = self.base_url
        if self.model:
            env["WALLATAG_AI_MODEL"] = self.model
        if self.api_key.get_secret_value():
            env["WALLATAG_AI_API_KEY"] = self.api_key.get_secret_value()
        return env


class WallabagCredentials(Block):
    """Credentials for a Wallabag instance (OAuth2 password grant).

    ``base_url`` is the Wallabag server root; ``client_id`` / ``client_secret``
    come from the Wallabag API client settings and ``username`` / ``password``
    are the account to read articles with. Seeded from the ``WALLATAG_*`` env
    vars on first release (see deploy/release.py); an empty block is valid so
    it can be filled in later from the Prefect UI. The block document is named
    ``wallabag`` and shared with the morning-digest project: both projects
    register the same block type, so one document serves both. The wallatag
    CLI keeps reading config/env; non-empty block fields are used as DEFAULTS
    for scheduled runs, and WALLATAG_* environment variables override them.
    """

    _block_type_name = "Wallabag Credentials"
    _block_type_slug = "wallabag-credentials"
    base_url: str = ""
    client_id: str = ""
    client_secret: SecretStr = Field(default_factory=partial(SecretStr, ""))
    username: str = ""
    password: SecretStr = Field(default_factory=partial(SecretStr, ""))

    def wallabag_env(self) -> dict[str, str]:
        """Map non-empty fields to WALLATAG_* env vars (defaults).

        The wallatag CLI reads these vars directly (see wallatag/config.py);
        an empty block yields {} (no defaults provided), so scheduled runs
        fall back to the TOML config / container env. Never logs or prints the
        client secret or password.
        """
        env: dict[str, str] = {}
        if self.base_url:
            env["WALLATAG_URL"] = self.base_url
        if self.client_id:
            env["WALLATAG_CLIENT_ID"] = self.client_id
        if self.client_secret.get_secret_value():
            env["WALLATAG_CLIENT_SECRET"] = (
                self.client_secret.get_secret_value()
            )
        if self.username:
            env["WALLATAG_USERNAME"] = self.username
        if self.password.get_secret_value():
            env["WALLATAG_PASSWORD"] = self.password.get_secret_value()
        return env


def ensure_wallatag_llm_credentials_block(
    block_name: str = BLOCK_NAME,
    log: Callable[[str], None] | None = None,
) -> None:
    """Idempotently ensure the LLMCredentials block document exists.

    Mirrors ensure_miniflux_credentials_block in morning-digest. If the block
    document already exists, do nothing. Otherwise seed it from the
    WALLATAG_AI_* env vars when the provider/base_url/model trio is set (also
    taking the api key when present), else create an empty block that can be
    filled in later from the Prefect UI. Seeding is a one-time snapshot of the
    env: at run time the block only provides defaults, and WALLATAG_AI_* env
    vars still override its fields. The confidence threshold is not seeded
    here: it is owned by the wallatag_ai_confidence_threshold Prefect
    Variable. A save failure is logged as a warning and does not propagate:
    the release phase must not fail over a cosmetic block issue.
    """
    log = log or (lambda message: print(f"[blocks] {message}", flush=True))
    try:
        LLMCredentials.load(block_name)
    except Exception:
        provider = os.environ.get("WALLATAG_AI_PROVIDER")
        base_url = os.environ.get("WALLATAG_AI_BASE_URL")
        model = os.environ.get("WALLATAG_AI_MODEL")
        if provider and base_url and model:
            api_key = os.environ.get("WALLATAG_AI_API_KEY")
            block = LLMCredentials(
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=SecretStr(api_key or ""),
            )
            log(f"seeding {block_name!r} block from WALLATAG_AI_* env")
        else:
            block = LLMCredentials()
            log(
                f"creating empty {block_name!r} block "
                f"(no WALLATAG_AI_* trio in env)"
            )
        try:
            block.save(block_name)
        except Exception as exc:
            log(f"warning: could not save {block_name!r} block: {exc}")
            return
        log(f"created {block_name!r} block")
    else:
        log(f"{block_name!r} block already exists")


def ensure_wallabag_credentials_block(
    block_name: str = WALLABAG_BLOCK_NAME,
    log: Callable[[str], None] | None = None,
) -> None:
    """Idempotently ensure the WallabagCredentials block document exists.

    Mirrors ensure_wallabag_credentials_block in morning-digest: the block
    document name ``wallabag`` is shared between the two projects. If the
    block document already exists, do nothing. Otherwise seed it from the
    WALLATAG_* env vars when all five (url, client id, client secret,
    username, password) are set, else create an empty block that can be filled
    in later from the Prefect UI. Seeding is a one-time snapshot of the env:
    at run time the block only provides defaults, and WALLATAG_* env vars
    still override its fields. A save failure is logged as a warning and does
    not propagate: the release phase must not fail over a cosmetic block issue.
    """
    log = log or (lambda message: print(f"[blocks] {message}", flush=True))
    try:
        WallabagCredentials.load(block_name)
    except Exception:
        base_url = os.environ.get("WALLATAG_URL")
        client_id = os.environ.get("WALLATAG_CLIENT_ID")
        client_secret = os.environ.get("WALLATAG_CLIENT_SECRET")
        username = os.environ.get("WALLATAG_USERNAME")
        password = os.environ.get("WALLATAG_PASSWORD")
        if base_url and client_id and client_secret and username and password:
            block = WallabagCredentials(
                base_url=base_url,
                client_id=client_id,
                client_secret=SecretStr(client_secret),
                username=username,
                password=SecretStr(password),
            )
            log(f"seeding {block_name!r} block from WALLATAG_* env")
        else:
            block = WallabagCredentials()
            log(
                f"creating empty {block_name!r} block "
                f"(no WALLATAG_* quintet in env)"
            )
        try:
            block.save(block_name)
        except Exception as exc:
            log(f"warning: could not save {block_name!r} block: {exc}")
            return
        log(f"created {block_name!r} block")
    else:
        log(f"{block_name!r} block already exists")
