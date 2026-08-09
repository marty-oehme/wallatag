"""Prefect block definitions for wallatag (optional, Prefect-only surface).

Nothing in this module is imported by the wallatag package, which keeps zero
Prefect dependency and reads its config from TOML + env vars. The block below
is an ADDITIONAL configuration source for Prefect-scheduled runs: non-empty
block fields override the container env for scheduled runs only.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable

from prefect.blocks.core import Block
from pydantic import Field, SecretStr

BLOCK_NAME = "wallatag-llm"


class LLMCredentials(Block):
    """LLM URL/credentials for Prefect-scheduled wallatag runs.

    Seeded from the ``WALLATAG_AI_*`` env vars on first release (see
    deploy/release.py); an empty block is valid so it can be filled in later
    from the Prefect UI. The wallatag CLI keeps reading config/env; this block
    only overrides non-empty fields.
    """

    _block_type_name = "Wallatag LLM Credentials"
    _block_type_slug = "wallatag-llm-credentials"

    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key: SecretStr = Field(default_factory=partial(SecretStr, ""))
    confidence_threshold: float | None = None

    def llm_env(self) -> dict[str, str]:
        """Map non-empty fields to WALLATAG_AI_* env vars.

        An empty block yields {} (nothing overridden), so scheduled runs fall
        back to the TOML config / container env. Never logs or prints the api
        key.
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
        if self.confidence_threshold is not None:
            env["WALLATAG_AI_CONFIDENCE_THRESHOLD"] = str(
                self.confidence_threshold
            )
        return env


def ensure_wallatag_llm_credentials_block(
    block_name: str = BLOCK_NAME,
    log: Callable[[str], None] | None = None,
) -> None:
    """Idempotently ensure the LLMCredentials block document exists.

    Mirrors ensure_miniflux_credentials_block in morning-digest. If the block
    document already exists, do nothing. Otherwise seed it from the
    WALLATAG_AI_* env vars when the provider/base_url/model trio is set (also
    taking the api key and confidence threshold when present), else create an
    empty block that can be filled in later from the Prefect UI. A save
    failure is logged as a warning and does not propagate: the release phase
    must not fail over a cosmetic block issue.
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
            threshold_raw = os.environ.get("WALLATAG_AI_CONFIDENCE_THRESHOLD")
            threshold: float | None = None
            if threshold_raw:
                try:
                    threshold = float(threshold_raw)
                except ValueError:
                    log(
                        f"warning: WALLATAG_AI_CONFIDENCE_THRESHOLD "
                        f"{threshold_raw!r} is not a number; leaving the "
                        f"threshold unset in the block"
                    )
                if threshold is not None and not (0.0 < threshold <= 1.0):
                    log(
                        f"warning: WALLATAG_AI_CONFIDENCE_THRESHOLD "
                        f"{threshold_raw!r} is out of range (0, 1]; leaving "
                        f"the threshold unset in the block"
                    )
                    threshold = None
            block = LLMCredentials(
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=SecretStr(api_key) if api_key else "",
                confidence_threshold=threshold,
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
