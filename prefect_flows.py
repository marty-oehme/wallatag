"""Prefect flows for wallatag.

Prefect is the sole scheduler for wallatag; this module is never imported by
the wallatag package, which keeps zero Prefect dependency. The flow is executed
by a Prefect worker that runs as a process inside the Dokku container where
wallatag is installed and configured (git-bug issue 3d0b22f).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from blocks import (
    BLOCK_NAME,
    WALLABAG_BLOCK_NAME,
    LLMCredentials,
    WallabagCredentials,
)
from prefect import flow
from prefect.variables import Variable

if TYPE_CHECKING:
    from prefect.flows import Flow

# Scalar settings that can be overridden per project/deployment through the
# Prefect UI (Variables page). Names mirror the env vars EXACTLY, so each value
# is passed straight through to the wallatag CLI as a WALLATAG_* env var.
# Secrets (WALLATAG_CLIENT_SECRET, WALLATAG_PASSWORD, WALLATAG_AI_API_KEY) are
# deliberately absent: they stay in container env vars / credentials blocks.
WALLATAG_VARIABLES: tuple[str, ...] = (
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
)

# Focus groups as ONE JSON variable, translated to the existing
# WALLATAG_FOCUS_<NAME>_KEYWORDS/_TAGS/_FIELDS/_KEYWORDS_REGEX env convention.
FOCUS_GROUPS_VARIABLE = "WALLATAG_FOCUS_GROUPS"


def build_wallatag_command(
    max_articles: int,
    tag_policy: str | None,
    focus: str | None,
) -> list[str]:
    """Build the `wallatag run` command line for a batch.

    ``focus`` is a comma-separated list of focus-group names (e.g.
    ``"methods, languages"``); each name is emitted as its own ``--focus``
    flag, in order. Segments are stripped; empty/whitespace-only segments are
    dropped; a value with no usable names (or None) emits no ``--focus``
    flags at all. Comma-splitting lives here in the flow layer on purpose:
    the CLI never splits on commas, because a TOML focus-group NAME may
    itself contain a comma (such groups can only be selected via the CLI,
    not via the flow parameter).
    """
    cmd = ["wallatag", "run", "--max", str(max_articles)]
    if tag_policy is not None:
        cmd += ["--tag-policy", tag_policy]
    if focus is not None:
        for name in focus.split(","):
            name = name.strip()
            if name:
                cmd += ["--focus", name]
    return cmd


def llm_env_from_block() -> dict[str, str]:
    """Return WALLATAG_AI_* defaults from the wallatag-llm block, if any.

    The block provides defaults for scheduled runs; container env vars (dokku
    config:set) override them for the same variable. Fail-open: any error (no
    Prefect server, missing block, network) logs a warning and returns {}, so
    scheduled runs fall back to the container env / wallatag.toml.
    """
    try:
        return LLMCredentials.load(BLOCK_NAME).llm_env()
    except Exception:
        print(
            f"LLM credentials block {BLOCK_NAME!r} not available, "
            f"falling back to config/env",
            flush=True,
        )
        return {}


def wallabag_env_from_block() -> dict[str, str]:
    """Return WALLATAG_* defaults from the wallabag-credentials block, if any.

    The block provides defaults for scheduled runs; container env vars (dokku
    config:set) override them for the same variable. Fail-open: any error (no
    Prefect server, missing block, network) logs a warning and returns {}, so
    scheduled runs fall back to the container env / wallatag.toml.
    """
    try:
        return WallabagCredentials.load(WALLABAG_BLOCK_NAME).wallabag_env()
    except Exception:
        print(
            f"wallabag credentials block {WALLABAG_BLOCK_NAME!r} not "
            f"available, falling back to config/env",
            flush=True,
        )
        return {}


def variable_env() -> dict[str, str]:
    """Return WALLATAG_* settings from Prefect Variables, if any.

    Each name in WALLATAG_VARIABLES is read as a variable (managed in the
    Prefect UI, no redeploy needed) and normalized to the env-var string the
    wallatag CLI expects: bools become "true"/"false", everything else is
    str()'d. Unset variables are skipped, so a partial set falls through to
    the container env / wallatag.toml for the rest. Fail-open like the block
    helpers: Variable.get hits the Prefect API, so without a server (or on a
    mid-loop error) the values read so far are kept, a warning is logged
    once, and the remaining names fall through to the container env / TOML.
    """
    env: dict[str, str] = {}
    for name in WALLATAG_VARIABLES:
        try:
            value = Variable.get(name, default=None)
        except Exception:
            print(
                "prefect variables not available, falling back to config/env",
                flush=True,
            )
            break
        if value is None:
            continue
        if isinstance(value, bool):
            env[name] = "true" if value else "false"
        else:
            env[name] = str(value)
    return env


def _parse_focus_groups(raw: object) -> dict[str, dict[str, list[str]]]:
    """Validate the WALLATAG_FOCUS_GROUPS variable value (fail loud).

    Expects a JSON object (or already-parsed dict) mapping group names to
    ``{"keywords": [...], "tags": [...], "fields": [...], "keywords_regex":
    [...]}`` (all four keys optional; values are lists of non-empty strings).
    Group names must be strings and unique case-insensitively. ``keywords_regex``
    items must additionally be compilable regexes and must not contain a literal
    comma (the comma-separated env translation would silently split them, so
    comma-containing patterns are rejected fail-loud here — use TOML
    ``keywords_regex`` instead). Any violation raises a
    RuntimeError naming the variable and the offending group/key so a typo
    (e.g. "keywrods") is caught instead of silently changing tagging.
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{FOCUS_GROUPS_VARIABLE}: expected a JSON object mapping "
                f"group names to {{keywords, tags, fields, keywords_regex}}, "
                f"got invalid JSON: {exc}"
            ) from exc
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"{FOCUS_GROUPS_VARIABLE}: expected a JSON object (dict) mapping "
            f"group names to {{keywords, tags, fields, keywords_regex}}, got "
            f"{type(parsed).__name__}"
        )
    allowed = ("keywords", "tags", "fields", "keywords_regex")
    # Two group names that differ only in case (e.g. "methods" and "Methods")
    # would both map to WALLATAG_FOCUS_methods_* env vars and one would be
    # silently dropped, so they are rejected up front (wallatag itself rejects
    # them the same way for TOML focus tables, config.py merge-by-casefold).
    seen_casefold: dict[str, str] = {}
    groups: dict[str, dict[str, list[str]]] = {}
    for name, group in parsed.items():
        if not isinstance(name, str):
            raise RuntimeError(
                f"{FOCUS_GROUPS_VARIABLE}: group names must be strings, got "
                f"{type(name).__name__}: {name!r}"
            )
        folded = name.casefold()
        previous = seen_casefold.get(folded)
        if previous is not None:
            raise RuntimeError(
                f"{FOCUS_GROUPS_VARIABLE}: focus group names must be unique "
                f"case-insensitively: {previous!r} vs {name!r}"
            )
        seen_casefold[folded] = name
        if not isinstance(group, dict):
            raise RuntimeError(
                f"{FOCUS_GROUPS_VARIABLE}: group {name!r} must be a dict "
                f"with optional keys keywords/tags/fields/keywords_regex, got "
                f"{type(group).__name__}"
            )
        for key in group:
            if key not in allowed:
                raise RuntimeError(
                    f"{FOCUS_GROUPS_VARIABLE}: group {name!r} has unknown "
                    f"key {key!r}; allowed keys: keywords, tags, fields, "
                    f"keywords_regex"
                )
        cleaned: dict[str, list[str]] = {}
        for key in allowed:
            if key not in group:
                continue
            items = group[key]
            if not isinstance(items, list):
                raise RuntimeError(
                    f"{FOCUS_GROUPS_VARIABLE}: group {name!r} field {key!r} "
                    f"must be a list of strings, got {type(items).__name__}"
                )
            stripped_items: list[str] = []
            for item in items:
                if not isinstance(item, str):
                    raise RuntimeError(
                        f"{FOCUS_GROUPS_VARIABLE}: group {name!r} field "
                        f"{key!r} must contain only strings, got "
                        f"{type(item).__name__}"
                    )
                stripped = item.strip()
                if not stripped:
                    raise RuntimeError(
                        f"{FOCUS_GROUPS_VARIABLE}: group {name!r} field "
                        f"{key!r} contains an empty or whitespace-only string"
                    )
                stripped_items.append(stripped)
            # Fail fast on a bad regex: the UI shows the reason instead of only
            # failing when the wallatag subprocess exits 2. Un-compilable
            # patterns are one case; patterns containing a literal comma are
            # the other — the comma-separated env translation would silently
            # split them, so they must use TOML keywords_regex instead.
            if key == "keywords_regex":
                for item in stripped_items:
                    try:
                        re.compile(item)
                    except re.error as exc:
                        raise RuntimeError(
                            f"{FOCUS_GROUPS_VARIABLE}: group {name!r} field "
                            f"keywords_regex contains invalid regex {item!r}: {exc}"
                        ) from exc
                    if "," in item:
                        raise RuntimeError(
                            f"{FOCUS_GROUPS_VARIABLE}: group {name!r} field "
                            f"keywords_regex contains a comma in pattern {item!r}; "
                            f"the comma-separated env translation cannot "
                            f"represent it — use TOML keywords_regex"
                        )
            cleaned[key] = stripped_items
        groups[name] = cleaned
    return groups


def focus_groups_env() -> dict[str, str]:
    """Return WALLATAG_FOCUS_<NAME>_* env vars from the WALLATAG_FOCUS_GROUPS
    Prefect Variable, if any.

    The variable is a JSON object mapping group names to
    ``{"keywords": [...], "tags": [...], "fields": [...], "keywords_regex":
    [...]}`` (all four keys optional). It is translated to the env convention
    wallatag/config.py already parses:
    ``WALLATAG_FOCUS_<NAME>_KEYWORDS/_TAGS/_FIELDS/_KEYWORDS_REGEX``
    (comma-separated, group name lowercased). ``keywords_regex`` patterns
    containing a literal comma are rejected in ``_parse_focus_groups`` because
    this comma-join would silently split them; empty/absent keywords, tags or
    keywords_regex omit that env var entirely; a present ``fields`` key is
    always emitted —
    an empty list emits ``WALLATAG_FOCUS_<NAME>_FIELDS = ""``, which wallatag
    parses as fields=() and disables that group (matching the per-source-fields
    feature). Groups are iterated in sorted() order for determinism. Values
    are validated strictly (see _parse_focus_groups): malformed data raises
    RuntimeError (fail loud). Only the API read itself fails open (no Prefect
    server) with a warning, so scheduled runs fall back to env/TOML.
    """
    try:
        raw = Variable.get(FOCUS_GROUPS_VARIABLE, default=None)
    except Exception:
        print(
            "prefect variables not available, falling back to config/env",
            flush=True,
        )
        return {}
    if raw is None:
        return {}
    groups = _parse_focus_groups(raw)
    env: dict[str, str] = {}
    for name in sorted(groups):
        prefix = f"WALLATAG_FOCUS_{name.lower()}"
        keywords = groups[name].get("keywords")
        if keywords:
            env[f"{prefix}_KEYWORDS"] = ",".join(keywords)
        keywords_regex = groups[name].get("keywords_regex")
        if keywords_regex:
            env[f"{prefix}_KEYWORDS_REGEX"] = ",".join(keywords_regex)
        tags = groups[name].get("tags")
        if tags:
            env[f"{prefix}_TAGS"] = ",".join(tags)
        if "fields" in groups[name]:
            env[f"{prefix}_FIELDS"] = ",".join(groups[name]["fields"])
    return env


@flow(log_prints=True)
def wallatag_batch(
    max_articles: int = 50,
    tag_policy: str | None = None,
    focus: str | None = None,
) -> str:
    """Run one headless wallatag batch against the wallabag API.

    The flow executes inside the Dokku container, so it shells out to the
    installed `wallatag` console script (which reads WALLATAG_* env vars).
    Settings merge with the following precedence (lowest to highest):
    wallatag.toml defaults → the auto-created blocks (blocks.py), which
    provide defaults for scheduled runs: the wallatag-llm block (LLM
    settings) and the wallabag-credentials block (wallabag URL/credentials)
    → container env vars (dokku config:set) → Prefect Variables (the scalar
    settings in WALLATAG_VARIABLES plus the WALLATAG_FOCUS_GROUPS JSON,
    managed in the Prefect UI) → CLI options (--tag-policy/--focus). So a
    Prefect Variable overrides the container env var and the block for that
    setting; secrets (client_secret/password/api_key) never come from
    variables; missing variables fall through to the container env / TOML;
    and CLI flags still win for --tag-policy/--focus. Empty block fields fall
    back to TOML/env. The flow's ``focus`` parameter is a comma-separated
    list of group names, each emitted as its own ``--focus`` flag (a group
    whose NAME contains a literal comma is unreachable from the flow — use
    the CLI for those).
    Returns the captured stdout on success. Raises on a non-zero exit so
    Prefect marks the run Failed and can notify on problems.
    """
    if shutil.which("wallatag") is None:
        raise RuntimeError(
            "wallatag console script not found on PATH; is the package installed?"
        )
    env = {
        **llm_env_from_block(),
        **wallabag_env_from_block(),
        **os.environ,
        **variable_env(),
        **focus_groups_env(),
    }
    try:
        completed = subprocess.run(
            build_wallatag_command(max_articles, tag_policy, focus),
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("wallatag run timed out after 1800s") from exc
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"wallatag run exited with code {completed.returncode}: {detail}"
        )
    return completed.stdout.strip()
