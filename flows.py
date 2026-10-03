"""Prefect flows for wallatag.

Prefect is the sole scheduler for wallatag; this module is never imported by
the wallatag package, which keeps zero Prefect dependency. The flow is executed
by a Prefect worker that runs as a process inside the Dokku container where
wallatag is installed and configured (git-bug issue 3d0b22f).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import cast

import requests

from blocks import (
    BLOCK_NAME,
    WALLABAG_BLOCK_NAME,
    LLMCredentials,
    WallabagCredentials,
)
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.variables import Variable
from wallatag import auto
from wallatag.cli import _build_tagger
from wallatag.config import apply_run_overrides, load_config
from wallatag.store import Store
from wallatag.wallabag import WallabagClient, WallabagError

logger = logging.getLogger(__name__)

# Scalar settings that can be overridden per project/deployment through the
# Prefect UI (Variables page). Names mirror the env vars EXACTLY, LOWERCASED:
# Prefect requires variable names to be lowercase, so each name here is the
# lowercase form of a WALLATAG_* env var; variable_env() derives the env-var
# name via name.upper(), so variable and env names can never drift. Secrets
# (WALLATAG_CLIENT_SECRET, WALLATAG_PASSWORD, WALLATAG_AI_API_KEY) are
# deliberately absent: they stay in container env vars / credentials blocks.
WALLATAG_VARIABLES: tuple[str, ...] = (
    "wallatag_tag_policy",
    "wallatag_max_applied_tags",
    "wallatag_ignore_tags",
    "wallatag_ignore_tags_regex",
    "wallatag_enable_vocabulary",
    "wallatag_enable_rules",
    "wallatag_enable_llm",
    "wallatag_ai_confidence_threshold",
    "wallatag_ai_use_focus_groups",
    "wallatag_ai_max_proposals",
    "wallatag_vocabulary_fields",
    "wallatag_vocabulary_skip_ignored_tags",
)

# Focus groups as ONE JSON variable (lowercased like the scalars), translated
# to the existing WALLATAG_FOCUS_<NAME>_KEYWORDS/_TAGS/_FIELDS/_KEYWORDS_REGEX
# env convention.
FOCUS_GROUPS_VARIABLE = "wallatag_focus_groups"


# Per-article outcomes that FAIL the tag-article task run
# (auto.EntryResult.outcome), mapped to the human-readable reason shown in
# the run log and the task-run state message. "tagging failed" is add_tags
# exhaustion; "llm_failed" is the tagger erroring with nothing to apply. All
# other outcomes (tagged, skipped, no suggestions) are normal Completed
# outcomes.
TAGGING_FAILURE_REASONS: dict[str, str] = {
    "llm_failed": "LLM tagging failed",
    "tagging failed": "tagging failed",
}


class TaggingFailedError(RuntimeError):
    """A tag-article task run failed to tag its article.

    Raised when the engine reports a per-article error outcome — the tagger
    failed (``llm_failed``) or the final add_tags POST exhausted its engine
    retries (``tagging failed``). Carries the engine's ``EntryResult`` so the
    flow can keep summary accounting honest while Prefect marks the task run
    Failed. It is NOT a feed error: the flow catches it and continues the
    batch.
    """

    def __init__(self, result: auto.EntryResult):
        reason = TAGGING_FAILURE_REASONS.get(result.outcome, "tagging failed")
        super().__init__(f"{reason} article {result.entry_id}")
        self.result = result


# This task takes the shared, non-serializable runtime objects (client, tagger,
# store holding a sqlite3 connection, cfg, fallback_tagger) by reference, so the
# default cache policy's input hashing raises HashError when computing the cache
# key. Cache is disabled: per-article results must never be cache-reused —
# pick-up dedupe is the Store's job.
@task(name="tag-article", cache_policy=NO_CACHE)
def tag_article(
    entry,
    *,
    client,
    tagger,
    store,
    cfg,
    fallback_tagger=None,
) -> auto.EntryResult:
    """One Prefect task per candidate article: drive the engine in-process.

    Thin adapter over ``auto.process_entry`` (dry-run off). The shared
    client/tagger/store are stateful and not parallel-safe, so calls are
    strictly sequential; each task run shows the article's own logs/timing
    in the dashboard. When the article is tagged, the task logs one
    run-attributed INFO line via ``get_run_logger()`` — article id, title and
    the applied tags — so the UI/DB shows what each task run applied. When
    the engine reports a per-article error outcome — add_tags retries
    exhausted (``tagging failed``) or the tagger erroring with nothing to
    apply (``llm_failed``) — the task logs one run-attributed ERROR line and
    raises ``TaggingFailedError``, so final tagging/LLM failures surface as
    Failed task runs. Both lines must use the run logger,
    not the plain module logger: the engine's own per-article lines
    (wallatag.auto) never surface in the flow because the Prefect-installed
    root handler is at WARNING and ``logging.basicConfig`` is a no-op here
    (bug edf2338).
    """
    result = auto.process_entry(
        client,
        tagger,
        store,
        cfg,
        entry,
        dry_run=False,
        fallback_tagger=fallback_tagger,
    )
    if result.outcome == "tagged":
        get_run_logger().info(
            "tagged article %s (%s): %s",
            result.entry_id,
            entry.get("title", ""),
            ", ".join(result.tags),
        )
    elif result.outcome in TAGGING_FAILURE_REASONS:
        get_run_logger().error(
            "%s article %s (%s)",
            TAGGING_FAILURE_REASONS[result.outcome],
            result.entry_id,
            entry.get("title", ""),
        )
        raise TaggingFailedError(result)
    return result


def llm_env_from_block() -> dict[str, str]:
    """Return WALLATAG_AI_* defaults from the wallatag-llm block, if any.

    The block provides defaults for scheduled runs; container env vars (dokku
    config:set) override them for the same variable. Fail-open: any error (no
    Prefect server, missing block, network) logs a warning and returns {}, so
    scheduled runs fall back to the container env / wallatag.toml.
    """
    try:
        # Block.load is async_dispatch-typed: Self | Coroutine[...]. Prefect
        # only picks the coroutine implementation from an async context, and
        # these helpers are synchronous, so the cast is a no-op at runtime.
        block = cast(LLMCredentials, LLMCredentials.load(BLOCK_NAME))
        return block.llm_env()
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
        # async_dispatch-typed like llm_env_from_block; always the sync branch.
        block = cast(
            WallabagCredentials,
            WallabagCredentials.load(WALLABAG_BLOCK_NAME),
        )
        return block.wallabag_env()
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
    wallatag CLI expects: the env-var name is derived mechanically from the
    variable name via ``name.upper()`` (so variable and env names can never
    drift), and bools become "true"/"false", everything else is str()'d.
    Unset variables are skipped, so a partial set falls through to
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
            env[name.upper()] = "true" if value else "false"
        else:
            env[name.upper()] = str(value)
    return env


def _parse_focus_groups(raw: object) -> dict[str, dict[str, list[str]]]:
    """Validate the wallatag_focus_groups variable value (fail loud).

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
    """Return WALLATAG_FOCUS_<NAME>_* env vars from the wallatag_focus_groups
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
    focus: str | None = None,
) -> str:
    """Run one headless wallatag batch against the wallabag API.

    The flow executes inside the Dokku container and drives the wallatag
    tagging engine IN-PROCESS (no subprocess): one Prefect task per candidate
    article (``tag-article``), sharing the same engine code as the CLI
    (``wallatag run``) — two drivers, one engine. Settings merge with the
    following precedence (lowest to highest): wallatag.toml defaults → the
    auto-created blocks (blocks.py), which provide defaults for scheduled
    runs: the wallatag-llm block (LLM settings) and the wallabag-credentials
    block (wallabag URL/credentials) → container env vars (dokku config:set)
    → Prefect Variables (the scalar settings in WALLATAG_VARIABLES — lowercase
    names the flow uppercases into the WALLATAG_* env vars — plus the
    wallatag_focus_groups JSON, managed in the Prefect UI) → the flow's
    ``focus`` parameter (--focus equivalent). So a Prefect Variable overrides
    the container env var and the block for that setting; secrets
    (client_secret/password/api_key) never come from variables; missing
    variables fall through to the container env / TOML; and the ``focus``
    parameter still wins for group selection. The tag policy is owned by the
    wallatag_tag_policy variable (managed in the Prefect UI): it lands in the
    merged env / config, with container env / TOML as fallbacks — there is no
    --tag-policy flow parameter. Empty block fields fall back to TOML/env.
    The flow's ``focus`` parameter is a comma-separated list of group names
    (split on commas by the flow itself: segments are stripped and
    empty/whitespace-only segments dropped); a group whose NAME contains a
    literal comma is unreachable from the flow — use the CLI for those.

    Per-article work runs as sequential ``tag-article`` tasks (the shared
    client/tagger/store are stateful and not parallel-safe), so each article's
    logs and timing appear as its own task run in the dashboard. The engine
    itself has no run timeout (the old subprocess 1800s timeout was for the
    child process; Prefect's own run timeout applies to the flow). Returns the
    summary line (the same one the CLI prints), and raises RuntimeError when
    the article feed fails entirely (feed error with nothing presented) or
    when every presented article failed to be tagged (e.g. the LLM is down),
    so Prefect marks such a run Failed and can notify on problems; a partial
    run (some articles tagged or normally skipped) still returns its summary.
    """
    if max_articles == 0:
        return ""
    env = {
        **llm_env_from_block(),
        **wallabag_env_from_block(),
        **os.environ,
        **variable_env(),
        **focus_groups_env(),
    }
    focus_names = (
        None
        if focus is None
        else [name.strip() for name in focus.split(",") if name.strip()]
        or None
    )
    config = apply_run_overrides(
        load_config(env=env),
        max_articles=max_articles,
        focus=focus_names,
    )
    # Mirror cmd_run's logging setup so engine logs (wallatag.auto) land on
    # stdout and become flow log lines via log_prints; harmless when Prefect
    # log handlers already exist.
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    # Unlike cmd_run's exit codes, exceptions here propagate: a flow run must
    # FAIL on bad config, missing credentials, or an unreachable API. All
    # resource construction (client, tagger/llm_client, store) lives INSIDE
    # this try so the finally closes whichever were created even when a
    # pre-loop step raises (mirrors cmd_run's explicit closes): a get_tags()
    # failure, a _build_tagger failure, or a Store() failure leaks nothing.
    client = None
    llm_client = None
    store = None
    try:
        client = WallabagClient(
            config.wallabag.url,
            config.wallabag.client_id,
            config.wallabag.client_secret,
            username=config.wallabag.username,
            password=config.wallabag.password,
        )
        existing_tags = [tag["label"] for tag in client.get_tags()]
        tagger, llm_client, fallback_tagger = _build_tagger(
            config, existing_tags
        )
        store = Store(
            config.store.path,
            reconsider_after_days=config.store.reconsider_after_days,
        )
        summary = auto.AutoSummary()
        failures = 0  # presented articles whose tag-article task run raised
        try:
            # iter_candidates is a plain function whose returned islice
            # eagerly evaluates client.iter_untagged(...) at the call site —
            # which is inside this guard, exactly like run_auto — so a feed
            # error sets feed_error instead of escaping the flow.
            for entry in auto.iter_candidates(client, store, config):
                try:
                    result = tag_article(
                        entry,
                        client=client,
                        tagger=tagger,
                        store=store,
                        cfg=config,
                        fallback_tagger=fallback_tagger,
                    )
                except TaggingFailedError as e:
                    result = e.result
                    failures += 1
                summary.presented += result.presented
                summary.tagged += result.tagged
                summary.tags_applied += result.tags_applied
                summary.skipped += result.skipped
                summary.llm_failed += result.llm_failed
                summary.llm_fallback += result.llm_fallback
        except (WallabagError, requests.RequestException) as exc:
            # The feed fetch died: report and stop gracefully, the same way
            # run_auto does (the flag distinguishes total from partial runs).
            summary.feed_error = True
            logger.error("error fetching entries: %s", exc)
    finally:
        if store is not None:
            store.close()
        if client is not None:
            client.close()
        if llm_client is not None:
            llm_client.close()
    # Total feed failure (nothing presented) must fail the flow run, mirroring
    # cmd_run's exit-2 condition; a partial run still returns its summary.
    if summary.feed_error and summary.presented == 0:
        raise RuntimeError(
            "wallatag run failed: could not fetch the article feed "
            "(feed error, nothing presented)"
        )
    # A run where every presented article failed to be tagged (LLM down,
    # wallabag add_tags down) must fail the flow run too, so the
    # scheduler/automations can react; its tag-article task runs are already
    # individually Failed, this lifts the signal to the flow-run level. A
    # partial run (at least one article tagged or normally skipped) still
    # returns its summary line.
    if failures and failures == summary.presented:
        causes = []
        if summary.llm_failed:
            causes.append(f"{summary.llm_failed} llm failures")
        tagging_failures = failures - summary.llm_failed
        if tagging_failures:
            causes.append(f"{tagging_failures} tagging failures")
        raise RuntimeError(
            "wallatag run failed: no article was tagged "
            f"({summary.presented} presented, "
            f"{', '.join(causes) or 'no reason recorded'})"
        )
    line = auto.summary_line(summary)
    print(line)
    return line
