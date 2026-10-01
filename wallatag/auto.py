"""Headless one-shot batch tagging for the ``run`` subcommand.

Same pipeline as ``manual`` but with no prompts: fetch untagged entries, let
the tagger suggest tags, apply them per the tag policy, and log decisions to
the optional SQLite store. Deterministic and idempotent: the store's claim
(the seen dedupe, a 7-day cooldown so skipped/interrupted articles come back;
bug 8a7d089) plus wallabag's own state (tagged articles stop matching the
untagged filter) prevent double-tagging. Deliberately a one-shot run, NOT a
loop: Prefect is the sole scheduler, so there is no --interval flag.

All output goes through stdlib logging (no print()) so Prefect can capture it.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

import requests

from wallatag.config import Config
from wallatag.llm import LLMError
from wallatag.tagger import KeywordTagger
from wallatag.wallabag import WallabagError

logger = logging.getLogger(__name__)

# Retries around the final add_tags POST: the engine owns failure handling
# (bug d968e0f), so both the CLI and the Prefect flow get the same bounded,
# classified retry here — the Prefect task itself stays retries=0.
ADD_TAGS_RETRIES = 2  # retries AFTER the first attempt -> up to 3 total
ADD_TAGS_RETRY_DELAY = 2.0  # base seconds; exponential backoff: 2.0, 4.0
# HTTP statuses worth re-attempting: timeouts/queueing/transient server-side
# errors. Deterministic 4xx failures (400, 403, 404, ...) are NOT retried.
RETRYABLE_STATUSES: frozenset[int] = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)


@dataclass
class AutoSummary:
    presented: int = 0
    tagged: int = 0
    tags_applied: int = 0
    skipped: int = 0
    llm_failed: int = 0
    llm_fallback: int = 0
    feed_error: bool = False
    dry_run: bool = False


@dataclass
class EntryResult:
    """Per-article outcome of :func:`process_entry` (count deltas).

    The count fields are deltas for ONE article: ``presented`` is 1 when the
    article was presented, and exactly one of ``tagged``/``skipped`` is 1 —
    except an add_tags failure that exhausts its engine retries and fails
    PERMANENTLY (a deterministic 4xx): then the article is presented but
    neither tagged nor skipped and stays CLAIMED. A final TRANSIENT failure
    (transport-level or retryable HTTP) instead unmarks the article, so it is
    re-presented (and counted again) on the next run. ``tags`` holds the
    deduped, sorted tag strings that were applied (or that would be applied
    in dry-run); ``outcome`` names the branch that ran (``"tagged"``,
    ``"no suggestions"``, ``"llm_failed"``, ``"tagging failed"``, or the
    dry-run ``"would tag"``/``"would skip"``).
    """

    entry_id: int
    tags: tuple[str, ...]
    outcome: str
    presented: int = 0
    tagged: int = 0
    skipped: int = 0
    tags_applied: int = 0
    llm_failed: int = 0
    llm_fallback: int = 0


def summary_line(summary: AutoSummary, *, dry_run: bool = False) -> str:
    """One-line end-of-run summary (greppable for Prefect log capture)."""
    if dry_run:
        line = (
            f"dry run: would tag {summary.tagged} articles "
            f"({summary.tags_applied} tags), skipped {summary.skipped}"
        )
    else:
        line = (
            f"run: tagged {summary.tagged} articles "
            f"({summary.tags_applied} tags applied), skipped {summary.skipped}"
        )
    if summary.feed_error:
        line += ", feed error"
    if summary.llm_failed > 0:
        line += f", {summary.llm_failed} llm failures"
    if summary.llm_fallback > 0:
        line += f", {summary.llm_fallback} via fallback"
    return line


def iter_candidates(client, store, cfg: Config):
    """Yield the candidate entries for a headless run, in run order.

    The same iteration ``run_auto`` uses: the untagged feed (per_page=30, the
    ``[tagger]`` ignore lists from config) filtered by the store's TTL-aware
    seen cooldown, capped at ``cfg.max_articles``. Shared by ``run_auto`` and
    the Prefect flow so both drivers iterate the feed identically.
    """
    return itertools.islice(
        (
            entry
            for entry in client.iter_untagged(
                per_page=30,
                ignored_tags=cfg.tagger.ignore_tags,
                ignored_regex=cfg.tagger.ignore_tags_regex,
            )
            if not store.is_seen(entry["id"])
        ),
        cfg.max_articles,
    )


def _is_retryable_add_tags_error(exc: Exception) -> bool:
    """True for failures worth re-attempting: transport-level (WallabagError
    without a status code, e.g. connection errors/timeouts; or a raw
    requests.RequestException from a duck-typed client) and the usual
    retryable HTTP statuses. Deterministic 4xx failures are not retried.
    """
    if isinstance(exc, requests.RequestException):
        return True
    if isinstance(exc, WallabagError):
        return exc.status is None or exc.status in RETRYABLE_STATUSES
    return False


def process_entry(
    client,
    tagger,
    store,
    cfg: Config,
    entry: dict,
    *,
    dry_run: bool = False,
    fallback_tagger: KeywordTagger | None = None,
    add_tags_retries: int = ADD_TAGS_RETRIES,
    add_tags_retry_delay: float = ADD_TAGS_RETRY_DELAY,
) -> EntryResult:
    """Process one candidate article; returns the per-article count deltas.

    This is the per-article body of the headless batch loop, extracted from
    ``run_auto`` so the CLI driver and the Prefect flow share the exact same
    engine behavior: claim the article (unless dry-run), ask the tagger, apply
    tags per the tag policy, and record decisions. A lost claim to a
    concurrent run returns the zero-valued result without presenting the
    article (bug 8a7d089). Side-effect sequence and count increments are
    EXACTLY those of the original loop body otherwise.

    ``client`` only needs add_tags, ``store`` claim/unmark_seen/
    record_decision. An ``LLMError`` from the tagger is an article-level skip,
    NOT a feed error: the caller keeps iterating. ``fallback_tagger``
    semantics are those documented on ``run_auto``.

    The final add_tags POST is retried by the ENGINE (bug d968e0f): up to
    ``add_tags_retries`` retries after the first attempt (default 2 -> 3
    attempts total) with exponential backoff from ``add_tags_retry_delay``.
    Transport-level failures (connection error/timeout, status None) and the
    retryable HTTP statuses are re-attempted; deterministic 4xx failures are
    not. When even the final attempt fails TRANSIENTLY the article is
    unmarked (deferred to the next run, mirroring the LLM-failure path);
    when it fails DETERMINISTICALLY (4xx) the article stays seen and is
    dropped deliberately.
    """
    if add_tags_retries < 0:
        # range(add_tags_retries + 1) would be empty: the loop never runs,
        # last_exc stays None, and the article would be reported "tagged"
        # without add_tags ever being called (silent false success).
        raise ValueError("add_tags_retries must be >= 0")
    entry_id = entry["id"]
    result = EntryResult(entry_id=entry_id, tags=(), outcome="skipped")
    if not dry_run and not store.claim(entry_id):
        # Lost the atomic pick-up race to a concurrent run (the article was
        # claimed between candidate iteration and now): leave it to them and
        # report nothing presented rather than double-tagging.
        return result
    result.presented = 1

    used_fallback = False
    try:
        suggestions = tagger.suggest(entry)
    except LLMError as exc:
        # A model failure is an article-level skip, not a feed failure:
        # keep going with the rest of the run. With [ai]
        # fallback_on_fail configured, the keyword tagger takes over
        # for THIS article only (the LLM is still tried on subsequent
        # ones). The fallback is built exactly like a normal keyword-
        # mode run, so enable_vocabulary/enable_rules/tag_policy all
        # apply and its suggestions carry the usual "vocabulary"/
        # "rules" sources into the shared apply path below.
        if fallback_tagger is None:
            # No fallback: the failure is final for this article.
            logger.error("LLM tagging failed %s: %s", entry_id, exc)
            result.outcome = "llm_failed"
            result.skipped = 1
            result.llm_failed = 1
            if not dry_run:
                store.unmark_seen(
                    entry_id
                )  # defer: keep the article in the queue
            return result
        try:
            suggestions = fallback_tagger.suggest(entry)
        except Exception:
            suggestions = []
        if not suggestions:
            # The fallback yielded nothing (or failed): keep today's
            # LLM-failure path (the article is NOT counted via
            # fallback).
            logger.error("LLM tagging failed %s: %s", entry_id, exc)
            result.outcome = "llm_failed"
            result.skipped = 1
            result.llm_failed = 1
            if not dry_run:
                store.unmark_seen(
                    entry_id
                )  # defer: keep the article in the queue
            return result
        # The keyword fallback rescued this article: report the failure
        # as a WARNING (the article is still tagged below), and count
        # it as "via fallback" only once the apply path actually
        # succeeds below — empty deduped tags or add_tags failures do
        # NOT count.
        logger.warning(
            "LLM tagging failed %s: %s (keyword fallback applied)",
            entry_id,
            exc,
        )
        used_fallback = True
    tags = sorted({suggestion.tag for suggestion in suggestions})
    result.tags = tuple(tags)
    logger.debug(
        "entry %s: %s", entry_id, ", ".join(tags) or "(no suggestions)"
    )

    if dry_run:
        if tags:
            logger.info("would tag %s: %s", entry_id, ", ".join(tags))
            result.outcome = "would tag"
            result.tagged = 1
            result.tags_applied = len(tags)
            if used_fallback:
                result.llm_fallback = 1
        else:
            logger.info("would skip %s", entry_id)
            result.outcome = "would skip"
            result.skipped = 1
        return result

    if not tags:
        logger.info("no suggestions: %s", entry_id)
        result.outcome = "no suggestions"
        result.skipped = 1
        return result
    last_exc: Exception | None = None
    for attempt in range(add_tags_retries + 1):
        try:
            client.add_tags(entry_id, tags)
        except (WallabagError, requests.RequestException) as exc:
            last_exc = exc
            if (
                not _is_retryable_add_tags_error(exc)
                or attempt == add_tags_retries
            ):
                break
            logger.warning(
                "tagging failed %s: %s (retry %d/%d)",
                entry_id,
                exc,
                attempt + 1,
                add_tags_retries,
            )
            time.sleep(add_tags_retry_delay * (2**attempt))
        else:
            last_exc = None
            break
    if last_exc is not None:
        logger.error("tagging failed %s: %s", entry_id, last_exc)
        result.outcome = "tagging failed"
        # Transient failures (transport-level or retryable HTTP) defer the
        # article to the next run: unmark_seen mirrors the LLM-failure path
        # and lets the hourly/backfill schedule re-attempt it. Deterministic
        # failures (4xx) stay claimed: re-processing immediately would fail
        # identically and only burn tagger calls, so they are left to the
        # 7-day cooldown (bug 8a7d089) instead of being dropped forever.
        if not dry_run and _is_retryable_add_tags_error(last_exc):
            store.unmark_seen(entry_id)  # defer: retry on the next run
        return result
    logger.info("tagged %s: %s", entry_id, ", ".join(tags))
    result.outcome = "tagged"
    result.tagged = 1
    result.tags_applied = len(tags)
    if used_fallback:
        result.llm_fallback = 1
    for suggestion in suggestions:
        store.record_decision(
            entry_id, suggestion.tag, "accept", suggestion.source
        )
    return result


def run_auto(
    client,
    tagger,
    store,
    cfg: Config,
    *,
    dry_run: bool = False,
    fallback_tagger: KeywordTagger | None = None,
    add_tags_retries: int = ADD_TAGS_RETRIES,
    add_tags_retry_delay: float = ADD_TAGS_RETRY_DELAY,
) -> AutoSummary:
    """Run the headless batch loop; returns an AutoSummary.

    ``client`` only needs iter_untagged/get_tags/add_tags/close. ``store`` may
    be a history-less Store(None). In dry-run mode no writes happen at all.

    ``fallback_tagger`` (optional, keyword-only) is the keyword tagger used
    when the LLM tagger fails for an article ([ai] fallback_on_fail): it
    behaves exactly like a normal keyword-mode run, per-article, and the LLM
    is still tried on subsequent articles.

    ``add_tags_retries``/``add_tags_retry_delay`` (optional, keyword-only) are
    the engine's add_tags retry policy, passed through to ``process_entry``
    (see its docstring); CLI and flow may override, but don't by default.
    """
    summary = AutoSummary(dry_run=dry_run)
    # iter_candidates is a plain function whose returned islice eagerly
    # evaluates client.iter_untagged(...) at the call site — which is inside
    # this try — so a feed error (even from a non-generator iter_untagged)
    # is caught here and sets feed_error instead of escaping.
    try:
        for entry in iter_candidates(client, store, cfg):
            result = process_entry(
                client,
                tagger,
                store,
                cfg,
                entry,
                dry_run=dry_run,
                fallback_tagger=fallback_tagger,
                add_tags_retries=add_tags_retries,
                add_tags_retry_delay=add_tags_retry_delay,
            )
            summary.presented += result.presented
            summary.tagged += result.tagged
            summary.tags_applied += result.tags_applied
            summary.skipped += result.skipped
            summary.llm_failed += result.llm_failed
            summary.llm_fallback += result.llm_fallback
    except (WallabagError, requests.RequestException) as exc:
        # The feed fetch died: report and stop gracefully, no traceback. The
        # flag lets callers distinguish a total failure from a partial run.
        summary.feed_error = True
        logger.error("error fetching entries: %s", exc)

    return summary
