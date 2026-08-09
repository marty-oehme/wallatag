"""Headless one-shot batch tagging for the ``run`` subcommand.

Same pipeline as ``manual`` but with no prompts: fetch untagged entries, let
the tagger suggest tags, apply them per the tag policy, and log decisions to
the optional SQLite store. Deterministic and idempotent: the store's seen
dedupe plus wallabag's own state (tagged articles stop matching the untagged
filter) prevent double-tagging. Deliberately a one-shot run, NOT a loop:
Prefect is the sole scheduler, so there is no --interval flag.

All output goes through stdlib logging (no print()) so Prefect can capture it.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import requests

from wallatag.config import Config
from wallatag.llm import LLMError
from wallatag.wallabag import WallabagError

logger = logging.getLogger(__name__)


@dataclass
class AutoSummary:
    presented: int = 0
    tagged: int = 0
    tags_applied: int = 0
    skipped: int = 0
    feed_error: bool = False
    dry_run: bool = False


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
    return line


def run_auto(client, tagger, store, cfg: Config, *, dry_run: bool = False) -> AutoSummary:
    """Run the headless batch loop; returns an AutoSummary.

    ``client`` only needs iter_untagged/get_tags/add_tags/close. ``store`` may
    be a history-less Store(None). In dry-run mode no writes happen at all.
    """
    summary = AutoSummary(dry_run=dry_run)
    # The genexpr lives INSIDE the try: a genexpr evaluates its outermost
    # iterable eagerly at creation, so client.iter_untagged(...) must be
    # covered by the guard (a non-generator iter_untagged can raise here).
    try:
        unseen = (
            entry
            for entry in client.iter_untagged(
                per_page=30, ignored_tags=cfg.tagger.ignore_tags
            )
            if not store.is_seen(entry["id"])
        )
        for entry in itertools.islice(unseen, cfg.max_articles):
            entry_id = entry["id"]
            if not dry_run:
                store.mark_seen(entry_id)  # pick-up: dedupe concurrent runs
            summary.presented += 1

            try:
                suggestions = tagger.suggest(entry)
            except LLMError as exc:
                # A model failure is an article-level skip, not a feed failure:
                # keep going with the rest of the run.
                logger.error("LLM tagging failed %s: %s", entry_id, exc)
                summary.skipped += 1
                continue
            tags = sorted({suggestion.tag for suggestion in suggestions})
            logger.debug(
                "entry %s: %s", entry_id, ", ".join(tags) or "(no suggestions)"
            )

            if dry_run:
                if tags:
                    logger.info("would tag %s: %s", entry_id, ", ".join(tags))
                    summary.tagged += 1
                    summary.tags_applied += len(tags)
                else:
                    logger.info("would skip %s", entry_id)
                    summary.skipped += 1
                continue

            if not tags:
                logger.info("no suggestions: %s", entry_id)
                summary.skipped += 1
                continue
            try:
                client.add_tags(entry_id, tags)
            except (WallabagError, requests.RequestException) as exc:
                logger.error("tagging failed %s: %s", entry_id, exc)
                continue  # no decisions for a failed article; keep going
            logger.info("tagged %s: %s", entry_id, ", ".join(tags))
            summary.tagged += 1
            summary.tags_applied += len(tags)
            for suggestion in suggestions:
                store.record_decision(
                    entry_id, suggestion.tag, "accept", suggestion.source
                )
    except (WallabagError, requests.RequestException) as exc:
        # The feed fetch died: report and stop gracefully, no traceback. The
        # flag lets callers distinguish a total failure from a partial run.
        summary.feed_error = True
        logger.error("error fetching entries: %s", exc)

    return summary
