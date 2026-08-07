"""Interactive review loop for the ``manual`` subcommand.

Plain terminal prompts (no TUI). Presents one untagged article at a time,
shows tagger suggestions, and lets the user keep/reject/add/skip/quit. Confirmed
tags hit the wallabag API immediately; accept/reject decisions are logged to
the optional SQLite store. In dry-run mode (--no-apply) nothing is written:
no mark_seen, no record_decision, no add_tags.
"""

from __future__ import annotations

import html
import itertools
import re
import sys
from dataclasses import dataclass

import requests

from wallatag.config import Config
from wallatag.wallabag import WallabagError

_SNIPPET_LENGTH = 200
_TAG_RE = re.compile(r"<[^>]+>")


class _Quit(Exception):
    """Internal signal: the user chose to quit the session."""


class _Abort(Exception):
    """Internal signal: 'q' typed at a sub-prompt: return to the action menu."""


@dataclass
class ManualSummary:
    presented: int = 0
    tagged: int = 0
    accepted: int = 0
    rejected: int = 0
    added: int = 0
    feed_error: bool = False


def strip_html(text: str) -> str:
    """Strip HTML tags and unescape entities, then collapse whitespace."""
    if not isinstance(text, str):
        text = str(text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def summary_line(summary: ManualSummary, *, dry_run: bool = False) -> str:
    """One-line end-of-session summary."""
    if dry_run:
        return (
            f"dry run: would tag {summary.tagged} articles, accept "
            f"{summary.accepted} tags, reject {summary.rejected}, "
            f"add {summary.added} custom"
        )
    return (
        f"tagged {summary.tagged} articles, accepted {summary.accepted} tags, "
        f"rejected {summary.rejected}, added {summary.added} custom"
    )


def run_manual(client, tagger, store, cfg: Config, *, dry_run: bool = False) -> ManualSummary:
    """Run the interactive review loop; returns a ManualSummary.

    ``client`` only needs iter_untagged/add_tags/close. ``store`` may be a
    history-less Store(None). In dry-run mode no writes happen at all.
    """
    summary = ManualSummary()
    # The genexpr lives INSIDE the try: a genexpr evaluates its outermost
    # iterable eagerly at creation, so client.iter_untagged(...) must be
    # covered by the guard (a non-generator iter_untagged can raise here).
    try:
        unseen = (
            entry
            for entry in client.iter_untagged(per_page=30)
            if not store.is_seen(entry["id"])
        )
        for entry in itertools.islice(unseen, cfg.max_articles):
            entry_id = entry["id"]
            if cfg.verbose:
                print(f"[debug] presenting entry {entry_id}", file=sys.stderr)
            if not dry_run:
                store.mark_seen(entry_id)  # pick-up: dedupe concurrent runs
            summary.presented += 1

            _present_article(entry)
            suggestions = tagger.suggest(entry)
            _show_suggestions(suggestions)

            try:
                kept, rejected, added = _interact(suggestions)
            except _Quit:
                break
            _apply(client, store, entry_id, kept, rejected, added, dry_run, summary)
    except (WallabagError, requests.RequestException) as exc:
        # The feed fetch died (e.g. network failure while paginating): report
        # and finish with what we have instead of dumping a traceback. The
        # flag lets callers distinguish a total failure from a partial run.
        summary.feed_error = True
        print(f"error fetching entries: {exc}", file=sys.stderr)

    return summary


def _present_article(entry: dict) -> None:
    print(f"--- entry {entry.get('id')} ---")
    print(entry.get("title") or "(untitled)")
    print(f"url: {entry.get('url') or '(none)'}")
    print(f"domain: {entry.get('domain_name') or '(unknown)'}")
    print(f"reading time: ~{entry.get('reading_time') or 0} min")
    snippet = strip_html(entry.get("content") or "")[:_SNIPPET_LENGTH]
    if snippet:
        print(f"snippet: {snippet}")


def _show_suggestions(suggestions: list) -> None:
    if not suggestions:
        print("  (no suggestions)")
        return
    for i, suggestion in enumerate(suggestions, 1):
        print(
            f"  [{i}] {suggestion.tag}  "
            f"(source={suggestion.source}, confidence={suggestion.confidence})"
        )


def _interact(suggestions: list):
    """Loop the action menu; returns (kept, rejected, added). Raises _Quit."""
    while True:
        choice = input(
            "keep [k] | add [a] | reject [r] | skip [enter] | quit [q] > "
        ).strip().lower()
        if choice == "":
            return [], [], []
        if choice == "q":
            raise _Quit()
        if choice == "k":
            try:
                indices = _pick_indices(len(suggestions))
            except _Abort:
                continue  # 'q' at the sub-prompt: back to the action menu
            return [suggestions[i] for i in indices], [], []
        if choice == "a":
            try:
                tags = _ask_tags()
            except _Abort:
                continue
            return [], [], tags
        if choice == "r":
            try:
                indices = _pick_indices(len(suggestions), verb="reject")
            except _Abort:
                continue
            return [], [suggestions[i] for i in indices], []
        print("invalid choice; enter k, a, r, q or press enter to skip")


def _pick_indices(count: int, *, verb: str = "accept") -> list[int]:
    """Ask for comma-separated 1-based numbers; empty = all.

    Typing 'q' aborts the sub-prompt (raises _Abort) — it is never treated as
    index content. Invalid/out-of-range numbers are ignored; if nothing valid
    was given, a message is printed and the prompt repeats.
    """
    if count == 0:
        return []
    while True:
        raw = input(f"  {verb} numbers (comma-separated, empty = all): ").strip()
        if raw.lower() == "q":
            raise _Abort()
        if raw == "":
            return list(range(count))
        try:
            numbers = [int(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            print("  invalid selection")
            continue
        if not numbers or any(n < 1 or n > count for n in numbers):
            print("  invalid selection")
            continue
        return sorted({n - 1 for n in numbers})


def _ask_tags() -> list[str]:
    """Ask for comma-separated extra tags (deduped, order preserved).

    Typing 'q' aborts the sub-prompt (raises _Abort) — it is never treated as a
    tag. Re-prompts until a non-empty, deduplicated list is entered.
    """
    while True:
        raw = input("  extra tags (comma-separated): ").strip()
        if raw.lower() == "q":
            raise _Abort()
        tags = []
        for tag in (part.strip() for part in raw.split(",")):
            if tag and tag not in tags:
                tags.append(tag)
        if tags:
            return tags
        print("  no tags entered")


def _apply(
    client,
    store,
    entry_id: int,
    kept: list,
    rejected: list,
    added: list[str],
    dry_run: bool,
    summary: ManualSummary,
) -> None:
    """Apply accepted tags and record decisions for a settled article."""
    union = sorted({suggestion.tag for suggestion in kept} | set(added))
    if dry_run:
        if union:
            print(f"  (dry run) would apply: {', '.join(union)}")
            summary.tagged += 1
        summary.accepted += len(kept)
        summary.rejected += len(rejected)
        summary.added += len(added)
        return

    if union:
        try:
            client.add_tags(entry_id, union)
        except (WallabagError, requests.RequestException) as exc:
            print(f"  error tagging entry {entry_id}: {exc}")
            return
        summary.tagged += 1
    summary.accepted += len(kept)
    summary.rejected += len(rejected)
    summary.added += len(added)

    for suggestion in kept:
        store.record_decision(entry_id, suggestion.tag, "accept", suggestion.source)
    for suggestion in rejected:
        store.record_decision(entry_id, suggestion.tag, "reject", suggestion.source)
    for tag in added:
        store.record_decision(entry_id, tag, "accept", "manual")
