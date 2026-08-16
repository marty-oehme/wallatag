"""Interactive review loop for the ``manual`` subcommand.

Plain terminal prompts (no TUI). Presents one untagged article at a time and
lets the user build a per-article WORKING TAG LIST, pre-seeded with the
tagger's suggestions. The user edits it freely: add custom tags, drop entries,
then commits with "next" or leaves the article untouched with "skip". Confirmed
tags hit the wallabag API immediately; accept/reject decisions are logged to
the optional SQLite store. In dry-run mode (--no-apply) nothing is written: no
mark_seen, no record_decision, no add_tags."""

from __future__ import annotations

import html
import itertools
import re
import sys
from dataclasses import dataclass

import requests

from wallatag.config import Config
from wallatag.llm import LLMError
from wallatag.tagger import KeywordTagger, TagSuggestion
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
    dropped: int = 0
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
            f"{summary.accepted} tags, drop {summary.dropped}, "
            f"add {summary.added} custom"
        )
    return (
        f"tagged {summary.tagged} articles, accepted {summary.accepted} tags, "
        f"dropped {summary.dropped}, added {summary.added} custom"
    )


def run_manual(
    client,
    tagger,
    store,
    cfg: Config,
    *,
    dry_run: bool = False,
    fallback_tagger: KeywordTagger | None = None,
) -> ManualSummary:
    """Run the interactive review loop; returns a ManualSummary.

    ``client`` only needs iter_untagged/add_tags/close. ``store`` may be a
    history-less Store(None). In dry-run mode no writes happen at all.

    ``fallback_tagger`` (optional, keyword-only) is the keyword tagger used
    when the LLM tagger fails for an article ([ai] fallback_on_fail): its
    suggestions enter the normal review flow for THAT article, the LLM is
    still tried on subsequent articles, and no counter is surfaced (manual
    mode reports nothing extra, consistent with today).
    """
    summary = ManualSummary()
    # The genexpr lives INSIDE the try: a genexpr evaluates its outermost
    # iterable eagerly at creation, so client.iter_untagged(...) must be
    # covered by the guard (a non-generator iter_untagged can raise here).
    try:
        unseen = (
            entry
            for entry in client.iter_untagged(
                per_page=30,
                ignored_tags=cfg.tagger.ignore_tags,
                ignored_regex=cfg.tagger.ignore_tags_regex,
            )
            if not store.is_seen(entry["id"])
        )
        for entry in itertools.islice(unseen, cfg.max_articles):
            entry_id = entry["id"]
            if cfg.verbose:
                print(f"[debug] presenting entry {entry_id}", file=sys.stderr)
            if not dry_run:
                store.mark_seen(entry_id)  # pick-up: dedupe concurrent runs
            summary.presented += 1

            try:
                suggestions = tagger.suggest(entry)
            except LLMError as exc:
                # A model failure skips the article unless a keyword fallback
                # is configured ([ai] fallback_on_fail); the session keeps
                # going either way.
                deferred = (
                    True  # dropped unless the keyword fallback rescues it
                )
                if fallback_tagger is not None:
                    # Per-article fallback: the keyword tagger takes over for
                    # THIS article only — the LLM is still tried on subsequent
                    # articles. Its suggestions carry the usual "vocabulary"/
                    # "rules" sources and enter the normal review flow below.
                    try:
                        suggestions = fallback_tagger.suggest(entry)
                    except Exception:
                        suggestions = []
                    deferred = not suggestions
                if deferred:
                    print(
                        f"LLM tagging failed {entry_id}: {exc}",
                        file=sys.stderr,
                    )
                    if not dry_run:
                        store.unmark_seen(
                            entry_id
                        )  # defer: keep the article in the queue
                    continue
            try:
                action, working = _edit_working_list(entry, suggestions)
            except _Quit:
                break
            if action == "next":
                _apply(
                    client,
                    store,
                    entry_id,
                    suggestions,
                    working,
                    dry_run,
                    summary,
                )
            # "skip" applies and records nothing.
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


def _show_working_list(working: list) -> None:
    if not working:
        print("  (no tags)")
        return
    for i, item in enumerate(working, 1):
        if item.source == "manual":
            print(f"  [{i}] {item.tag}  (source=manual)")
        else:
            print(
                f"  [{i}] {item.tag}  "
                f"(source={item.source}, confidence={item.confidence})"
            )


def _redisplay(entry: dict, working: list) -> None:
    """Article header plus the current numbered working tag list."""
    _present_article(entry)
    _show_working_list(working)


def _edit_working_list(entry: dict, original_suggestions: list):
    """Loop the per-article edit menu.

    Returns ``("next", working)`` to commit the list or ``("skip", working)``
    to leave the article untouched; raises ``_Quit`` to end the session.
    """
    working = list(original_suggestions)
    while True:
        _redisplay(entry, working)
        choice = (
            input("add [a] | drop [d] | next [enter] | skip [s] | quit [q] > ")
            .strip()
            .lower()
        )
        if choice == "":
            return "next", working
        if choice == "q":
            raise _Quit()
        if choice == "s":
            return "skip", working
        if choice == "a":
            try:
                new_tags = _ask_tags([item.tag for item in working])
            except _Abort:
                continue  # 'q' at the sub-prompt: back to the action menu
            working.extend(
                TagSuggestion(tag=tag, source="manual", confidence=1.0)
                for tag in new_tags
            )
            continue  # re-display the article + updated list
        if choice == "d":
            try:
                indices = _pick_indices(len(working), verb="drop")
            except _Abort:
                continue
            remove = set(indices)
            working = [
                item for i, item in enumerate(working) if i not in remove
            ]
            continue  # re-display the article + updated list
        print(
            "invalid choice; enter a, d, enter for next, s to skip, q to quit"
        )


def _pick_indices(count: int, *, verb: str = "drop") -> list[int]:
    """Ask for comma-separated 1-based numbers; empty = all.

    Typing 'q' aborts the sub-prompt (raises _Abort), it is never treated as
    index content. Invalid/out-of-range numbers are ignored; if nothing valid
    was given, a message is printed and the prompt repeats.
    """
    if count == 0:
        return []
    while True:
        raw = input(
            f"  {verb} numbers (comma-separated, empty = all): "
        ).strip()
        if raw.lower() == "q":
            raise _Abort()
        if raw == "":
            return list(range(count))
        try:
            numbers = [int(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            print("  invalid selection")
            continue
        valid = sorted({n - 1 for n in numbers if 1 <= n <= count})
        if not valid:
            print("  invalid selection")
            continue
        return valid


def _ask_tags(existing: list[str]) -> list[str]:
    """Ask for comma-separated extra tags (deduped, order preserved).

    Tags already present in the working list are skipped; typing 'q' aborts
    the sub-prompt (raises _Abort). Re-prompts until a non-empty list is given.
    """
    existing_folded = {tag.casefold() for tag in existing}
    while True:
        raw = input("  extra tags (comma-separated): ").strip()
        if raw.lower() == "q":
            raise _Abort()
        tags = []
        for tag in (part.strip() for part in raw.split(",")):
            if (
                tag
                and tag.casefold() not in existing_folded
                and tag not in tags
            ):
                tags.append(tag)
        if tags:
            return tags
        print("  no tags entered")


def _tally(summary: ManualSummary, working: list, dropped: list) -> None:
    """Accumulate summary counts for a settled article."""
    summary.accepted += sum(1 for item in working if item.source != "manual")
    summary.added += sum(1 for item in working if item.source == "manual")
    summary.dropped += len(dropped)


def _apply(
    client,
    store,
    entry_id: int,
    original_suggestions: list,
    working: list,
    dry_run: bool,
    summary: ManualSummary,
) -> None:
    """Commit a settled article: apply the working list and record decisions."""
    tags = sorted({item.tag for item in working})
    working_folded = {item.tag.casefold() for item in working}
    dropped = [
        suggestion
        for suggestion in original_suggestions
        if suggestion.tag.casefold() not in working_folded
    ]

    if dry_run:
        if tags:
            print(f"  (dry run) would apply: {', '.join(tags)}")
            summary.tagged += 1
        if dropped:
            print(
                f"  (dry run) would reject: "
                f"{', '.join(sorted({suggestion.tag for suggestion in dropped}))}"
            )
        _tally(summary, working, dropped)
        return

    if tags:
        try:
            client.add_tags(entry_id, tags)
        except (WallabagError, requests.RequestException) as exc:
            print(f"  error tagging entry {entry_id}: {exc}")
            return
        summary.tagged += 1
    _tally(summary, working, dropped)

    for item in working:
        store.record_decision(entry_id, item.tag, "accept", item.source)
    for suggestion in dropped:
        store.record_decision(
            entry_id, suggestion.tag, "reject", suggestion.source
        )
