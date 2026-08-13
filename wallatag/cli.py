"""Command-line interface for wallatag."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sqlite3
import sys

from wallatag import __version__
from wallatag.auto import run_auto, summary_line as auto_summary_line
from wallatag.config import (
    VALID_TAG_POLICIES,
    Config,
    ConfigError,
    StoreConfig,
    load_config,
)
from wallatag.manual import run_manual, summary_line
from wallatag.llm import LLMClient
from wallatag.store import Store
from wallatag.tagger import KeywordTagger, LLMTagger
from wallatag.wallabag import WallabagClient, WallabagError


def _non_negative_max(value: str) -> int:
    """argparse type for --max: a non-negative integer or argparse errors."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if n < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wallatag",
        description="AI-assisted auto-tagger for wallabag",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"wallatag {__version__}",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", metavar="PATH", help="path to configuration file")
    common.add_argument(
        "--max",
        type=_non_negative_max,
        metavar="N",
        help="max articles per run (default: unlimited)",
    )
    common.add_argument("--focus", metavar="NAME", help="activate one focus group")
    common.add_argument(
        "--tag-policy",
        choices=VALID_TAG_POLICIES,
        help="how to combine existing and suggested tags",
    )
    common.add_argument(
        "--no-history",
        action="store_true",
        help="disable the SQLite decision log for this run",
    )
    common.add_argument(
        "--no-apply",
        action="store_true",
        help="dry run: do not apply any changes",
    )
    common.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "manual",
        parents=[common],
        help="interactive review loop",
    )
    subparsers.add_parser(
        "run",
        parents=[common],
        help="headless batch tagging",
    )
    subparsers.add_parser(
        "status",
        parents=[common],
        help="show configuration summary",
    )
    return parser


def apply_flag_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI flag overrides on top of a loaded config (flags win)."""
    if args.max is not None:
        if args.max < 0:
            raise ConfigError("--max must be a non-negative integer")
        # Runtime run limit only: never touches [tagger] max_suggestions.
        config = dataclasses.replace(config, max_articles=args.max)
    if args.tag_policy is not None:
        config = dataclasses.replace(
            config, tagger=dataclasses.replace(config.tagger, tag_policy=args.tag_policy)
        )
    if args.no_history:
        config = dataclasses.replace(config, store=StoreConfig(path=None))
    if getattr(args, "verbose", False):
        config = dataclasses.replace(config, verbose=True)
    focus = getattr(args, "focus", None)
    if focus is not None:
        selected = config.tagger.focus_groups.get(focus)
        if selected is None:
            available = ", ".join(config.tagger.focus_groups) or "(none configured)"
            raise ConfigError(
                f"unknown focus group {focus!r}; available focus groups: {available}"
            )
        config = dataclasses.replace(
            config,
            tagger=dataclasses.replace(config.tagger, focus_groups={focus: selected}),
        )
    return config


def _build_tagger(
    config: Config, existing_tags: list[str]
) -> tuple[KeywordTagger | LLMTagger, LLMClient | None]:
    """Build the tagger selected by config; returns (tagger, llm_client).

    The LLM tagger is active iff ``config.ai.provider`` is non-empty AND
    ``config.tagger.enable_llm`` is true; otherwise KeywordTagger is used.
    The returned ``llm_client`` (when not None) owns a session and MUST
    be closed by the caller alongside the wallabag client and store.
    """
    if config.ai.provider and config.tagger.enable_llm:
        llm_client = LLMClient(
            config.ai.provider,
            config.ai.base_url,
            config.ai.model,
            api_key=config.ai.api_key,
        )
        tagger = LLMTagger(
            llm_client,
            focus_groups=config.tagger.focus_groups,
            max_suggestions=config.tagger.max_suggestions,
            tag_policy=config.tagger.tag_policy,
            existing_tags=existing_tags,
            confidence_threshold=config.ai.confidence_threshold,
            use_focus_groups=config.ai.use_focus_groups,
        )
        return tagger, llm_client
    tagger = KeywordTagger(
        config.tagger.focus_groups,
        max_suggestions=config.tagger.max_suggestions,
        tag_policy=config.tagger.tag_policy,
        existing_tags=existing_tags,
        vocabulary_fields=config.vocabulary.fields,
        enable_vocabulary=config.tagger.enable_vocabulary,
        enable_rules=config.tagger.enable_rules,
    )
    return tagger, None


def cmd_manual(config: Config, args: argparse.Namespace) -> int:
    """Interactive review loop."""
    if config.max_articles == 0:
        return 0
    if not config.wallabag.url:
        print(
            "wallatag: error: wallabag url is not configured; "
            "set [wallabag] url or WALLATAG_URL",
            file=sys.stderr,
        )
        return 2
    try:
        client = WallabagClient(
            config.wallabag.url,
            config.wallabag.client_id,
            config.wallabag.client_secret,
            username=config.wallabag.username,
            password=config.wallabag.password,
        )
    except ValueError as exc:
        print(f"wallatag: error: {exc}", file=sys.stderr)
        return 2
    try:
        existing_tags = [tag["label"] for tag in client.get_tags()]
    except WallabagError as exc:
        print(f"wallatag: error: could not fetch existing tags: {exc}", file=sys.stderr)
        client.close()
        return 2
    tagger, llm_client = _build_tagger(config, existing_tags)
    # Store creation is guarded so a sqlite failure (e.g. unwritable path)
    # reports cleanly and never leaks the client connection.
    store = None
    try:
        store = Store(config.store.path)
    except sqlite3.Error as exc:
        print(f"wallatag: error: could not open store: {exc}", file=sys.stderr)
        client.close()
        if llm_client is not None:
            llm_client.close()
        return 2
    try:
        summary = run_manual(client, tagger, store, config, dry_run=args.no_apply)
    except KeyboardInterrupt:
        # Clean exit on Ctrl-C: exit 130, close resources via finally.
        print("interrupted: exiting", file=sys.stderr)
        return 130
    except EOFError:
        # Ctrl-D at a prompt is a clean early exit: code 0.
        print("no input: exiting", file=sys.stderr)
        return 0
    finally:
        store.close()
        client.close()
        if llm_client is not None:
            llm_client.close()
    print(summary_line(summary, dry_run=args.no_apply))
    # Total feed failure (nothing presented) must look like a failure to a
    # scheduler/cron caller; a partial run still exits 0.
    if summary.feed_error and summary.presented == 0:
        return 2
    return 0


def cmd_run(config: Config, args: argparse.Namespace) -> int:
    """Headless one-shot batch tagging."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if config.max_articles == 0:
        return 0
    if not config.wallabag.url:
        print(
            "wallatag: error: wallabag url is not configured; "
            "set [wallabag] url or WALLATAG_URL",
            file=sys.stderr,
        )
        return 2
    try:
        client = WallabagClient(
            config.wallabag.url,
            config.wallabag.client_id,
            config.wallabag.client_secret,
            username=config.wallabag.username,
            password=config.wallabag.password,
        )
    except ValueError as exc:
        print(f"wallatag: error: {exc}", file=sys.stderr)
        return 2
    try:
        existing_tags = [tag["label"] for tag in client.get_tags()]
    except WallabagError as exc:
        print(f"wallatag: error: could not fetch existing tags: {exc}", file=sys.stderr)
        client.close()
        return 2
    tagger, llm_client = _build_tagger(config, existing_tags)
    # Store creation is guarded so a sqlite failure (e.g. unwritable path)
    # reports cleanly and never leaks the client connection.
    store = None
    try:
        store = Store(config.store.path)
    except sqlite3.Error as exc:
        print(f"wallatag: error: could not open store: {exc}", file=sys.stderr)
        client.close()
        if llm_client is not None:
            llm_client.close()
        return 2
    try:
        summary = run_auto(client, tagger, store, config, dry_run=args.no_apply)
    except KeyboardInterrupt:
        # Clean exit on Ctrl-C: exit 130, close resources via finally.
        print("interrupted: exiting", file=sys.stderr)
        return 130
    finally:
        store.close()
        client.close()
        if llm_client is not None:
            llm_client.close()
    logging.info(auto_summary_line(summary, dry_run=args.no_apply))
    # Total feed failure (nothing presented) must look like a failure to a
    # scheduler/cron caller; a partial run still exits 0.
    if summary.feed_error and summary.presented == 0:
        return 2
    return 0


def cmd_status(config: Config, args: argparse.Namespace) -> int:
    """Print a non-secret summary of the effective configuration."""
    url = config.wallabag.url or "(not configured)"
    store = config.store.path or "history-less"
    tagger = config.tagger
    print(f"wallatag {__version__}")
    print(f"wallabag: {url}")
    username = config.wallabag.username or "(not configured)"
    print(f"auth: username={username}")
    print(f"store: {store}")
    print(f"tagger: policy={tagger.tag_policy} max_suggestions={tagger.max_suggestions}")
    ai = config.ai
    if ai.provider:
        if tagger.enable_llm:
            print(
                f"ai: provider={ai.provider} model={ai.model} "
                f"(confidence_threshold={ai.confidence_threshold})"
            )
        else:
            print(
                f"ai: provider={ai.provider} model={ai.model} "
                f"(confidence_threshold={ai.confidence_threshold}) "
                f"(llm enabled=no; set [tagger] enable_llm = true)"
            )
    else:
        print("ai: not configured")
    if config.max_articles is not None:
        print(f"run limit: {config.max_articles} articles (from --max)")
    groups = ", ".join(
        f"{name} ({len(group.keywords)} keywords, {len(group.tags)} tags)"
        for name, group in tagger.focus_groups.items()
    )
    print(f"focus groups: {groups or '(none)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(config_path=args.config)
        config = apply_flag_overrides(config, args)
    except ConfigError as exc:
        print(f"wallatag: error: {exc}", file=sys.stderr)
        return 2

    dispatch = {
        "manual": cmd_manual,
        "run": cmd_run,
        "status": cmd_status,
    }
    return dispatch[args.command](config, args)
