"""Command-line interface for wallatag."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sqlite3
import sys
from pathlib import Path

from wallatag import __version__
from wallatag.auto import run_auto, summary_line as auto_summary_line
from wallatag.config import (
    VALID_TAG_POLICIES,
    Config,
    ConfigError,
    StoreConfig,
    apply_run_overrides,
    default_config_path,
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
    common.add_argument(
        "--config", metavar="PATH", help="path to configuration file"
    )
    common.add_argument(
        "--max",
        type=_non_negative_max,
        metavar="N",
        help="max articles per run (default: unlimited)",
    )
    common.add_argument(
        "--focus",
        metavar="NAME",
        action="append",
        help="activate one focus group (repeatable to select several)",
    )
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
    config_parser = subparsers.add_parser(
        "config",
        parents=[common],
        help="inspect or generate the configuration",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser(
        "show",
        parents=[common],
        help="show the effective configuration (default)",
    )
    config_init = config_subparsers.add_parser(
        "init",
        help="write a starter configuration file",
    )
    config_init.add_argument(
        "--config",
        metavar="PATH",
        help="destination path (default: the user config directory)",
    )
    config_init.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing configuration file",
    )
    # `wallatag status` is kept as an alias for backwards compatibility;
    # `wallatag config`/`wallatag config show` is the documented name.
    subparsers.add_parser(
        "status",
        parents=[common],
        help="show configuration summary (alias for `config show`)",
    )
    return parser


def apply_flag_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI flag overrides on top of a loaded config (flags win).

    The ``--max`` and ``--focus`` parts are shared with the Prefect flow via
    ``apply_run_overrides`` (both drivers must apply identical run-scope
    semantics); the rest is CLI-only flag handling.
    """
    config = apply_run_overrides(
        config,
        max_articles=args.max,
        focus=getattr(args, "focus", None),
    )
    if args.tag_policy is not None:
        config = dataclasses.replace(
            config,
            tagger=dataclasses.replace(
                config.tagger, tag_policy=args.tag_policy
            ),
        )
    if args.no_history:
        config = dataclasses.replace(config, store=StoreConfig(path=None))
    if getattr(args, "verbose", False):
        config = dataclasses.replace(config, verbose=True)
    return config


def _build_tagger(
    config: Config, existing_tags: list[str]
) -> tuple[KeywordTagger | LLMTagger, LLMClient | None, KeywordTagger | None]:
    """Build the tagger selected by config; returns
    (tagger, llm_client, fallback_tagger).

    The LLM tagger is active iff ``config.ai.provider`` is non-empty AND
    ``config.tagger.enable_llm`` is true; otherwise KeywordTagger is used.
    The returned ``llm_client`` (when not None) owns a session and MUST
    be closed by the caller alongside the wallabag client and store.

    When the LLM tagger is active AND ``config.ai.fallback_on_fail`` is true,
    ``fallback_tagger`` is a KeywordTagger built exactly like a normal
    keyword-mode run (same focus groups, max_applied_tags, tag_policy,
    existing tags, vocabulary fields, enable_vocabulary/enable_rules switches
    and ignore-list handling): the pipelines use it to tag an article the LLM
    failed on, per-article. Otherwise ``fallback_tagger`` is None.
    """
    if config.ai.provider and config.tagger.enable_llm:
        llm_client = LLMClient(
            config.ai.provider,
            config.ai.base_url,
            config.ai.model,
            api_key=config.ai.api_key,
        )
        # Explicit union: the non-LLM branch below assigns a KeywordTagger to
        # the same name, so mypy must not narrow it to LLMTagger here.
        tagger: KeywordTagger | LLMTagger = LLMTagger(
            llm_client,
            focus_groups=config.tagger.focus_groups,
            max_applied_tags=config.tagger.max_applied_tags,
            max_proposals=config.ai.max_proposals,
            tag_policy=config.tagger.tag_policy,
            existing_tags=existing_tags,
            confidence_threshold=config.ai.confidence_threshold,
            use_focus_groups=config.ai.use_focus_groups,
            verbose=config.verbose,
            ignore_tags=config.tagger.ignore_tags,
            ignore_tags_regex=config.tagger.ignore_tags_regex,
            skip_ignored_tags=config.vocabulary.skip_ignored_tags,
        )
        fallback_tagger = None
        if config.ai.fallback_on_fail:
            # Keyword-mode fallback for articles the LLM fails on: mirror the
            # non-LLM branch construction so it behaves exactly like a normal
            # keyword-mode run (enable_vocabulary/enable_rules/tag_policy and
            # the ignore-list handling all apply).
            fallback_tagger = KeywordTagger(
                config.tagger.focus_groups,
                max_applied_tags=config.tagger.max_applied_tags,
                tag_policy=config.tagger.tag_policy,
                existing_tags=existing_tags,
                vocabulary_fields=config.vocabulary.fields,
                enable_vocabulary=config.tagger.enable_vocabulary,
                enable_rules=config.tagger.enable_rules,
                ignore_tags=config.tagger.ignore_tags,
                ignore_tags_regex=config.tagger.ignore_tags_regex,
                skip_ignored_tags=config.vocabulary.skip_ignored_tags,
            )
        return tagger, llm_client, fallback_tagger
    tagger = KeywordTagger(
        config.tagger.focus_groups,
        max_applied_tags=config.tagger.max_applied_tags,
        tag_policy=config.tagger.tag_policy,
        existing_tags=existing_tags,
        vocabulary_fields=config.vocabulary.fields,
        enable_vocabulary=config.tagger.enable_vocabulary,
        enable_rules=config.tagger.enable_rules,
        ignore_tags=config.tagger.ignore_tags,
        ignore_tags_regex=config.tagger.ignore_tags_regex,
        skip_ignored_tags=config.vocabulary.skip_ignored_tags,
    )
    return tagger, None, None


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
        print(
            f"wallatag: error: could not fetch existing tags: {exc}",
            file=sys.stderr,
        )
        client.close()
        return 2
    tagger, llm_client, fallback_tagger = _build_tagger(config, existing_tags)
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
        summary = run_manual(
            client,
            tagger,
            store,
            config,
            dry_run=args.no_apply,
            fallback_tagger=fallback_tagger,
        )
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
        print(
            f"wallatag: error: could not fetch existing tags: {exc}",
            file=sys.stderr,
        )
        client.close()
        return 2
    tagger, llm_client, fallback_tagger = _build_tagger(config, existing_tags)
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
        summary = run_auto(
            client,
            tagger,
            store,
            config,
            dry_run=args.no_apply,
            fallback_tagger=fallback_tagger,
        )
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


def cmd_config_show(config: Config, args: argparse.Namespace) -> int:
    """`wallatag config` (bare) and `wallatag config show`; read-only summary."""
    if getattr(args, "config_command", None) not in (None, "show"):
        print(
            "wallatag: error: unknown config command",
            file=sys.stderr,
        )
        return 2
    url = config.wallabag.url or "(not configured)"
    store = config.store.path or "history-less"
    tagger = config.tagger
    print(f"wallatag {__version__}")
    print(f"wallabag: {url}")
    username = config.wallabag.username or "(not configured)"
    print(f"auth: username={username}")
    print(f"store: {store}")
    # max_proposals is an [ai] setting (LLM-only prompt bound); the keyword
    # tagger has no such knob, so it is shown only when set. getattr keeps
    # this line unchanged for taggers/configs without the attribute.
    max_proposals = getattr(config.ai, "max_proposals", None)
    max_proposals_suffix = (
        f" max_proposals={max_proposals}" if max_proposals is not None else ""
    )
    print(
        f"tagger: policy={tagger.tag_policy} "
        f"max_applied_tags={tagger.max_applied_tags}{max_proposals_suffix}"
    )
    ai = config.ai
    if ai.provider:
        # (llm fallback on) mirrors the parenthetical marker style of the
        # enable_llm note below; it appears only when [ai] fallback_on_fail
        # is set, so non-fallback runs stay byte-identical.
        fallback_marker = " (llm fallback on)" if ai.fallback_on_fail else ""
        if tagger.enable_llm:
            print(
                f"ai: provider={ai.provider} model={ai.model} "
                f"(confidence_threshold={ai.confidence_threshold})"
                f"{fallback_marker}"
            )
        else:
            print(
                f"ai: provider={ai.provider} model={ai.model} "
                f"(confidence_threshold={ai.confidence_threshold}) "
                f"(llm enabled=no; set [tagger] enable_llm = true)"
                f"{fallback_marker}"
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


def cmd_config_init(args: argparse.Namespace) -> int:
    """Write a starter configuration file, refusing to clobber by default.

    The annotated example shipped with the package is written to ``--config``
    when given, else to ``$XDG_CONFIG_HOME/wallatag/wallatag.toml`` (default
    ``~/.config/...``); the parent directory is created with mode 0700 because
    the file holds Wallabag credentials. An existing destination is only
    overwritten with ``--force``. Does not require a valid configuration.
    """
    import importlib.resources

    destination = Path(args.config) if args.config else default_config_path()
    if destination.exists() and not args.force:
        print(
            f"wallatag: error: {destination} already exists "
            f"(use --force to overwrite)",
            file=sys.stderr,
        )
        return 2
    try:
        template = (
            importlib.resources.files("wallatag")
            .joinpath("wallatag.toml.example")
            .read_text(encoding="utf-8")
        )
    except (OSError, FileNotFoundError) as exc:
        print(
            f"wallatag: error: cannot read example config: {exc}",
            file=sys.stderr,
        )
        return 2
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.write_text(template, encoding="utf-8")
    except OSError as exc:
        print(
            f"wallatag: error: cannot write {destination}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"wrote {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `config init` writes a fresh file and must work before any config
    # exists, so it bypasses config loading entirely.
    if (
        args.command == "config"
        and getattr(args, "config_command", None) == "init"
    ):
        return cmd_config_init(args)

    try:
        config = load_config(config_path=args.config)
        config = apply_flag_overrides(config, args)
    except ConfigError as exc:
        print(f"wallatag: error: {exc}", file=sys.stderr)
        return 2

    dispatch = {
        "manual": cmd_manual,
        "run": cmd_run,
        "config": cmd_config_show,
        "status": cmd_config_show,
    }
    return dispatch[args.command](config, args)
