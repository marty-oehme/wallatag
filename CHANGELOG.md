# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `[store] reconsider_after_days` (env `WALLATAG_STORE_RECONSIDER_AFTER_DAYS`)
  makes the pick-up cooldown configurable instead of hard-coded at seven days.
  `0` disables the cooldown so a skipped or rejected article is offered again
  on the next run.
- Active article claims now use renewable five-minute leases, separate from the
  reconsider cooldown, so concurrent runs stay deduplicated even when the
  cooldown is disabled and interrupted work can be retried after lease expiry.
- `wallatag run --reset-seen` (also available on `manual`) clears all
  post-attempt cooldowns to requeue articles immediately, leaving live leases
  and the decision history untouched. In dry-run mode it only reports the count.

### Changed

- Tests are split into `unit`, `integration`, and `e2e` tiers, selectable with
  `WALLATAG_TEST_TIERS` (default `unit`). Integration tests (Prefect's
  ephemeral server, concurrent SQLite writers) now pin their own throwaway
  `PREFECT_HOME`, so a stale shared database can no longer break a run. Run all
  tiers with `WALLATAG_TEST_TIERS=all`.

## [0.1.0] - 2026-10-03

First public release. `wallatag` reads articles from your
[wallabag](https://wallabag.org/) instance and applies tags to them, either in
a manual terminal UI or automatically in batches, using a keyword tagger or an
optional LLM.

### Added

- CLI with three commands: `wallatag run` for automatic batch tagging,
  `wallatag manual` for reviewing and tagging one article at a time in the
  terminal, and `wallatag config` (`wallatag config show`, or the `wallatag
  status` alias) to inspect the resolved configuration.
- Keyword tagger using focus groups (`[focus.<name>]` config sections) with
  match modes and per-rule controls. Several focus groups can be combined by
  repeating the `--focus` CLI option, and rule keywords can be matched as
  regular expressions. Rules can additionally be restricted to tags already
  present on the instance via the `only-existing` tag policy.
- Optional LLM tagger backed by any OpenAI-compatible chat completions API,
  including providers that require an explicit `api_key`. LLM focus areas are
  matched per field just like the keyword tagger. The number of tags requested
  from the model (`max_proposals`) is separate from the number of tags
  ultimately applied (`max_applied_tags`); suggestions are sorted by
  confidence, and failed model calls are retried a bounded number of times and
  reported in the run summary. The base install only pulls `requests`; LLM use
  is a configuration choice, not an extra dependency.
- Optional keyword fallback when an LLM call ultimately fails, so an article is
  still tagged instead of being dropped.
- Optional verbose LLM output in the manual tagger to inspect model responses.
- Tag policies `only-existing`, `prefer-existing`, and `all` to control whether
  the tagger may invent tags or only reuse tags already on the instance.
- Vocabulary and ignore-tag handling: tags ignored for fetching can be excluded
  from the LLM's and the vocabulary tagger's view, with a configuration toggle.
- Local SQLite history store so articles are not re-tagged between runs, with a
  7-day claim cooldown so skipped or rejected articles become eligible again
  after a week. Articles whose tagging attempt fails are deferred for that
  cooldown instead of being retried immediately.
- Configuration layering: built-in defaults, then a TOML config file, then
  environment variables, then CLI flags. Config files are discovered from the
  current directory, `$XDG_CONFIG_HOME/wallatag/wallatag.toml`,
  `$HOME/.config/wallatag/wallatag.toml`, or `$XDG_CONFIG_DIRS`. A relative
  `[store] path` is resolved against the config file's directory, and a missing
  parent directory is created on first use.
- `wallatag config init` writes a commented example configuration to the XDG
  user config directory, or to a path given with `--config`.
- Prefect integration (optional extra): flows, blocks, and a Dokku deployment
  recipe for running wallatag on a schedule, splitting an hourly tag check from
  a nightly backfill and streaming run output into the flow log. Install with
  `pip install "wallatag[prefect]"`; Prefect is never a base dependency. See
  [docs/prefect.md](docs/prefect.md).
- `--dry-run`/`--no-apply` and `--no-history` flags for trying changes safely
  and for running without a history store.

### Changed

- The README is now written for people using the CLI directly; the Prefect
  deployment guide lives in [docs/prefect.md](docs/prefect.md) and the
  configuration reference in [docs/configuration.md](docs/configuration.md).
- `max_suggestions` was renamed to `max_applied_tags` (config and environment
  variable), and `max_proposals` was added for the LLM tagger.
- Environment variables take precedence over Prefect credential blocks when
  running as a flow.
- Failed Prefect tasks and flows now propagate their errors and carry failed
  results through, instead of being swallowed.

### Fixed

- Articles skipped or rejected by the tagger are no longer excluded from future
  runs forever. A claim now expires after 7 days, so an untagged article is
  reconsidered instead of silently dropped.
- LLM tag suggestions are sorted by confidence before truncation, so the
  highest-confidence tags are kept when a limit applies.
- Ollama base URLs no longer receive a doubled API version prefix.
- A user-provided confidence threshold outside the valid range is rejected
  instead of silently accepted.

[Unreleased]: https://github.com/marty-oehme/wallatag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marty-oehme/wallatag/releases/tag/v0.1.0
