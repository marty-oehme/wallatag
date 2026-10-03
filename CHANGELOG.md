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
  match modes and per-rule controls.
- Optional LLM tagger backed by any OpenAI-compatible chat completions API.
  The base install only pulls `requests`; LLM use is a configuration choice,
  not an extra dependency.
- Tag policies `only-existing`, `prefer-existing`, and `all` to control whether
  the tagger may invent tags or only reuse tags already on the instance.
- Local SQLite history store so articles are not re-tagged between runs, with a
  7-day claim cooldown so skipped or rejected articles become eligible again
  after a week.
- Configuration layering: built-in defaults, then a TOML config file, then
  environment variables, then CLI flags. Config files are discovered from the
  current directory, `$XDG_CONFIG_HOME/wallatag/wallatag.toml`,
  `$HOME/.config/wallatag/wallatag.toml`, or `$XDG_CONFIG_DIRS`. A relative
  `[store] path` is resolved against the config file's directory, and a missing
  parent directory is created on first use.
- Prefect integration (optional extra): flows, blocks, and a Dokku deployment
  recipe for running wallatag on a schedule. Install with
  `pip install "wallatag[prefect]"`; Prefect is never a base dependency. See
  [docs/prefect.md](docs/prefect.md).
- `--dry-run`/`--no-apply` and `--no-history` flags for trying changes safely
  and for running without a history store.

### Changed

- The README is now written for people using the CLI directly; the Prefect
  deployment guide lives in [docs/prefect.md](docs/prefect.md) and the
  configuration reference in [docs/configuration.md](docs/configuration.md).

### Fixed

- Articles skipped or rejected by the tagger are no longer excluded from future
  runs forever. A claim now expires after 7 days, so an untagged article is
  reconsidered instead of silently dropped.

[Unreleased]: https://github.com/marty-oehme/wallatag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marty-oehme/wallatag/releases/tag/v0.1.0
