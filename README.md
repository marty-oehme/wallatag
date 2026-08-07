# wallatag

AI-assisted auto-tagger for [wallabag](https://wallabag.org/). MVP: an
argparse-based CLI that reads wallabag entries, suggests tags, and applies them
through the wallabag REST API.

Python 3.11+ (uses stdlib `tomllib`), single runtime dependency: `requests`.
No web endpoint, no framework, plain terminal prompts (no TUI).

## Quick start

```sh
pip install -r requirements.txt            # or: pip install .
cp wallatag.toml.example wallatag.toml     # then edit url/client_id/client_secret
python -m wallatag --help
python -m wallatag status
python -m wallatag run --no-apply          # dry run
python -m wallatag manual                  # interactive review loop (later)
```

Installed as a console script too: `wallatag run` (used e.g. by
`dokku run wallatag ...`).

## Configuration layering

Values are merged from lowest to highest precedence: later sources win:

1. Built-in defaults
2. `wallatag.toml` in the current directory (or `--config PATH`)
3. Environment variables
4. CLI flags

| Setting                  | TOML                        | Environment variable          |
| ------------------------ | --------------------------- | ----------------------------- |
| wallabag URL             | `[wallabag] url`            | `WALLATAG_URL`                |
| API client id            | `[wallabag] client_id`      | `WALLATAG_CLIENT_ID`          |
| API client secret        | `[wallabag] client_secret`  | `WALLATAG_CLIENT_SECRET`      |
| SQLite decision log path | `[store] path`              | `WALLATAG_DB`                 |
| Config file location     | `--config PATH`             | `WALLATAG_CONFIG`             |

`WALLATAG_DB` set to an empty string means history-less mode (no database at
all). `wallatag.toml` contains secrets and is gitignored; only
`wallatag.toml.example` is committed.

### Focus groups

`[focus.<name>]` tables define named keyword/tag rule groups. `--focus NAME`
activates one group; the default is all groups.

### Tagger settings

- `[tagger] max_suggestions`: how many tag suggestions per article (default 5).
- `--max N`: maximum articles processed per run (default: unlimited). This is
  a runtime-only flag, not a config-file key; it never changes
  `max_suggestions`.
- `[tagger] tag_policy`: `only-existing` | `prefer-existing` | `all`
  (default `prefer-existing`). Override with `--tag-policy`.
- `[ai]` is reserved for phase 2 (LLM tagger) and ignored by the MVP.

## Note on wallabag's native regex tagging rules

wallabag has a built-in, regex-based tagging rules feature. It is managed in
the wallabag UI, is applied **only to new entries at save time**, and has no
API: so it cannot back-fill existing entries or be scripted. Users who just
want simple regex rulesets on new entries should use that feature directly.

wallatag covers the gaps:

- **Back-fill**: tagging articles that are already in the wallabag account.
- **Vocabulary consistency**: sharing a common vocabulary across keyword rules
  and focus groups.
- **Interactive review**: human-in-the-loop confirmation before tags are
  applied (`manual`).
- **AI tagging**: LLM-based suggestion (phase 2).

## Commands

| Command | Purpose                                              | Status   |
| ------- | ---------------------------------------------------- | -------- |
| `status`| Print a non-secret configuration summary            | MVP      |
| `run`   | Headless batch tagging                               | stub     |
| `manual`| Interactive review loop                              | stub     |

`--no-history` disables the decision log for a run; `--no-apply` is a dry run
that changes nothing.

## Development

```sh
python3 -m unittest discover -s tests -v
```

Scheduling is handled exclusively by Prefect (see `prefect_flows.py`); wallatag
itself has zero Prefect dependency. `Procfile` only keeps the dokku container
alive: no worker loops here.
