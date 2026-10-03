# wallatag

**A CLI for finding and tagging your untagged [Wallabag](https://wallabag.org/)
articles.**

wallatag can apply tags from your existing Wallabag vocabulary, match keywords
and rules you define, or ask an optional LLM for suggestions.
Use `run` for a batch job or `manual` to review suggestions in your terminal.
By default it runs as a command line application, but it can optionally be
integrated into [Prefect](https://www.prefect.io/) scheduling; see the
[Prefect integration guide](docs/prefect.md).
See [CHANGELOG.md](CHANGELOG.md) for what changed between releases.

Minimum requirements:

- Python 3.11 or newer
- A Wallabag instance and API credentials
- A virtual environment utility like [uv](https://docs.astral.sh/uv/) for installation or running from source

The CLI does not require Prefect or an LLM.
Its only runtime dependency is `requests`.

## Install and configure

Install through pip, pipx or uv:

```sh
uv tool install git+https://github.com/marty-oehme/wallatag.git
```

Then generate a starter configuration file:

```sh
wallatag config init
```

This writes an annotated `wallatag.toml` to
`$XDG_CONFIG_HOME/wallatag/wallatag.toml` (`~/.config/wallatag/wallatag.toml`
by default), creating the directory if needed. Pass `--config PATH` to write it
elsewhere (for a project-local `./wallatag.toml`) and `--force` to overwrite an
existing file.

Edit that file with your Wallabag URL, API client ID and secret, username,
and password.
Create an API client in Wallabag under **Developer** ->  **Create your own client**.
A project-local `wallatag.toml` is gitignored so your credentials stay local.

For a local SQLite history file, set:

```toml
[store]
path = "./wallatag.db"
```

A user-wide config should keep its database out of the config directory, e.g.
`path = "~/.local/share/wallatag/wallatag.db"`.

You can also supply settings through environment variables instead of TOML.
Configuration precedence is built-in defaults, a TOML config file, environment
variables, then command-line flags.
wallatag looks for `./wallatag.toml` first, then
`$XDG_CONFIG_HOME/wallatag/wallatag.toml` (`~/.config/...` by default) and the
`$XDG_CONFIG_DIRS` entries (`/etc/xdg`). Use `--config PATH` or
`WALLATAG_CONFIG` to point at a specific file.
A relative `[store] path` is resolved against the config file's directory.
The [configuration reference](docs/configuration.md) covers all settings; the
annotated example config is also available in the repository as
[`wallatag/wallatag.toml.example`](wallatag/wallatag.toml.example).

Check the effective (non-secret) configuration before connecting:

```sh
wallatag config
```

`wallatag config show` is the explicit form, and `wallatag status` remains as an
alias.

Start with a dry run.
It fetches eligible articles and reports what it would tag without applying
changes or writing history:

```sh
uv run wallatag run --max 5 --no-apply
```

To review suggestions one article at a time, use the interactive mode:

```sh
uv run wallatag manual --max 5
```

At each article, press `Enter` to apply the current suggestions, `a` to add
tags, `d` to drop suggestions, `s` to skip, or `q` to quit.
A skipped article is eligible again after the seven-day cooldown when history is
enabled.

For unattended batch tagging, omit `--no-apply`:

```sh
uv run wallatag run --max 5
```

`wallatag --help`, `wallatag run --help`, and `wallatag manual --help` list the
available options.
Common flags:

| Flag                  | Purpose                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| `--max N`             | Limit a run to at most N articles.                                                                     |
| `--focus NAME`        | Activate selected focus groups; repeat to choose several. By default, all groups are active.           |
| `--tag-policy POLICY` | Choose `only-existing`, `prefer-existing` (default), or `all`.                                         |
| `--no-apply`          | Preview without applying tags or writing history.                                                      |
| `--no-history`        | Run without the SQLite store; entries can recur on every run and concurrent runs are not deduplicated. |
| `--config PATH`       | Read settings from a specific TOML file.                                                               |
| `--verbose`           | Print more diagnostic information.                                                                     |

## How tagging works

- Existing-tag vocabulary: match Wallabag’s tags against article fields.
  By default, matching checks the title, URL, domain, and content.
- Focus groups: define keywords that map to tags, then select groups with
  `--focus`.
  Groups can also match regular expressions and choose which article fields to
  inspect.
- Optional LLM suggestions: disabled by default.
  Enable with `[tagger] enable_llm = true` and configure an `ollama` or
  `openai-compatible` provider under `[ai]`.
  The confidence threshold gates headless application; manual mode lets you
  review suggestions first.
  When LLM tagging is enabled, the article title, URL, domain, language, reading
  time, and content excerpt are sent to the configured model endpoint.

`prefer-existing` favors tags already in your Wallabag vocabulary while still
allowing new suggestions.
`only-existing` prevents new tags; `all` allows them.
The final number of tags applied per article is capped by `[tagger]
max_applied_tags` (default:
5).

### History and reconsideration

With history enabled, wallatag atomically claims an article when processing
starts.
When CLI and Prefect runs share the same SQLite store, this prevents them from
double-processing an article.
The claim expires after seven days:
skipped, rejected-wholesale, no-suggestion, interrupted, or deterministically
failed articles can return to the queue instead of being excluded forever.
Transient and LLM failures are requeued immediately.
Without history (`--no-history` or an empty `WALLATAG_DB`), there is no
persistent cooldown or cross-run deduplication.

## Versus Wallabag's built-in rules

Wallabag has native regex tagging rules, but they apply only to new entries and
cannot be managed through its API.
wallatag complements them by backfilling existing entries, applying shared
vocabulary and focus-group rules, offering interactive review, and optionally
using an LLM.

## Prefect integration

Prefect is an optional way to schedule wallatag and monitor each article as a
Prefect task.
The standalone CLI works without it.
For flow behavior, UI configuration, and the Dokku deployment, see the dedicated
[Prefect integration and deployment guide](docs/prefect.md).

## Development

### Clone from source

Clone the project and install its locked dependencies:

```sh
git clone https://github.com/marty-oehme/wallatag.git
cd wallatag
uv sync --locked
uv run wallatag config init --config ./wallatag.toml
```

### Version control

I develop wallatag with [Jujutsu](https://jj-vcs.github.io/jj/) (`jj`), a
Git-compatible VCS. You do not need it: the repository is a normal Git
repository, and contributions through plain `git` (branches, commits, pull
requests) are just as welcome. Use whichever you prefer.

### Install dev dependencies

Install development and optional integration dependencies, then run the checks:

```sh
uv sync --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -t .
```

Tests are grouped into tiers (`unit`, `integration`, `e2e`) selected with the
`WALLATAG_TEST_TIERS` environment variable; the default is `unit` only, so the
command above is the fast loop. Use `WALLATAG_TEST_TIERS=all` to run
everything, or e.g. `WALLATAG_TEST_TIERS=unit,integration` to skip `e2e`.
Integration tests need the optional `prefect` group and exercise real child
processes (Prefect's ephemeral API server, concurrent SQLite writers) over
fakes for the network; there are no `e2e` tests yet. The `-t .` flag is
required for the selection to take effect (it imports `tests/` as a package).

Woodpecker CI runs linting, formatting, and all test tiers on pushes to `main`
and pull requests targeting `main`.

## License

[MIT](LICENSE)
