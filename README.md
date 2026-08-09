# wallatag

AI-assisted auto-tagger for [wallabag](https://wallabag.org/). MVP: an
argparse-based CLI that reads wallabag entries, suggests tags, and applies them
through the wallabag REST API.

Python 3.11+ (uses stdlib `tomllib`), single runtime dependency: `requests`.
No web endpoint, no framework, plain terminal prompts (no TUI).

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/); the lockfile
`uv.lock` is committed and is the source of truth. `pyproject.toml` declares
the project metadata and the `requests` dependency.

```sh
uv sync                              # create .venv and install dependencies
cp wallatag.toml.example wallatag.toml     # then edit url/client_id/client_secret/username/password
uv run python -m wallatag --help
uv run python -m wallatag status
uv run python -m wallatag run --no-apply   # dry run
uv run python -m wallatag manual           # interactive review loop
```

Installed as a console script too: `uv run wallatag run` (used e.g. by
`dokku run wallatag ...`).

## Configuration layering

Values are merged from lowest to highest precedence: later sources win:

1. Built-in defaults
2. `wallatag.toml` in the current directory (or `--config PATH`)
3. Environment variables
4. CLI flags

| Setting                  | TOML                        | Environment variable               |
| ------------------------ | --------------------------- | ---------------------------------- |
| wallabag URL             | `[wallabag] url`            | `WALLATAG_URL`                     |
| API client id            | `[wallabag] client_id`      | `WALLATAG_CLIENT_ID`               |
| API client secret        | `[wallabag] client_secret`  | `WALLATAG_CLIENT_SECRET`           |
| wallabag username        | `[wallabag] username`       | `WALLATAG_USERNAME`                |
| wallabag password        | `[wallabag] password`       | `WALLATAG_PASSWORD`                |
| SQLite decision log path | `[store] path`              | `WALLATAG_DB`                      |
| AI provider              | `[ai] provider`             | `WALLATAG_AI_PROVIDER`             |
| AI base URL              | `[ai] base_url`             | `WALLATAG_AI_BASE_URL`             |
| AI model                 | `[ai] model`                | `WALLATAG_AI_MODEL`                |
| AI confidence threshold  | `[ai] confidence_threshold` | `WALLATAG_AI_CONFIDENCE_THRESHOLD` |
| AI API key               | `[ai] api_key`              | `WALLATAG_AI_API_KEY`              |
| Config file location     | `--config PATH`             | `WALLATAG_CONFIG`                  |

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
- `[tagger] ignore_tags`: list of tags treated as untagged. Articles carrying
  ONLY those tags are still fetched (e.g. maintenance tags like `fix`), while
  articles carrying any other tag are not. Matching is an exact full-string
  match, case-insensitive (`str.casefold()`). Default empty: only fully
  untagged articles are fetched.
- `[ai]` enables the LLM tagger: `provider` (`ollama` or `openai-compatible`),
  `base_url`, and `model`; it is active iff `provider` is set, otherwise the
  keyword tagger is used. `confidence_threshold` (default 0.7) gates headless
  apply, further limited by `--tag-policy`. LLM suggestions carry source `llm`
  and are recorded in the SQLite decision log. `api_key` is optional: when set
  it is sent as an `Authorization: Bearer <api_key>` header on every LLM
  request, which is only needed for keyed openai-compatible providers (OpenAI,
  OpenRouter, ...); unset or empty means no auth header.

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
- **AI tagging**: LLM-based suggestion.

## Commands

| Command | Purpose                                              | Status   |
| ------- | ---------------------------------------------------- | -------- |
| `status`| Print a non-secret configuration summary            | MVP      |
| `run`   | Headless one-shot batch tagging                      | MVP      |
| `manual`| Interactive tag review loop                          | MVP      |

`--no-history` disables the decision log for a run; `--no-apply` is a dry run
that changes nothing.

## Deployment (Dokku)

wallatag is hosted on a Dokku server and runs its own Prefect worker there. The
`worker` process keeps the container alive and joins the `wallatag-pool` work
pool; there is no web process. The Prefect **server** (on your incus host)
schedules runs, stores results, and can notify you on failures; the worker in
this container executes them by running the installed `wallatag run` against
the wallabag API. A `git push dokku main` deploys the app and the flow together.

Requirements: Dokku with the current default Python buildpack
(heroku-buildpack-python ≥ v286, i.e. uv support) and `dokku run` support.

One-time setup, run on the Dokku host:

```sh
dokku apps:create wallatag
dokku config:set wallatag WALLATAG_URL=https://your-wallabag.example WALLATAG_CLIENT_ID=... WALLATAG_CLIENT_SECRET=... WALLATAG_USERNAME=... WALLATAG_PASSWORD=... WALLATAG_DB=/data/wallatag.db PREFECT_API_URL=http://<prefect-server>:4200/api WALLATAG_AI_PROVIDER=ollama WALLATAG_AI_BASE_URL=http://<dokku-host-address>:11434 WALLATAG_AI_MODEL=<model>
dokku storage:ensure-directory wallatag
dokku storage:mount wallatag /var/lib/dokku/data/storage/wallatag:/data
dokku ps:scale wallatag worker=1
```

`WALLATAG_AI_API_KEY` is not in the `config:set` above: it is optional and only
needed for keyed openai-compatible gateways (OpenAI, OpenRouter, ...). If you
use one, add `WALLATAG_AI_API_KEY=<key>` to the command (or set it via
`dokku config:set wallatag WALLATAG_AI_API_KEY=<key>` separately); for keyless
setups (e.g. local ollama) leave it unset.

The release phase auto-creates the `wallatag-llm` credentials block on every
push (deploy/release.py): on first creation it is seeded from the
`WALLATAG_AI_*` container env vars when `WALLATAG_AI_PROVIDER`,
`WALLATAG_AI_BASE_URL` and `WALLATAG_AI_MODEL` are all set (the
`WALLATAG_AI_API_KEY` env var is folded in too when present), otherwise it is
created empty. The block's fields are provider, base_url, model,
confidence_threshold, and an OPTIONAL `api_key` (needed only for keyed
openai-compatible providers). Fill or edit it in the Prefect UI (Blocks >
Wallatag LLM Credentials) to configure the LLM for scheduled runs without
redeploying.

For scheduled runs, the flow reads the block each run and passes its non-empty
fields to the wallatag CLI as `WALLATAG_AI_*` env vars. Precedence for LLM
settings, lowest to highest: `wallatag.toml` defaults → LLM credentials block
(defaults for scheduled runs) → `WALLATAG_AI_*` env vars (`dokku config:set`)
→ CLI options (none exist for AI config today). So container env vars
**override** the block: rotate or override credentials with `dokku config:set
wallatag WALLATAG_AI_...` and the change takes effect on scheduled runs
without touching the block, while editing the block in the Prefect UI changes
the default. To CLEAR a block value, use `dokku config:unset` (e.g. `dokku
config:unset wallatag WALLATAG_AI_MODEL`): do NOT `config:set` it to an
empty string: a present-but-empty `WALLATAG_AI_*` var overrides the block with
`""` and fails wallatag's config validation (ConfigError) on every scheduled
run. The one exception is `WALLATAG_AI_API_KEY=""`, which intentionally
clears the api key (unset or empty means no auth header) instead of failing.
Empty block fields fall back to the TOML config / container env.
Block values apply to Prefect-scheduled runs only: `dokku run wallatag ...`
(manual/status) reads config/env only and never sees the block, so keep the
`WALLATAG_AI_*` vars in the `config:set` above if you also run the LLM tagger
manually. No secrets end up in git either way.

The release phase auto-creates the shared `Wallabag Credentials` block too
(deploy/release.py), under the block document name `wallabag` — the same block
document that the morning-digest project reads, so one block on the Prefect
server serves both projects. On first creation it is seeded from the
`WALLATAG_*` container env vars when all five are set (`WALLATAG_URL`,
`WALLATAG_CLIENT_ID`, `WALLATAG_CLIENT_SECRET`, `WALLATAG_USERNAME`,
`WALLATAG_PASSWORD`), otherwise it is created empty. The block's fields are
base_url, client_id, client_secret, username and password. Fill or edit it in
the Prefect UI (Blocks > Wallabag Credentials) to configure the wallabag
URL/credentials for scheduled runs without redeploying.

For scheduled runs, the flow reads the wallabag-credentials block each run and
passes its non-empty fields to the wallatag CLI as `WALLATAG_*` env vars.
Precedence for wallabag credentials, lowest to highest: `wallatag.toml`
defaults → wallabag-credentials block (defaults for scheduled runs) →
`WALLATAG_*` env vars (`dokku config:set`). So container env vars
**override** the block: rotate or override credentials with `dokku config:set
wallatag WALLATAG_...` and the change takes effect on scheduled runs without
touching the block, while editing the block in the Prefect UI changes the
default. To CLEAR a block value, use `dokku config:unset` (e.g. `dokku
config:unset wallatag WALLATAG_URL`): do NOT `config:set` it to an empty
string: a present-but-empty `WALLATAG_*` var overrides the block with `""` and
fails wallatag's config validation (ConfigError) on every scheduled run.
Empty block fields fall back to the TOML config / container env. Block values
apply to Prefect-scheduled runs only: `dokku run wallatag ...`
(manual/status) reads config/env only and never sees the block, so keep the
`WALLATAG_*` vars in the `config:set` above if you also run wallatag manually.
No secrets end up in git either way.

Process scaling is also declared via `app.json` (web 0, worker 1), so a fresh
deploy gets the right formation even before scaling is set by hand.

All secrets go via env, never in git: `wallatag.toml` is gitignored and not
used here. `dokku storage:ensure-directory` creates the host directory and
chowns it to the container user (uid 32767), so the bind mount at `/data` works
and keeps `/data/wallatag.db` across redeploys.

`PREFECT_API_URL` points at your self-hosted Prefect server; the container
reaches it outbound (no inbound port needed on Dokku: the worker polls). Set
`PREFECT_API_KEY` too only if the server enforces auth.

Deploy, from your local machine:

```sh
git remote add dokku dokku@your-host:wallatag
git push dokku main
```

The Procfile `release:` process type registers the deployment (running
`prefect deploy --all`) and creates the `wallatag-pool` work pool and the
`wallatag-llm` credentials block if they are missing, automatically on every
push: before the worker starts. No manual `prefect deploy` step is needed.

Verify:

- Deploy output ends with success and no errors.
- `dokku ps` shows the `worker` process running (e.g. `wallatag.worker.1 running`).
- On the incus host: the release phase creates `wallatag-pool` and the
  `wallatag-llm` block on first deploy; `prefect work-pool inspect
  wallatag-pool` shows it Ready once the worker heartbeats, and
  `prefect worker ls` lists the worker.
- In the Prefect UI, the `wallatag-batch` deployment shows scheduled runs, and
  each run's state (Completed/Failed) appears as it executes.
- `dokku run wallatag wallatag status` prints the config summary (proves the
  package and env vars work in-container). Note it exits 0 even with everything
  unset, so only treat env vars as working if the output shows the
  URL/username populated.
- `dokku run wallatag wallatag run --max 1 --no-apply` performs a real dry-run
  against the wallabag API (proves network and credentials).
- Redeploy (or `dokku ps:restart wallatag`) and confirm the worker comes back up.

Notes / troubleshooting:

- prefect is an optional dependency group; `bin/post_compile` installs it on
  Dokku builds only, so local installs stay lean (`uv sync` without
  `--group prefect`).
- The release phase needs `PREFECT_API_URL` (and `PREFECT_API_KEY` if the
  server enforces auth) to be set on the app BEFORE the first push, otherwise
  the deploy fails loudly: set config first, then push.
- The Python buildpack must support uv (v286+, May 2025). If `git push` fails
  while installing dependencies / detecting the package manager, the host
  buildpack is too old: update Dokku/herokuish on the host and redeploy. Do NOT
  add `requirements.txt` next to `uv.lock`: current buildpacks error on multiple
  package-manager files.
- `WALLATAG_URL` must be reachable from the container. If wallabag runs on the
  same host, use its public/trusted address, not `localhost`.
- The AI `base_url` must be reachable from inside the container: `localhost`
  refers to the container itself, so it only works if ollama runs on the same
  host as the Dokku worker (use the host's address instead).
- Interactive use is possible too: `dokku run wallatag wallatag manual --max 10`
  starts the manual tag review loop on demand (wallatag is still hosting-only;
  Prefect normally schedules the headless `run`). Note `dokku run` executes its
  command verbatim, so the second `wallatag` is the installed console script:
  there is no `web` process type.

## Development

```sh
uv run python -m unittest discover -s tests -v
```

Scheduling is handled exclusively by Prefect (see `prefect_flows.py`); wallatag
itself has zero Prefect dependency. `prefect` is installed via the optional
`prefect` dependency group (`uv sync --group prefect`): needed only when
developing flows or rebuilding the Dokku image (see `bin/post_compile`).
