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

| Setting                  | TOML                        | Environment variable          |
| ------------------------ | --------------------------- | ----------------------------- |
| wallabag URL             | `[wallabag] url`            | `WALLATAG_URL`                |
| API client id            | `[wallabag] client_id`      | `WALLATAG_CLIENT_ID`          |
| API client secret        | `[wallabag] client_secret`  | `WALLATAG_CLIENT_SECRET`      |
| wallabag username        | `[wallabag] username`       | `WALLATAG_USERNAME`           |
| wallabag password        | `[wallabag] password`       | `WALLATAG_PASSWORD`           |
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
| `run`   | Headless one-shot batch tagging                      | MVP      |
| `manual`| Interactive tag review loop                          | MVP      |

`--no-history` disables the decision log for a run; `--no-apply` is a dry run
that changes nothing.

## Deployment (Dokku)

wallatag is hosted on a Dokku server and runs its own Prefect worker there. The
`prefect` process keeps the container alive and joins the `wallatag-pool` work
pool; there is no web process. The Prefect **server** (on your incus host)
schedules runs, stores results, and can notify you on failures; the worker in
this container executes them by running the installed `wallatag run` against
the wallabag API. A `git push dokku main` deploys the app and the flow together.

Requirements: Dokku with the current default Python buildpack
(heroku-buildpack-python ≥ v286, i.e. uv support) and `dokku run` support.

One-time setup, run on the Dokku host:

```sh
dokku apps:create wallatag
dokku config:set wallatag WALLATAG_URL=https://your-wallabag.example WALLATAG_CLIENT_ID=... WALLATAG_CLIENT_SECRET=... WALLATAG_USERNAME=... WALLATAG_PASSWORD=... WALLATAG_DB=/data/wallatag.db PREFECT_API_URL=http://<prefect-server>:4200/api
dokku storage:ensure-directory wallatag
dokku storage:mount wallatag /var/lib/dokku/data/storage/wallatag:/data
dokku ps:scale wallatag prefect=1
```

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

Register the deployment (from the deployed container, so the entrypoint
resolves against the code in /app):

```sh
dokku run wallatag prefect deploy
```

Verify:

- Deploy output ends with success and no errors.
- `dokku ps` shows the `prefect` process running (e.g. `wallatag.prefect.1 running`).
- On the incus host: `prefect work-pool inspect wallatag-pool` shows the pool
  Ready (a worker is heartbeating) and `prefect worker ls` lists the worker.
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
- The Python buildpack must support uv (v286+, May 2025). If `git push` fails
  while installing dependencies / detecting the package manager, the host
  buildpack is too old: update Dokku/herokuish on the host and redeploy. Do NOT
  add `requirements.txt` next to `uv.lock`: current buildpacks error on multiple
  package-manager files.
- `WALLATAG_URL` must be reachable from the container. If wallabag runs on the
  same host, use its public/trusted address, not `localhost`.
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
