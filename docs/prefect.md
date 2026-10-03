# Prefect integration and deployment

Prefect is optional:
wallatag works as a standalone CLI without it.
This guide covers the Prefect flow, UI configuration, and the project’s Dokku
deployment.
For CLI installation and use, start with the [README](../README.md).

## Deployment (Dokku)

In my setup wallatag is hosted on a Dokku server and runs its own Prefect worker
there.
The `worker` process keeps the container alive and joins the `wallatag-pool`
work pool; there is no web process.
The Prefect _server_ (in my case on an incus host) schedules runs, stores
results, and can notify on failures; the worker in this container executes them
by driving the wallatag tagging engine IN-PROCESS — one Prefect task per article
(`tag-article`), sharing the same engine code as the CLI — with each article's
logs and timing visible as its own task run in the dashboard.
Each `tag-article` task run also logs its per-article tagged line to that task
run in the dashboard — `tagged article <id> (<title>):
<tags>` — via the run logger, so applied tags are visible per article in the
UI/DB (the engine's own INFO lines don't surface in the flow).
A `git push dokku main` deploys the app and the flow together.

Requirements:
Dokku with the current default Python buildpack (heroku-buildpack-python ≥ v286,
i.e. uv support) and `dokku run` support.

One-time setup, run on the Dokku host:

```sh
dokku apps:create wallatag
dokku config:set wallatag WALLATAG_URL=https://your-wallabag.example WALLATAG_CLIENT_ID=... WALLATAG_CLIENT_SECRET=... WALLATAG_USERNAME=... WALLATAG_PASSWORD=... WALLATAG_DB=/data/wallatag.db PREFECT_API_URL=http://<prefect-server>:4200/api WALLATAG_AI_PROVIDER=ollama WALLATAG_AI_BASE_URL=http://<dokku-host-address>:11434 WALLATAG_AI_MODEL=<model>
dokku storage:ensure-directory wallatag
dokku storage:mount wallatag /var/lib/dokku/data/storage/wallatag:/data
dokku ps:scale wallatag worker=1
```

`WALLATAG_AI_API_KEY` is not in the `config:set` above:
it is optional and only needed for keyed openai-compatible gateways (OpenAI,
OpenRouter, ...).
If you use one, add `WALLATAG_AI_API_KEY=<key>` to the command (or set it via
`dokku config:set wallatag WALLATAG_AI_API_KEY=<key>` separately); for keyless
setups (e.g. local ollama) leave it unset.

The release phase auto-creates the `wallatag-llm` credentials block on every
push (deploy/release.py):
on first creation it is seeded from the `WALLATAG_AI_*` container env vars when
`WALLATAG_AI_PROVIDER`, `WALLATAG_AI_BASE_URL` and `WALLATAG_AI_MODEL` are all
set (the `WALLATAG_AI_API_KEY` env var is folded in too when present), otherwise
it is created empty.
The block's fields are provider, base_url, model, and an OPTIONAL `api_key`
(needed only for keyed openai-compatible providers).
Fill or edit it in the Prefect UI (Blocks > Wallatag LLM Credentials) to
configure the LLM for scheduled runs without redeploying.

For scheduled runs, the flow reads the block each run and merges its non-empty
fields into the `WALLATAG_AI_*` env vars it feeds wallatag's config loading.
Precedence for LLM settings, lowest to highest:
`wallatag.toml` defaults → LLM credentials block (defaults for scheduled runs) →
`WALLATAG_AI_*` env vars (`dokku config:set`).
So container env vars _override_ the block:
rotate or override credentials with `dokku config:set wallatag WALLATAG_AI_...`
and the change takes effect on scheduled runs without touching the block, while
editing the block in the Prefect UI changes the default.
To CLEAR a block value, use `dokku config:unset` (e.g. `dokku config:unset
wallatag WALLATAG_AI_MODEL`):
do NOT `config:set` it to an empty string:
a present-but-empty `WALLATAG_AI_*` var overrides the block with `""` and fails
wallatag's config validation (ConfigError) on every scheduled run.
The one exception is `WALLATAG_AI_API_KEY=""`, which intentionally clears the
api key (unset or empty means no auth header) instead of failing.
Empty block fields fall back to the TOML config / container env.
Block values apply to Prefect-scheduled runs only:
`dokku run wallatag ...` (manual/config) reads config/env only and never sees
the block, so keep the `WALLATAG_AI_*` vars in the `config:set` above if you
also run the LLM tagger manually.
No secrets end up in git either way.

The release phase auto-creates the shared `Wallabag Credentials` block too
(deploy/release.py), under the block document name `wallabag`, the same block
document that the morning-digest project reads, so one block on the Prefect
server serves both projects.
On first creation it is seeded from the `WALLATAG_*` container env vars when all
five are set (`WALLATAG_URL`, `WALLATAG_CLIENT_ID`, `WALLATAG_CLIENT_SECRET`,
`WALLATAG_USERNAME`, `WALLATAG_PASSWORD`), otherwise it is created empty.
The block's fields are base_url, client_id, client_secret, username and
password.
Fill or edit it in the Prefect UI (Blocks > Wallabag Credentials) to configure
the wallabag URL/credentials for scheduled runs without redeploying.

For scheduled runs, the flow reads the wallabag-credentials block each run and
merges its non-empty fields into the `WALLATAG_*` env vars it feeds wallatag's
config loading.
Precedence for wallabag credentials, lowest to highest:
`wallatag.toml` defaults → wallabag-credentials block (defaults for scheduled
runs) → `WALLATAG_*` env vars (`dokku config:set`).
So container env vars _override_ the block:
rotate or override credentials with `dokku config:set wallatag WALLATAG_...` and
the change takes effect on scheduled runs without touching the block, while
editing the block in the Prefect UI changes the default.
To CLEAR a block value, use `dokku config:unset` (e.g. `dokku config:unset
wallatag WALLATAG_URL`):
do NOT `config:set` it to an empty string:
a present-but-empty `WALLATAG_*` var overrides the block with `""` and fails
wallatag's config validation (ConfigError) on every scheduled run.
Empty block fields fall back to the TOML config / container env.
Block values apply to Prefect-scheduled runs only:
`dokku run wallatag ...` (manual/config) reads config/env only and never sees
the block, so keep the `WALLATAG_*` vars in the `config:set` above if you also
run wallatag manually.
No secrets end up in git either way.

### Flows configuration ownership

Three Prefect-side mechanisms feed the same config stack, each with a distinct
role per Prefect best practices:

| Role                        | Mechanism                          | What lives there                                                                                                                                                                     |
| --------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Per-run inputs              | Flow parameters (`wallatag_batch`) | `max_articles`, `focus` — what varies between runs/schedules                                                                                                                         |
| Shared non-secret settings  | Prefect Variables                  | the 12 lowercase `wallatag_*` scalars + the `wallatag_focus_groups` JSON (each name uppercases into the matching `WALLATAG_*` env var) — UI-editable without redeploy, never secrets |
| Secrets + connection config | Blocks                             | `wallatag-llm` (provider/base_url/model/api_key) and `wallabag` (url/client_id/client_secret/username/password), SecretStr-encrypted                                                 |
| Container-level paths       | dokku `config:set`                 | `WALLATAG_DB`, `WALLATAG_CONFIG`, `PREFECT_API_URL`, `PREFECT_API_KEY`                                                                                                               |
| Defaults                    | `wallatag.toml`                    | everything else                                                                                                                                                                      |

Rules of thumb:

- Flow parameters are for values that change per run or per schedule
  (per-schedule parameters are the documented mechanism for different config per
  cron).
  Keep the surface small and typed.
- Variables are for settings you want to change from the Prefect UI without
  redeploying.
  They are NOT encrypted — never put secrets in them.
- Blocks are for credentials and connection details; SecretStr fields are
  encrypted at rest and rotatable without redeploy.
- Precedence, lowest to highest:
  `wallatag.toml` → blocks → container env vars → Prefect Variables → flow
  parameters (`focus`, the CLI `--focus` equivalent).

### Prefect UI configuration (Variables)

Scalar settings and focus groups can also be configured from the Prefect UI
(Variables page) instead of container env vars — no redeploy needed, and the
values stay visible and editable in the UI.
Create a variable in the UI or with the CLI:

```sh
prefect variable set wallatag_tag_policy all
prefect variable set wallatag_enable_llm true
```

The 12 scalar variables are lowercase because Prefect requires variable names to
be lowercase; the flow uppercases them into the `WALLATAG_*` env vars the CLI
reads, so the names still mirror the env vars exactly:

| Variable                                | Description                                                            | Default                            |
| --------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------- |
| `wallatag_tag_policy`                   | `only-existing` \| `prefer-existing` \| `all`                          | `prefer-existing`                  |
| `wallatag_max_applied_tags`             | tags applied per article (int)                                         | `5`                                |
| `wallatag_ignore_tags`                  | comma-separated tags treated as untagged                               | `none (empty)`                     |
| `wallatag_ignore_tags_regex`            | regex patterns treated as untagged (comma-separated)                   | `none (empty)`                     |
| `wallatag_enable_vocabulary`            | existing-tag vocabulary matching on/off (bool)                         | `true`                             |
| `wallatag_enable_rules`                 | focus-group rule matching on/off (bool)                                | `true`                             |
| `wallatag_enable_llm`                   | LLM tagging on/off (bool)                                              | `false`                            |
| `wallatag_ai_confidence_threshold`      | LLM apply gate (float)                                                 | `0.7`                              |
| `wallatag_ai_use_focus_groups`          | focus groups in the LLM prompt on/off (bool)                           | `true`                             |
| `wallatag_ai_max_proposals`             | LLM tag proposals per article (int)                                    | unset → follows `max_applied_tags` |
| `wallatag_vocabulary_fields`            | vocabulary match fields (comma-separated)                              | `title,url,domain_name,content`    |
| `wallatag_vocabulary_skip_ignored_tags` | vocabulary matcher & LLM system prompt skip ignored tags on/off (bool) | `true`                             |

The Default column shows the built-in config default, used only when the
variable is unset and no container env var / `wallatag.toml` value overrides it
— an unset variable falls through to the container env var / `wallatag.toml`
value first.

Booleans are normalized to `true`/`false`, numbers to their plain string form —
the same values the env vars accept.
Focus groups go in ONE variable, `wallatag_focus_groups`, as a JSON object (all
four keys optional, values are lists of strings; an empty
`keywords`/`tags`/`keywords_regex` list omits that field, `fields:
[]` disables the group exactly like `WALLATAG_FOCUS_<NAME>_FIELDS=""`):

```sh
prefect variable set wallatag_focus_groups '{"methods": {"keywords": ["howto", "tutorial"], "tags": ["dev"], "fields": ["title", "url"], "keywords_regex": ["^how.?to"]}, "languages": {"tags": ["english"]}}'
```

It is translated to the usual
`WALLATAG_FOCUS_<NAME>_KEYWORDS`/`_TAGS`/`_FIELDS`/`_KEYWORDS_REGEX` env
convention (group names are lowercased).
The JSON value is validated strictly:
unknown keys (typo protection, e.g. `keywrods`), non-list fields, empty strings,
un-compilable `keywords_regex` patterns and `keywords_regex` patterns containing
a literal comma (which the comma-joined env translation could not represent)
make the run fail loudly instead of silently changing tagging.
An empty-string `wallatag_focus_groups` value (e.g. clearing the field in the
UI) also fails the run loudly — it is not a valid JSON object — so to remove
focus groups entirely, delete the variable rather than blanking it out.

Precedence for Prefect-scheduled runs, lowest to highest:
`wallatag.toml` defaults → blocks → container env vars (`dokku config:set`) →
Prefect Variables (scalar settings + `wallatag_focus_groups`) → the flow's
`focus` parameter (the CLI `--focus` equivalent).
So a variable overrides the container env var for that setting; the focus JSON
overrides same-named `WALLATAG_FOCUS_<NAME>_*` env vars for the fields it emits,
per-field (an empty `keywords`/`tags`/`keywords_regex` list omits that env var
entirely, so the container env var survives); and the flow's `focus` parameter
still wins.
Variables set only partially fall through:
a missing variable leaves the container env / TOML value in place for that
setting, so you can migrate settings to the UI one at a time.
Secrets never come from variables — `WALLATAG_CLIENT_SECRET`,
`WALLATAG_PASSWORD` and `WALLATAG_AI_API_KEY` stay in `dokku config:set` and the
credentials blocks.

The `wallatag_batch` flow itself takes two parameters (defaults shown):
`max_articles=50`, `focus=None`.
`focus` is a comma-separated list of focus-group names (e.g. `methods,
languages`), split in the flow layer and narrowed onto the config through the
shared `apply_run_overrides` helper (the CLI `--focus` equivalent); a group
whose NAME contains a literal comma can only be selected via the CLI, not via
the flow parameter.
The tag policy is not a flow parameter:
it is owned by the `wallatag_tag_policy` Prefect Variable (see the table below).

Process scaling is also declared via `app.json` (web 0, worker 1), so a fresh
deploy gets the right formation even before scaling is set by hand.

All secrets go via env, never in git:
`wallatag.toml` is gitignored and not used here.
`dokku storage:ensure-directory` creates the host directory and chowns it to the
container user (uid 32767), so the bind mount at `/data` works and keeps
`/data/wallatag.db` across redeploys.

`PREFECT_API_URL` points at your self-hosted Prefect server; the container
reaches it outbound (no inbound port needed on Dokku:
the worker polls).
Set `PREFECT_API_KEY` too only if the server enforces auth.

Deploy, from your local machine:

```sh
git remote add dokku dokku@your-host:wallatag
git push dokku main
```

The Procfile `release:` process type registers the deployment (running `prefect
deploy --all`) and creates the `wallatag-pool` work pool and the `wallatag-llm`
credentials block if they are missing, automatically on every push:
before the worker starts.
No manual `prefect deploy` step is needed.

Verify:

- Deploy output ends with success and no errors.
- `dokku ps` shows the `worker` process running (e.g. `wallatag.worker.1
  running`).
- On the incus host:
  the release phase creates `wallatag-pool` and the `wallatag-llm` block on
  first deploy; `prefect work-pool inspect wallatag-pool` shows it Ready once
  the worker heartbeats, and `prefect worker ls` lists the worker.
- In the Prefect UI, the `wallatag-batch` deployment shows scheduled runs, and
  each run's state (Completed/Failed) appears as it executes.
- `dokku run wallatag wallatag config` prints the config summary (proves the
  package and env vars work in-container).
  Note it exits 0 even with everything unset, so only treat env vars as working
  if the output shows the URL/username populated.
- `dokku run wallatag wallatag run --max 1 --no-apply` performs a real dry-run
  against the wallabag API (proves network and credentials).
- Redeploy (or `dokku ps:restart wallatag`) and confirm the worker comes back
  up.

Notes / troubleshooting:

- prefect is an optional dependency group; `bin/post_compile` installs it on
  Dokku builds only, so local installs stay lean (`uv sync` without `--group
  prefect`).
- The release phase needs `PREFECT_API_URL` (and `PREFECT_API_KEY` if the server
  enforces auth) to be set on the app BEFORE the first push, otherwise the
  deploy fails loudly:
  set config first, then push.
- The Python buildpack must support uv (v286+, May 2025).
  If `git push` fails while installing dependencies / detecting the package
  manager, the host buildpack is too old:
  update Dokku/herokuish on the host and redeploy.
  Do NOT add `requirements.txt` next to `uv.lock`:
  current buildpacks error on multiple package-manager files.
- `WALLATAG_URL` must be reachable from the container.
  If wallabag runs on the same host, use its public/trusted address, not
  `localhost`.
- The AI `base_url` must be reachable from inside the container:
  `localhost` refers to the container itself, so it only works if ollama runs on
  the same host as the Dokku worker (use the host's address instead).
- Interactive use is possible too:
  `dokku run wallatag wallatag manual --max 10` starts the manual tag review
  loop on demand (wallatag is still hosting-only; Prefect normally schedules the
  headless `run`).
  Note `dokku run` executes its command verbatim, so the second `wallatag` is
  the installed console script:
  there is no `web` process type.
