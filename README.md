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
| Use focus groups         | `[ai] use_focus_groups`     | `WALLATAG_AI_USE_FOCUS_GROUPS`     |
| AI fallback on failure   | `[ai] fallback_on_fail`     | `WALLATAG_AI_FALLBACK_ON_FAIL`     |
| Tag suggestions per article | `[tagger] max_suggestions` | `WALLATAG_MAX_SUGGESTIONS`         |
| Tag policy               | `[tagger] tag_policy`       | `WALLATAG_TAG_POLICY`              |
| Ignored tags             | `[tagger] ignore_tags`      | `WALLATAG_IGNORE_TAGS`             |
| Vocabulary matching      | `[tagger] enable_vocabulary`| `WALLATAG_ENABLE_VOCABULARY`       |
| Focus-group rules        | `[tagger] enable_rules`     | `WALLATAG_ENABLE_RULES`            |
| LLM classification       | `[tagger] enable_llm`       | `WALLATAG_ENABLE_LLM`              |
| Vocabulary match fields  | `[vocabulary] fields`       | `WALLATAG_VOCABULARY_FIELDS`       |
| Focus groups             | `[focus.<name>]` keywords/tags/fields/keywords_regex | `WALLATAG_FOCUS_<NAME>_KEYWORDS`, `WALLATAG_FOCUS_<NAME>_TAGS`, `WALLATAG_FOCUS_<NAME>_FIELDS`, `WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX` |
| Config file location     | `--config PATH`             | `WALLATAG_CONFIG`                  |

`WALLATAG_DB` set to an empty string means history-less mode (no database at
all). `WALLATAG_IGNORE_TAGS` is a comma-separated list (items are stripped of
whitespace); an empty string clears the TOML value. `wallatag.toml` contains
secrets and is gitignored; only `wallatag.toml.example` is committed.

### Focus groups

`[focus.<name>]` tables define named keyword/tag rule groups. `--focus NAME`
activates one group; the default is all groups.

Each `[focus.<name>]` table also accepts an optional `keywords_regex` list
(default empty): Python regex patterns matched against the group's fields *in
addition to* the literal `keywords`. A group fires when ANY literal keyword
matches OR ANY regex matches. Matching is per field, never across fields (same
as literal keywords), and case-insensitive by default. Case-sensitive sections
are possible with inline `(?-i:...)` overrides, e.g. `"(?-i:GTD)"` matches
`GTD` but not `gtd` — regex matching reads the article fields raw, so the
override works. Patterns are validated at load time: an invalid or
empty/whitespace-only pattern is a `ConfigError` naming the group and the
pattern (an empty regex matches everything, so it is rejected).

### Tagger settings

- `[tagger] max_suggestions`: how many tag suggestions per article (default 5).
  Override with `WALLATAG_MAX_SUGGESTIONS`.
- `--max N`: maximum articles processed per run (default: unlimited). This is
  a runtime-only flag, not a config-file key; it never changes
  `max_suggestions`.
- `[tagger] tag_policy`: `only-existing` | `prefer-existing` | `all`
  (default `prefer-existing`). Override with `--tag-policy` or
  `WALLATAG_TAG_POLICY`. `only-existing` never suggests tags that are not
  already in the wallabag vocabulary: vocabulary matches are kept, and
  focus-group rules still fire but are filtered down to their existing-tag
  results.
- `[tagger] ignore_tags`: list of tags treated as untagged. Articles carrying
  ONLY those tags are still fetched (e.g. maintenance tags like `fix`), while
  articles carrying any other tag are not. Matching is an exact full-string
  match, case-insensitive (`str.casefold()`). Default empty: only fully
  untagged articles are fetched. Override with `WALLATAG_IGNORE_TAGS`
  (comma-separated string, e.g. `fix,_frigo`); an empty value clears the list
  (whitespace-only or comma-only values are rejected, since they would
  silently clear it).
- Per-source match fields: the keyword tagger matches against the article's
  `title`, `url`, `domain_name` and `content` fields by default. Which fields
  are checked is configurable *per source*, the vocabulary matcher and each
  focus group are independent:
  - `[vocabulary] fields` (env `WALLATAG_VOCABULARY_FIELDS`) restricts the
    existing-tag vocabulary matcher (which fields existing labels are matched
    against).
  - `[focus.<name>] fields` (env `WALLATAG_FOCUS_<NAME>_FIELDS`) restricts
    that group's keywords AND regexes to its own subset; `--focus NAME` and
    `--tag-policy` are unaffected, and each group keeps its own fields.
  - A missing key means all four fields; an empty list `[]` (or `""` via env)
    disables that source entirely, it never matches. For a focus group, the
    disable is uniform across BOTH taggers: a group with `fields = []` also
    drops out of the LLM tagger's focus areas (its tags never appear in the
    LLM system prompt). Field names are validated strictly (exact,
    case-sensitive): only `title`, `url`, `domain_name`, `content` are
    accepted, anything else is a ConfigError. Env values are comma-separated
    (e.g. `title,url`); a non-empty value that parses to nothing (only
    separators or whitespace) is rejected. The vocabulary matcher is keyword
    only: the LLM tagger has no per-field restriction equivalent and is
    untouched by `[vocabulary] fields`.
- Empty values for `WALLATAG_TAG_POLICY` and `WALLATAG_MAX_SUGGESTIONS` are
  not clears, they raise a ConfigError, so remove those variables
  (`dokku config:unset`) rather than setting them to `""` (unlike
  `WALLATAG_IGNORE_TAGS`, `WALLATAG_DB`, and `WALLATAG_AI_API_KEY`, where
  empty means clear).
- `[ai]` enables the LLM tagger: `provider` (`ollama` or `openai-compatible`),
  `base_url`, and `model`; it is active iff `provider` is set *and*
  `[tagger] enable_llm = true` (or `WALLATAG_ENABLE_LLM=true`), otherwise the
  keyword tagger is used. LLM tagging is opt-in: `enable_llm` defaults to
  `false`, so configuring `[ai]` alone no longer activates the LLM tagger —
  it also requires the switch. `confidence_threshold` (default 0.7) gates
  headless apply, further limited by `--tag-policy`. LLM suggestions carry
  source `llm` and are recorded in the SQLite decision log. `api_key` is
  optional: when set it is sent as an `Authorization: Bearer <api_key>` header
  on every LLM request, which is only needed for keyed openai-compatible
  providers (OpenAI, OpenRouter, ...); unset or empty means no auth header.
- Per-source off-switches for the keyword tagger (both default `true`):
  `[tagger] enable_vocabulary = false` (env `WALLATAG_ENABLE_VOCABULARY`)
  disables existing-tag vocabulary matching; `[tagger] enable_rules = false`
  (env `WALLATAG_ENABLE_RULES`) disables focus-group rule matching. The
  switches are strict booleans (env accepts `true`/`1`/`yes` or
  `false`/`0`/`no`, case-insensitive) and compose with the existing
  `tag_policy` gate (`only-existing` keeps only rule suggestions whose tag
  already exists in the vocabulary; an off-switch disables its source
  regardless of policy).
- `[ai] use_focus_groups` (default `true`, env `WALLATAG_AI_USE_FOCUS_GROUPS`,
  accepts `true`/`1`/`yes` or `false`/`0`/`no`; values are matched
  case-insensitively and surrounding whitespace is ignored) controls
  whether focus groups influence LLM tagging. When `true` (default) the LLM
  system prompt carries a "Focus areas" line built from the focus groups the
  article matches (groups with `fields = []` are excluded). When `false`, the LLM
  ignores focus groups entirely, the "Focus areas" line is omitted from the
  prompt, so focus-group keywords remain meaningful only for the keyword
  tagger (keyword-only mode).
- `[ai] fallback_on_fail` (default `false`, env
  `WALLATAG_AI_FALLBACK_ON_FAIL`, accepts `true`/`1`/`yes` or
  `false`/`0`/`no`; values are matched case-insensitively and surrounding
  whitespace is ignored) enables a per-article keyword fallback: when the LLM
  tagger fails for an article (an `LLMError` from `suggest` after retries are
  exhausted), the keyword tagger takes over for THAT article — the LLM is
  still tried on subsequent articles. The fallback behaves exactly like a
  normal keyword-mode run: `enable_vocabulary`, `enable_rules` and
  `tag_policy` all apply, and its suggestions carry the usual `vocabulary`/
  `rules` sources into the decision log. A fallback that succeeds tags the
  article as usual and is surfaced in the run summary as `N via fallback`; a
  fallback that yields no suggestions (or itself fails) keeps today's
  LLM-failure path (`skipped`/`llm failures`, article deferred).

Focus groups can be defined or overridden via environment variables too:
`WALLATAG_FOCUS_<NAME>_KEYWORDS`, `WALLATAG_FOCUS_<NAME>_TAGS`,
`WALLATAG_FOCUS_<NAME>_FIELDS` and `WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX` map
to a `[focus.<name>]` group's `keywords`, `tags`, `fields` and `keywords_regex`.
The group name is the text between the `WALLATAG_FOCUS_`
prefix and the trailing `_KEYWORDS`/`_TAGS`/`_FIELDS`/`_KEYWORDS_REGEX` suffix,
and those exact suffixes are required (`WALLATAG_FOCUS_<NAME>` with no suffix
is ignored).
Names may contain underscores: only the trailing suffix is stripped, so
`WALLATAG_FOCUS_METHODS_KEYWORDS_TAGS` is group `methods_keywords` with its
`tags` field set (mind the nesting). Values are comma-separated (items
stripped of whitespace, empty items dropped, e.g. `fix,_frigo`); a non-empty
value that parses to nothing (only separators or whitespace) is rejected, and
an empty value clears (disables) that field, for `_FIELDS`, `""` disables the
group's keyword matching entirely. A regex containing a literal comma cannot
be expressed via `WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX`: the value is split on
every comma and each fragment is then validated INDEPENDENTLY, so the split
can SILENTLY change matching with no error (e.g. `"^a,b$"` becomes the two
patterns `^a` and `b$`, both valid) — use TOML `keywords_regex` for
comma-containing patterns. The `WALLATAG_FOCUS_GROUPS` JSON variable rejects
comma-containing `keywords_regex` items loudly for the same reason. Group
names are case-insensitive and
groups merge by name: an env var overrides the same-named TOML group
per-field (only the fields it sets),
env-only groups are created with the missing field defaulting to empty, and
TOML groups with no env counterpart survive unchanged. `--focus NAME`
selection is unchanged and works on the merged result.

## Note on wallabag's native regex tagging rules

wallabag has a built-in, regex-based tagging rules feature. It is managed in
the wallabag UI, is applied *only to new entries at save time*, and has no
API: so it cannot back-fill existing entries or be scripted. Users who just
want simple regex rulesets on new entries should use that feature directly.

wallatag covers the gaps:

- *Back-fill*: tagging articles that are already in the wallabag account.
- *Vocabulary consistency*: sharing a common vocabulary across keyword rules
  and focus groups.
- *Interactive review*: human-in-the-loop confirmation before tags are
  applied (`manual`).
- *AI tagging*: LLM-based suggestion.

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
pool; there is no web process. The Prefect *server* (on your incus host)
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
*override* the block: rotate or override credentials with `dokku config:set
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
(deploy/release.py), under the block document name `wallabag`, the same block
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
*override* the block: rotate or override credentials with `dokku config:set
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

### Prefect UI configuration (Variables)

Scalar settings and focus groups can also be configured from the Prefect UI
(Variables page) instead of container env vars — no redeploy needed, and the
values stay visible and editable in the UI. Create a variable in the UI or
with the CLI:

```sh
prefect variable set WALLATAG_TAG_POLICY all
prefect variable set WALLATAG_ENABLE_LLM true
```

The 9 scalar variables (names mirror the env vars exactly):

| Variable | Meaning |
| -------- | ------- |
| `WALLATAG_TAG_POLICY` | `only-existing` \| `prefer-existing` \| `all` |
| `WALLATAG_MAX_SUGGESTIONS` | tag suggestions per article (int) |
| `WALLATAG_IGNORE_TAGS` | comma-separated tags treated as untagged |
| `WALLATAG_ENABLE_VOCABULARY` | existing-tag vocabulary matching on/off (bool) |
| `WALLATAG_ENABLE_RULES` | focus-group rule matching on/off (bool) |
| `WALLATAG_ENABLE_LLM` | LLM tagging on/off (bool) |
| `WALLATAG_AI_CONFIDENCE_THRESHOLD` | LLM apply gate (float, default 0.7) |
| `WALLATAG_AI_USE_FOCUS_GROUPS` | focus groups in the LLM prompt on/off (bool) |
| `WALLATAG_VOCABULARY_FIELDS` | vocabulary match fields (comma-separated) |

Booleans are normalized to `true`/`false`, numbers to their plain string form
— the same values the env vars accept. Focus groups go in ONE variable,
`WALLATAG_FOCUS_GROUPS`, as a JSON object (all four keys optional, values are
lists of strings; an empty `keywords`/`tags`/`keywords_regex` list omits that
field, `fields: []` disables the group exactly like
`WALLATAG_FOCUS_<NAME>_FIELDS=""`):

```sh
prefect variable set WALLATAG_FOCUS_GROUPS '{"methods": {"keywords": ["howto", "tutorial"], "tags": ["dev"], "fields": ["title", "url"], "keywords_regex": ["^how.?to"]}, "languages": {"tags": ["english"]}}'
```

It is translated to the usual
`WALLATAG_FOCUS_<NAME>_KEYWORDS`/`_TAGS`/`_FIELDS`/`_KEYWORDS_REGEX`
env convention (group names are lowercased). The JSON value is validated
strictly: unknown keys (typo protection, e.g. `keywrods`), non-list fields,
empty strings, un-compilable `keywords_regex` patterns and `keywords_regex`
patterns containing a literal comma (which the comma-joined env translation
could not represent) make the run fail loudly instead of silently changing
tagging.
An empty-string `WALLATAG_FOCUS_GROUPS` value (e.g. clearing the field in the
UI) also fails the run loudly — it is not a valid JSON object — so to remove
focus groups entirely, delete the variable rather than blanking it out.

Precedence for Prefect-scheduled runs, lowest to highest: `wallatag.toml`
defaults → blocks → container env vars (`dokku config:set`) → Prefect
Variables (scalar settings + `WALLATAG_FOCUS_GROUPS`) → CLI options
(`--tag-policy`/`--focus`). So a variable overrides the container env var for
that setting; the focus JSON overrides same-named `WALLATAG_FOCUS_<NAME>_*`
env vars for the fields it emits, per-field (an empty `keywords`/`tags`/
`keywords_regex` list omits that env var entirely, so the container env var
survives); and CLI flags
still win. Variables set only partially fall through:
a missing variable leaves the container env / TOML value in place for that
setting, so you can migrate settings to the UI one at a time. Secrets never
come from variables — `WALLATAG_CLIENT_SECRET`, `WALLATAG_PASSWORD` and
`WALLATAG_AI_API_KEY` stay in `dokku config:set` and the credentials blocks.

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
