# Configuration reference

This reference covers the general-purpose CLI configuration.
For an overview and first run, see the [README](../README.md).
Prefect-specific configuration belongs in the
[Prefect integration guide](prefect.md).

## Configuration layering

Values are merged from lowest to highest precedence:
later sources win:

1. Built-in defaults
2. A TOML config file (discovery order below)
3. Environment variables
4. CLI flags

| Setting                                              | TOML                                                 | Environment variable                                                                                                                   |
| ---------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| wallabag URL                                         | `[wallabag] url`                                     | `WALLATAG_URL`                                                                                                                         |
| API client id                                        | `[wallabag] client_id`                               | `WALLATAG_CLIENT_ID`                                                                                                                   |
| API client secret                                    | `[wallabag] client_secret`                           | `WALLATAG_CLIENT_SECRET`                                                                                                               |
| wallabag username                                    | `[wallabag] username`                                | `WALLATAG_USERNAME`                                                                                                                    |
| wallabag password                                    | `[wallabag] password`                                | `WALLATAG_PASSWORD`                                                                                                                    |
| SQLite decision log path                             | `[store] path`                                       | `WALLATAG_DB`                                                                                                                          |
| AI provider                                          | `[ai] provider`                                      | `WALLATAG_AI_PROVIDER`                                                                                                                 |
| AI base URL                                          | `[ai] base_url`                                      | `WALLATAG_AI_BASE_URL`                                                                                                                 |
| AI model                                             | `[ai] model`                                         | `WALLATAG_AI_MODEL`                                                                                                                    |
| AI confidence threshold                              | `[ai] confidence_threshold`                          | `WALLATAG_AI_CONFIDENCE_THRESHOLD`                                                                                                     |
| AI API key                                           | `[ai] api_key`                                       | `WALLATAG_AI_API_KEY`                                                                                                                  |
| Use focus groups                                     | `[ai] use_focus_groups`                              | `WALLATAG_AI_USE_FOCUS_GROUPS`                                                                                                         |
| AI fallback on failure                               | `[ai] fallback_on_fail`                              | `WALLATAG_AI_FALLBACK_ON_FAIL`                                                                                                         |
| Tag proposals per article (LLM)                      | `[ai] max_proposals`                                 | `WALLATAG_AI_MAX_PROPOSALS`                                                                                                            |
| Tag suggestions applied per article                  | `[tagger] max_applied_tags`                          | `WALLATAG_MAX_APPLIED_TAGS`                                                                                                            |
| Tag policy                                           | `[tagger] tag_policy`                                | `WALLATAG_TAG_POLICY`                                                                                                                  |
| Ignored tags                                         | `[tagger] ignore_tags`                               | `WALLATAG_IGNORE_TAGS`                                                                                                                 |
| Ignored tag patterns                                 | `[tagger] ignore_tags_regex`                         | `WALLATAG_IGNORE_TAGS_REGEX`                                                                                                           |
| Vocabulary matching                                  | `[tagger] enable_vocabulary`                         | `WALLATAG_ENABLE_VOCABULARY`                                                                                                           |
| Focus-group rules                                    | `[tagger] enable_rules`                              | `WALLATAG_ENABLE_RULES`                                                                                                                |
| LLM classification                                   | `[tagger] enable_llm`                                | `WALLATAG_ENABLE_LLM`                                                                                                                  |
| Vocabulary match fields                              | `[vocabulary] fields`                                | `WALLATAG_VOCABULARY_FIELDS`                                                                                                           |
| Skip ignored tags (vocabulary matching & LLM prompt) | `[vocabulary] skip_ignored_tags`                     | `WALLATAG_VOCABULARY_SKIP_IGNORED_TAGS`                                                                                                |
| Focus groups                                         | `[focus.<name>]` keywords/tags/fields/keywords_regex | `WALLATAG_FOCUS_<NAME>_KEYWORDS`, `WALLATAG_FOCUS_<NAME>_TAGS`, `WALLATAG_FOCUS_<NAME>_FIELDS`, `WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX` |
| Config file location                                 | `--config PATH`                                      | `WALLATAG_CONFIG`                                                                                                                      |

`WALLATAG_DB` set to an empty string means history-less mode (no database at
all).
`WALLATAG_IGNORE_TAGS` is a comma-separated list (items are stripped of
whitespace); an empty string clears the TOML value.
`wallatag.toml` contains secrets and is gitignored; only `wallatag.toml.example`
is committed.

Inspect the effective configuration with `wallatag config` (alias: `wallatag
config show`; the old `wallatag status` name still works).

### Config file discovery

wallatag looks for a TOML config file in this order and uses the first one
that exists:

1. `--config PATH`
2. `WALLATAG_CONFIG` environment variable
3. `./wallatag.toml` in the current working directory (project-local)
4. `$XDG_CONFIG_HOME/wallatag/wallatag.toml`, defaulting to
   `~/.config/wallatag/wallatag.toml` when `XDG_CONFIG_HOME` is unset
5. `wallatag/wallatag.toml` under each `$XDG_CONFIG_DIRS` entry, defaulting
   to `/etc/xdg`

A project-local file therefore overrides a per-user XDG config. `--config` and
`WALLATAG_CONFIG` must point at an existing file; the other candidates are
simply skipped when absent.

A relative `[store] path` is resolved against the directory of the config file
that declared it, so an XDG-discovered config keeps its database beside itself
regardless of the working directory. Absolute paths are used as-is, and a
`WALLATAG_DB` value is never rewritten.

### Focus groups

`[focus.<name>]` tables define named keyword/tag rule groups.
`--focus NAME` activates one group; repeat the flag (e.g. `--focus methods
--focus languages`) to activate several; the default is all groups.

Each `[focus.<name>]` table also accepts an optional `keywords_regex` list
(default empty):
Python regex patterns matched against the group's fields _in addition to_ the
literal `keywords`.
A group fires when ANY literal keyword matches OR ANY regex matches.
Matching is per field, never across fields (same as literal keywords), and
case-insensitive by default.
Case-sensitive sections are possible with inline `(?-i:...)` overrides, e.g.
`"(?-i:GTD)"` matches `GTD` but not `gtd` — regex matching reads the article
fields raw, so the override works.
Patterns are validated at load time:
an invalid or empty/whitespace-only pattern is a `ConfigError` naming the group
and the pattern (an empty regex matches everything, so it is rejected).

### Tagger settings

- `[tagger] max_applied_tags`:
  hard cap on the number of tags applied per article, used by BOTH taggers
  (final truncation; default 5).
  This key was renamed (the old key and its env var are gone, with no
  compatibility shim); override with `WALLATAG_MAX_APPLIED_TAGS`.
- `[ai] max_proposals`:
  how many tags the LLM is asked to propose per article (default:
  unset -> follows `max_applied_tags`).
  The LLM system prompt asks for at most this many tags; the applied list is
  always capped by `max_applied_tags`.
  KeywordTagger has no such knob.
  Override with `WALLATAG_AI_MAX_PROPOSALS`; an empty env value clears it (back
  to following `max_applied_tags`).
- `--max N`:
  maximum articles processed per run (default:
  unlimited).
  This is a runtime-only flag, not a config-file key; it never changes
  `max_applied_tags`.
- `[tagger] tag_policy`:
  `only-existing` | `prefer-existing` | `all` (default `prefer-existing`).
  Override with `--tag-policy` or `WALLATAG_TAG_POLICY`.
  `only-existing` never suggests tags that are not already in the wallabag
  vocabulary:
  vocabulary matches are kept, and focus-group rules still fire but are filtered
  down to their existing-tag results.
- `[tagger] ignore_tags`:
  list of tags treated as untagged.
  Articles carrying ONLY those tags are still fetched (e.g. maintenance tags
  like `fix`), while articles carrying any other tag are not.
  Matching is an exact full-string match, case-insensitive (`str.casefold()`).
  Default empty:
  only fully untagged articles are fetched.
  Override with `WALLATAG_IGNORE_TAGS` (comma-separated string, e.g.
  `fix,_frigo`); an empty value clears the list (whitespace-only or comma-only
  values are rejected, since they would silently clear it).
  By default ignored tags are also skipped wherever the vocabulary is used:
  the vocabulary matcher skips tags on the ignore lists (exact or regex match,
  same semantics as the fetch filter), and the LLM tagger excludes them from the
  "Existing tag vocabulary" line of its system prompt too (and drops a
  model-proposed ignored tag from its suggestions even if the model names one),
  so maintenance tags are neither re-fetched nor re-applied; set `[vocabulary]
  skip_ignored_tags = false` (env `WALLATAG_VOCABULARY_SKIP_IGNORED_TAGS`) to
  apply ignored tags again.
- `[tagger] ignore_tags_regex`:
  list of Python regex patterns (default empty) matched against each tag _in
  addition to_ the literal `ignore_tags` list.
  A tag counts as ignored if it equals a literal entry OR matches any pattern,
  so articles carrying ONLY tags that are literal-ignored or match a pattern are
  still fetched.
  Patterns are substring matches (`re.search`), so anchor with `^...$` for
  full-tag matching.
  Matching is case-insensitive by default; case-sensitive sections are possible
  with inline `(?-i:...)` overrides (regexes read the raw tag label, so the
  override works).
  Patterns are validated at load time:
  an invalid or empty/whitespace-only pattern is a `ConfigError` naming the key
  (an empty regex matches everything, so it is rejected).
  Override with `WALLATAG_IGNORE_TAGS_REGEX` (comma-separated, e.g.
  `^todo$,(?-i:^fix$)`); an empty value clears the list (whitespace-only or
  comma-only values are rejected).
- Per-source match fields:
  the keyword tagger matches against the article's `title`, `url`, `domain_name`
  and `content` fields by default.
  Which fields are checked is configurable _per source_, the vocabulary matcher
  and each focus group are independent:
  - `[vocabulary] fields` (env `WALLATAG_VOCABULARY_FIELDS`) restricts the
    existing-tag vocabulary matcher (which fields existing labels are matched
    against).
  - `[focus.<name>] fields` (env `WALLATAG_FOCUS_<NAME>_FIELDS`) restricts that
    group's keywords AND regexes to its own subset; `--focus NAME` and
    `--tag-policy` are unaffected, and each group keeps its own fields.
  - A missing key means all four fields; an empty list `[]` (or `""` via env)
    disables that source entirely, it never matches.
    For a focus group, the disable is uniform across BOTH taggers:
    a group with `fields = []` also drops out of the LLM tagger's focus areas
    (its tags never appear in the LLM system prompt).
    Field names are validated strictly (exact, case-sensitive):
    only `title`, `url`, `domain_name`, `content` are accepted, anything else is
    a ConfigError.
    Env values are comma-separated (e.g. `title,url`); a non-empty value that
    parses to nothing (only separators or whitespace) is rejected.
    The vocabulary matcher is keyword only:
    the LLM tagger has no per-field restriction equivalent and is untouched by
    `[vocabulary] fields`.
- Empty values for `WALLATAG_TAG_POLICY` and `WALLATAG_MAX_APPLIED_TAGS` are not
  clears, they raise a ConfigError, so remove those variables (`dokku
  config:unset`) rather than setting them to `""` (unlike
  `WALLATAG_IGNORE_TAGS`, `WALLATAG_DB`, and `WALLATAG_AI_API_KEY`, where empty
  means clear).
- `[ai]` enables the LLM tagger:
  `provider` (`ollama` or `openai-compatible`), `base_url`, and `model`; it is
  active iff `provider` is set _and_ `[tagger] enable_llm = true` (or
  `WALLATAG_ENABLE_LLM=true`), otherwise the keyword tagger is used.
  LLM tagging is opt-in:
  `enable_llm` defaults to `false`, so configuring `[ai]` alone no longer
  activates the LLM tagger — it also requires the switch.
  `confidence_threshold` (default 0.7) gates headless apply, further limited by
  `--tag-policy`.
  LLM suggestions carry source `llm` and are recorded in the SQLite decision
  log.
  Applied LLM tags are ranked by model confidence, highest first (ties keep the
  model's output order).
  `api_key` is optional:
  when set it is sent as an `Authorization:
  Bearer <api_key>` header on every LLM request, which is only needed for keyed
  openai-compatible providers (OpenAI, OpenRouter, ...); unset or empty means no
  auth header.
- Per-source off-switches for the keyword tagger (both default `true`):
  `[tagger] enable_vocabulary = false` (env `WALLATAG_ENABLE_VOCABULARY`)
  disables existing-tag vocabulary matching; `[tagger] enable_rules = false`
  (env `WALLATAG_ENABLE_RULES`) disables focus-group rule matching.
  The switches are strict booleans (env accepts `true`/`1`/`yes` or
  `false`/`0`/`no`, case-insensitive) and compose with the existing `tag_policy`
  gate (`only-existing` keeps only rule suggestions whose tag already exists in
  the vocabulary; an off-switch disables its source regardless of policy).
- `[ai] use_focus_groups` (default `true`, env `WALLATAG_AI_USE_FOCUS_GROUPS`,
  accepts `true`/`1`/`yes` or `false`/`0`/`no`; values are matched
  case-insensitively and surrounding whitespace is ignored) controls whether
  focus groups influence LLM tagging.
  When `true` (default) the LLM system prompt carries a "Focus areas" line built
  from the focus groups the article matches (groups with `fields = []` are
  excluded).
  When `false`, the LLM ignores focus groups entirely, the "Focus areas" line is
  omitted from the prompt, so focus-group keywords remain meaningful only for
  the keyword tagger (keyword-only mode).
- `[ai] fallback_on_fail` (default `false`, env `WALLATAG_AI_FALLBACK_ON_FAIL`,
  accepts `true`/`1`/`yes` or `false`/`0`/`no`; values are matched
  case-insensitively and surrounding whitespace is ignored) enables a
  per-article keyword fallback:
  when the LLM tagger fails for an article (an `LLMError` from `suggest` after
  retries are exhausted), the keyword tagger takes over for THAT article — the
  LLM is still tried on subsequent articles.
  The fallback behaves exactly like a normal keyword-mode run:
  `enable_vocabulary`, `enable_rules` and `tag_policy` all apply, and its
  suggestions carry the usual `vocabulary`/ `rules` sources into the decision
  log.
  A fallback that succeeds tags the article as usual and is surfaced in the run
  summary as `N via fallback`; a fallback that yields no suggestions (or itself
  fails) keeps today's LLM-failure path (`skipped`/`llm failures`, article
  deferred).

Focus groups can be defined or overridden via environment variables too:
`WALLATAG_FOCUS_<NAME>_KEYWORDS`, `WALLATAG_FOCUS_<NAME>_TAGS`,
`WALLATAG_FOCUS_<NAME>_FIELDS` and `WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX` map to
a `[focus.<name>]` group's `keywords`, `tags`, `fields` and `keywords_regex`.
The group name is the text between the `WALLATAG_FOCUS_` prefix and the trailing
`_KEYWORDS`/`_TAGS`/`_FIELDS`/`_KEYWORDS_REGEX` suffix, and those exact suffixes
are required (`WALLATAG_FOCUS_<NAME>` with no suffix is ignored).
Names may contain underscores:
only the trailing suffix is stripped, so `WALLATAG_FOCUS_METHODS_KEYWORDS_TAGS`
is group `methods_keywords` with its `tags` field set (mind the nesting).
Values are comma-separated (items stripped of whitespace, empty items dropped,
e.g. `fix,_frigo`); a non-empty value that parses to nothing (only separators or
whitespace) is rejected, and an empty value clears (disables) that field, for
`_FIELDS`, `""` disables the group's keyword matching entirely.
A regex containing a literal comma cannot be expressed via
`WALLATAG_FOCUS_<NAME>_KEYWORDS_REGEX`:
the value is split on every comma and each fragment is then validated
INDEPENDENTLY, so the split can SILENTLY change matching with no error (e.g.
`"^a,b$"` becomes the two patterns `^a` and `b$`, both valid) — use TOML
`keywords_regex` for comma-containing patterns.
The `wallatag_focus_groups` JSON variable rejects comma-containing
`keywords_regex` items loudly for the same reason.
The same rule applies to `WALLATAG_IGNORE_TAGS_REGEX`:
its value is split on every comma and each fragment validated independently, so
regexes containing a literal comma must be configured via TOML `[tagger]
ignore_tags_regex`.
Group names are case-insensitive and groups merge by name:
an env var overrides the same-named TOML group per-field (only the fields it
sets), env-only groups are created with the missing field defaulting to empty,
and TOML groups with no env counterpart survive unchanged.
`--focus NAME` selection is unchanged and works on the merged result.
