"""Tag suggestion engine.

Public types: ``TagSuggestion`` and the ``Tagger`` protocol. Two
implementations sit behind the same protocol:

- ``KeywordTagger``: deterministic keyword-rule groups plus the existing-tag
  vocabulary. Pure logic: no I/O.
- ``LLMTagger``: an LLM (via the injectable ``LLMClientLike`` client) proposes
  tags and self-reports a per-tag confidence. Suggestions below
  ``confidence_threshold`` are dropped inside ``suggest()``; the pipelines
  apply what is left, further gated by ``tag_policy``. The decision log
  (store.py) records accept/reject decisions with source ``"llm"`` as the
  training/eval signal (issue 4); the model confidence itself is NOT
  persisted: the decisions table has no confidence column: and the signal
  is documented but not consumed at runtime in this phase.
"""

from __future__ import annotations

import html
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from wallatag.config import VALID_MATCH_FIELDS, VALID_TAG_POLICIES, FocusGroup
from wallatag.llm import LLMError

_VOCABULARY_CONFIDENCE = 1.0
_RULE_CONFIDENCE = 0.7
# Hard cap on the article text sent to the model: long HTML bodies would blow
# up the prompt and cost tokens without adding signal.
_MAX_CONTENT_CHARS = 6000

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_content(text: str) -> str:
    """Strip HTML tags and unescape entities, then collapse whitespace.

    Deliberately local to tagger.py (manual.py's strip_html is not imported):
    the LLM prompt needs a plain-text view of the article, and keeping the
    helper here leaves the tagger self-contained and import-cycle-free.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _existing_tag_keys(existing_tags: Iterable[str]) -> frozenset[str]:
    """Normalized (stripped, casefolded) existing vocabulary labels.

    Single source of truth for membership in the existing tag vocabulary,
    shared by both taggers' "only-existing" policy gates. Non-string or
    blank labels can never be suggested tags, so they are skipped.
    """
    return frozenset(
        tag.strip().casefold()
        for tag in existing_tags
        if isinstance(tag, str) and tag.strip()
    )


_DEFAULT_MATCH_FIELDS = VALID_MATCH_FIELDS


def _field_needles(entry: dict, fields) -> tuple[str, ...]:
    """Casefolded per-field article text, fields kept separate.

    ``fields`` names which article keys to read, so a source pinned to
    ``("title",)`` never sees needles from the content field.
    """
    needles = []
    for key in fields:
        value = entry.get(key)
        if isinstance(value, str) and value:
            needles.append(value.casefold())
    return tuple(needles)


def _field_values(entry: dict, fields) -> tuple[str, ...]:
    """Raw per-field article text for regex matching, fields kept separate.

    Parallel to ``_field_needles`` but WITHOUT casefolding: regexes carry
    their own case handling (compiled with ``re.IGNORECASE``), and inline
    flags like ``(?-i:...)`` must see the original text to take effect.
    """
    values = []
    for key in fields:
        value = entry.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(values)


def _matches(needle: object, fields: tuple[str, ...]) -> bool:
    """True if the (stripped, casefolded) needle is in at least one field.

    Matching is per-field, so a multi-word needle never matches across
    field boundaries. Blank/whitespace-only or non-string needles never
    match.
    """
    if not isinstance(needle, str):
        return False
    folded = needle.strip().casefold()
    return bool(folded) and any(folded in field for field in fields)


def _compile_group_regexes(groups) -> dict[str, tuple[re.Pattern, ...]]:
    """Compile each group's keywords_regex, validating them eagerly.

    Shared by both taggers: the LLM prompt builder matches focus groups
    with the same regex semantics as the keyword tagger's rules.
    """
    compiled: dict[str, tuple[re.Pattern, ...]] = {}
    for name, group in groups.items():
        patterns: list[re.Pattern] = []
        for pattern in group.keywords_regex:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"focus group {name!r}: keywords_regex contains an "
                    f"empty, whitespace-only or non-string pattern {pattern!r}"
                )
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise ValueError(
                    f"focus group {name!r}: keywords_regex contains invalid "
                    f"regex {pattern!r}: {exc}"
                ) from exc
        compiled[name] = tuple(patterns)
    return compiled


def _group_fires(group, entry: dict, regexes: tuple[re.Pattern, ...]) -> bool:
    """True if the article matches the group within its configured fields.

    Same semantics as the keyword tagger's rule matching: the group's field
    subset (None -> all four fields; () -> never fires), ANY literal keyword
    matching (casefolded substring, per field) OR ANY regex searching the
    RAW field values (re.IGNORECASE, per-field, never across fields).
    """
    group_fields = group.fields if group.fields is not None else _DEFAULT_MATCH_FIELDS
    fields = _field_needles(entry, group_fields)
    if any(_matches(kw, fields) for kw in group.keywords):
        return True
    if regexes:
        values = _field_values(entry, group_fields)
        return any(
            pattern.search(field) for pattern in regexes for field in values
        )
    return False


@dataclass(frozen=True)
class TagSuggestion:
    tag: str
    source: str
    confidence: float = 1.0


class Tagger(Protocol):
    def suggest(self, entry: dict) -> list[TagSuggestion]:
        ...


class LLMClientLike(Protocol):
    """The minimal client surface LLMTagger needs (implemented by LLMClient).

    Declared here so LLMTagger stays testable with a stub and never imports
    wallatag.llm at runtime beyond the error type.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class KeywordTagger:
    """Keyword tagger: deterministic rule groups plus the existing-tag vocabulary.

    Pure logic: no I/O and no dependency on the wallabag client. Matching is a
    case-insensitive raw substring check, evaluated per field: a keyword or an
    existing tag label must be a substring of at least one of the entry's
    title, url, domain_name and content fields individually: a multi-word
    needle never matches across field boundaries. The content is HTML; raw
    substring matching against it is acceptable for the MVP.

    Each focus group may additionally carry ``keywords_regex`` patterns: the
    group fires when ANY literal keyword matches OR ANY regex matches (searched
    per field against the RAW field values, never across fields). Regexes are
    compiled with ``re.IGNORECASE`` by default; inline overrides like
    ``(?-i:GTD)`` work because matching reads the original text, not the
    casefolded needles literal matching uses.

    Which fields are matched is configurable PER SOURCE. The vocabulary
    matcher (existing-tag labels) checks ``vocabulary_fields`` (default: all
    four fields); each focus group checks ``group.fields`` when set, otherwise
    the same all-four default. Absent/None means all four fields; an
    explicitly empty tuple disables that source entirely (it never matches).
    Field names are strict: only the exact ``VALID_MATCH_FIELDS`` names are
    accepted, at config-load time (ConfigError) and at construction
    (ValueError).

    tag_policy semantics: both "prefer-existing" and "all" produce the same
    ordering: vocabulary suggestions first, then rule suggestions, with
    case-insensitive first-wins dedup. "only-existing" never suggests tags
    that are not already in the existing vocabulary: vocabulary matches are
    kept and rule suggestions are filtered to tags present in the vocabulary
    (they fire, but cannot introduce new tags).

    Two per-source off-switches (keyword-only, both default True):
    ``enable_vocabulary=False`` disables the existing-tag vocabulary matcher
    (no ``source="vocabulary"`` suggestions, whatever ``vocabulary_fields``
    says); ``enable_rules=False`` disables focus-group rule matching (no
    ``source="rules"`` suggestions, whatever the groups say). Both are
    strict booleans; anything else raises ValueError at construction.

    By default (``skip_ignored_tags=True``) the vocabulary matcher skips
    tags on the ``ignore_tags`` / ``ignore_tags_regex`` lists: a tag that is
    both an existing vocabulary label and ignored (exact casefolded match
    against ``ignore_tags``, or ``re.search`` against ``ignore_tags_regex``,
    mirroring the wallabag fetch-side semantics) is never suggested from the
    vocabulary. Set ``skip_ignored_tags=False`` to restore the old behavior of
    suggesting ignored vocabulary labels. Rule-derived suggestions and the
    LLM path are unaffected: an ignored tag fired by a focus group still
    survives as a ``source="rules"`` suggestion.
    """

    # Default set of article fields matched against when a source does not
    # pin its own: aliased to the shared module constant (same value as
    # config.VALID_MATCH_FIELDS) so there is one source of truth
    # (KeywordTagger.__init__ accepts only those names too).
    _FIELDS = _DEFAULT_MATCH_FIELDS

    def __init__(
        self,
        focus_groups: Mapping[str, FocusGroup],
        *,
        max_applied_tags: int,
        tag_policy: str,
        existing_tags: Iterable[str] = (),
        vocabulary_fields: Iterable[str] = _FIELDS,
        enable_vocabulary: bool = True,
        enable_rules: bool = True,
        ignore_tags: Iterable[str] = (),
        ignore_tags_regex: Iterable[str] = (),
        skip_ignored_tags: bool = True,
    ) -> None:
        if tag_policy not in VALID_TAG_POLICIES:
            raise ValueError(
                "invalid tag_policy %r; valid choices: %s"
                % (tag_policy, ", ".join(VALID_TAG_POLICIES))
            )
        if (
            not isinstance(max_applied_tags, int)
            or isinstance(max_applied_tags, bool)
            or max_applied_tags < 0
        ):
            raise ValueError("max_applied_tags must be a non-negative integer")
        if not isinstance(enable_vocabulary, bool):
            raise ValueError("enable_vocabulary must be a boolean")
        if not isinstance(enable_rules, bool):
            raise ValueError("enable_rules must be a boolean")
        if not isinstance(skip_ignored_tags, bool):
            raise ValueError("skip_ignored_tags must be a boolean")
        # Materialize ONCE before validating: vocabulary_fields is only
        # declared Iterable, so a one-shot generator would otherwise be
        # consumed by the validation loop and the stored tuple would come out
        # empty (vocabulary silently disabled). Mirrors the single-iteration
        # existing_tags pattern above.
        vocabulary_fields = tuple(vocabulary_fields)
        for field in vocabulary_fields:
            if field not in VALID_MATCH_FIELDS:
                raise ValueError(
                    "invalid field %r; valid choices: %s"
                    % (field, ", ".join(VALID_MATCH_FIELDS))
                )
        # dict preserves config insertion order -> deterministic iteration.
        self.focus_groups = dict(focus_groups)
        # Pre-compile each group's regex patterns ONCE via the shared helper
        # (the LLM prompt builder matches focus groups with the same
        # semantics). config.py already validated compilability, so the
        # ValueError backstop is for groups constructed directly (mirroring
        # the enable_* ValueError style).
        self._group_regexes: dict[str, tuple[re.Pattern, ...]] = _compile_group_regexes(
            self.focus_groups
        )
        self.max_applied_tags = max_applied_tags
        self.tag_policy = tag_policy
        # Labels are normalized for matching but suggested in their original
        # casing (wallabag labels as provided).
        self.existing_tags = list(existing_tags)
        self.vocabulary_fields = vocabulary_fields
        self.enable_vocabulary = enable_vocabulary
        self.enable_rules = enable_rules
        # Ignore lists mirrored from the wallabag fetch side (wallabag.py
        # iter_untagged/untagged_entries/_should_fetch): exact matching is
        # casefolded (with surrounding whitespace stripped, consistent with
        # _existing_tag_keys); regexes search the RAW label with
        # re.IGNORECASE. config.py already validated compilability, so the
        # patterns are compiled directly.
        self._ignored_keys: frozenset[str] = frozenset(
            tag.strip().casefold()
            for tag in ignore_tags
            if isinstance(tag, str) and tag.strip()
        )
        self._ignored_patterns: tuple[re.Pattern, ...] = tuple(
            re.compile(p, re.IGNORECASE) for p in ignore_tags_regex
        )
        self.skip_ignored_tags = skip_ignored_tags

    def _field_needles(self, entry: dict, fields) -> tuple[str, ...]:
        """Casefolded per-field text to match against, fields kept separate.

        ``fields`` names which article keys to read (the configured per-source
        subset), so a source pinned to ``("title",)`` never sees needles from
        the content field.
        """
        return _field_needles(entry, fields)

    def _field_values(self, entry: dict, fields) -> tuple[str, ...]:
        """Raw per-field text for regex matching, fields kept separate.

        Parallel to ``_field_needles`` but WITHOUT casefolding: regexes carry
        their own case handling (compiled with ``re.IGNORECASE``), and inline
        flags like ``(?-i:...)`` must see the original text to take effect.
        """
        return _field_values(entry, fields)

    @staticmethod
    def _matches(needle: object, fields: tuple[str, ...]) -> bool:
        """True if the (stripped, casefolded) needle is in at least one field.

        Matching is per-field, so a multi-word needle never matches across
        field boundaries. Blank/whitespace-only or non-string needles never
        match.
        """
        return _matches(needle, fields)

    def _ignored(self, tag: str) -> bool:
        """True if the tag is on the ignore lists (fetch-side semantics).

        Exact matching is casefolded (and whitespace-stripped on both
        sides); regexes search the RAW label (``re.search``, compiled with
        ``re.IGNORECASE``) — mirroring wallabag.py's ``_should_fetch``.
        """
        key = tag.strip().casefold()
        if key in self._ignored_keys:
            return True
        return any(p.search(tag) for p in self._ignored_patterns)

    def _vocabulary_suggestions(self, fields: tuple[str, ...]) -> list[TagSuggestion]:
        suggestions = []
        for label in self.existing_tags:
            if self._matches(label, fields):
                suggestions.append(
                    TagSuggestion(
                        tag=label,
                        source="vocabulary",
                        confidence=_VOCABULARY_CONFIDENCE,
                    )
                )
        return suggestions

    def _rule_suggestions(self, entry: dict) -> list[TagSuggestion]:
        """Rule suggestions, matching each group against its OWN field subset.

        A group with ``fields`` set matches its keywords/regexes against only
        those article fields; ``fields=None`` falls back to the class default
        (all four); an explicitly empty tuple means the group never matches.
        A group fires when ANY literal keyword matches (the current
        case-insensitive substring semantics) OR ANY regex matches (compiled
        with ``re.IGNORECASE``, searched against the RAW field values,
        per-field like the literals, never across fields).
        """
        suggestions = []
        for name, group in self.focus_groups.items():
            if not _group_fires(group, entry, self._group_regexes[name]):
                continue
            for tag in group.tags:
                if isinstance(tag, str) and tag.strip():
                    suggestions.append(
                        TagSuggestion(
                            tag=tag, source="rules", confidence=_RULE_CONFIDENCE
                        )
                    )
        return suggestions

    def suggest(self, entry: dict) -> list[TagSuggestion]:
        vocabulary = (
            self._vocabulary_suggestions(self._field_needles(entry, self.vocabulary_fields))
            if self.enable_vocabulary
            else []
        )

        # Issue 96d81f6: by default the vocabulary matcher skips tags on the
        # ignore lists (exact casefolded ignore_tags match or ignore_tags_regex
        # re.search, mirroring the fetch-side _should_fetch semantics). Filtered
        # at the SOURCE, before the policy filter and the vocabulary+rules
        # merge, so an ignored tag that a rule also fires still survives as a
        # normal RULE suggestion.
        if self.skip_ignored_tags and vocabulary:
            vocabulary = [
                s for s in vocabulary
                if not self._ignored(s.tag)
            ]

        # Issue 69ad357: "only-existing" no longer drops rule suggestions; it
        # filters them to the existing vocabulary. The policy means "never
        # suggest tags that aren't already in the wallabag instance", not
        # "ignore focus groups". enable_rules stays an explicit off-switch.
        rules = self._rule_suggestions(entry) if self.enable_rules else []
        if self.tag_policy == "only-existing" and rules:
            keys = _existing_tag_keys(self.existing_tags)
            rules = [s for s in rules if s.tag.strip().casefold() in keys]

        # "prefer-existing" and "all" share the same ordering: vocabulary
        # matches first, then rule-derived tags.
        ordered = vocabulary + rules

        # Dedup case-insensitively, keeping the FIRST occurrence so a tag
        # present both in vocabulary and rules keeps the vocabulary suggestion.
        seen: set[str] = set()
        unique: list[TagSuggestion] = []
        for suggestion in ordered:
            key = suggestion.tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(suggestion)

        return unique[: self.max_applied_tags]


class LLMTagger:
    """LLM tagger: the model ranks candidate tags and self-reports confidence.

    This is the ONE deliberate I/O exception in the tag engine: the model is
    reached through the injectable ``client`` (an ``LLMClientLike``) so tests
    can stub it and the pure filtering/parsing logic below stays deterministic.

    ``suggest()`` builds a system prompt that pins the model to the archive's
    existing tag vocabulary (the issue's rule is embedded verbatim), the
    focus-area tags of the groups the article matches as preferred topics,
    and a strict JSON-only output format. The model returns a JSON array of
    ``{"tag", "confidence"}``;
    suggestions below ``confidence_threshold`` are dropped inside ``suggest``
    (headless apply only ever sees above-threshold tags) and then gated by
    ``tag_policy``. An ``LLMError`` from the client propagates to the caller
    (pipelines decide how to degrade), never swallowed here.

    Focus areas are controlled by ``use_focus_groups`` (default True). When
    False, the "Focus areas" line is omitted from the system prompt entirely:
    focus groups become keyword-only. When True, the "Focus areas" line lists
    only the groups the article ACTUALLY matches within their configured
    fields (keywords OR regexes, reusing the keyword tagger's matching
    semantics); groups with ``fields == ()`` never match, so their tags never
    reach the prompt. Non-matching articles see "Focus areas: none.".

    The number of tags the model is ASKED to propose is ``max_proposals``
    (LLM-only; the "Return at most N tags." line). None follows
    ``max_applied_tags``, which is the hard cap on the APPLIED list for both
    taggers; an explicit ``max_proposals`` overrides the prompt bound without
    changing the applied cap.

    With ``verbose=True``, ``suggest()`` dumps the full system prompt, the
    full user prompt and the raw model response (untruncated) to stderr with
    a ``[debug]`` prefix, so an interactive ``wallatag manual --verbose`` run
    shows exactly what is sent to and returned by the model. Default False:
    no extra output is produced.

    By default (``skip_ignored_tags=True``) tags on the ``ignore_tags`` /
    ``ignore_tags_regex`` lists (exact casefolded match against
    ``ignore_tags``, or ``re.search`` against ``ignore_tags_regex``, mirroring
    the wallabag fetch-side semantics) are treated as unavailable vocabulary:
    they are excluded from the system prompt's "Existing tag vocabulary" line,
    and a model-proposed ignored tag is dropped inside ``suggest()`` even when
    it clears the confidence threshold (defense in depth against a model that
    ignores the prompt). Set ``skip_ignored_tags=False`` to present ignored
    tags as available vocabulary again. Focus-area tags are preferred topics,
    not available vocabulary, so the "Focus areas" line is never filtered this
    way.
    """

    def __init__(
        self,
        client: LLMClientLike,
        *,
        focus_groups: Mapping[str, FocusGroup],
        max_applied_tags: int,
        max_proposals: int | None = None,
        tag_policy: str,
        existing_tags: Iterable[str] = (),
        confidence_threshold: float = 0.7,
        use_focus_groups: bool = True,
        verbose: bool = False,
        ignore_tags: Iterable[str] = (),
        ignore_tags_regex: Iterable[str] = (),
        skip_ignored_tags: bool = True,
    ) -> None:
        if tag_policy not in VALID_TAG_POLICIES:
            raise ValueError(
                "invalid tag_policy %r; valid choices: %s"
                % (tag_policy, ", ".join(VALID_TAG_POLICIES))
            )
        if (
            not isinstance(max_applied_tags, int)
            or isinstance(max_applied_tags, bool)
            or max_applied_tags < 0
        ):
            raise ValueError("max_applied_tags must be a non-negative integer")
        # max_proposals is the LLM-only proposal bound (the "Return at most N
        # tags." prompt line). None follows max_applied_tags. It never limits
        # the APPLIED list: that stays max_applied_tags' job.
        if max_proposals is not None and (
            not isinstance(max_proposals, int)
            or isinstance(max_proposals, bool)
            or max_proposals < 0
        ):
            raise ValueError("max_proposals must be a non-negative integer")
        # bool is a float subclass: 0 < True <= 1 passes a bare range check,
        # so the isinstance(bool) guard mirrors max_applied_tags' trap.
        if (
            not isinstance(confidence_threshold, (int, float))
            or isinstance(confidence_threshold, bool)
            or not 0 < confidence_threshold <= 1
        ):
            raise ValueError(
                "confidence_threshold must be a number in the range (0, 1]"
            )
        if not isinstance(use_focus_groups, bool):
            raise ValueError("use_focus_groups must be a boolean")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        if not isinstance(skip_ignored_tags, bool):
            raise ValueError("skip_ignored_tags must be a boolean")
        self.client = client
        self.focus_groups = dict(focus_groups)
        # Pre-compile each group's regex patterns (shared with KeywordTagger)
        # so _system_prompt can reuse the keyword tagger's matching semantics.
        # config.py already validated compilability, so no practical behavior
        # change at load time; the eager validation mirrors KeywordTagger.
        self._group_regexes: dict[str, tuple[re.Pattern, ...]] = _compile_group_regexes(
            self.focus_groups
        )
        self.max_applied_tags = max_applied_tags
        self.max_proposals = max_proposals
        # The prompt bound: None follows max_applied_tags (backward-compatible
        # default), an explicit max_proposals overrides it for the prompt only.
        self._proposal_bound = (
            max_applied_tags if max_proposals is None else max_proposals
        )
        self.tag_policy = tag_policy
        self.existing_tags = list(existing_tags)
        self.confidence_threshold = float(confidence_threshold)
        self.use_focus_groups = use_focus_groups
        self.verbose = verbose
        # Ignore lists mirrored from the wallabag fetch side (wallabag.py
        # iter_untagged/untagged_entries/_should_fetch): exact matching is
        # casefolded (with surrounding whitespace stripped, consistent with
        # _existing_tag_keys); regexes search the RAW label with
        # re.IGNORECASE. config.py already validated compilability, so the
        # patterns are compiled directly. Same semantics as KeywordTagger's
        # ignore handling, so the two taggers agree on what is ignored.
        self._ignored_keys: frozenset[str] = frozenset(
            tag.strip().casefold()
            for tag in ignore_tags
            if isinstance(tag, str) and tag.strip()
        )
        self._ignored_patterns: tuple[re.Pattern, ...] = tuple(
            re.compile(p, re.IGNORECASE) for p in ignore_tags_regex
        )
        self.skip_ignored_tags = skip_ignored_tags

    def _ignored(self, tag: str) -> bool:
        """True if the tag is on the ignore lists (fetch-side semantics).

        Identical to KeywordTagger._ignored: exact matching is casefolded
        (and whitespace-stripped on both sides); regexes search the RAW label
        (``re.search``, compiled with ``re.IGNORECASE``) — mirroring
        wallabag.py's ``_should_fetch``.
        """
        key = tag.strip().casefold()
        if key in self._ignored_keys:
            return True
        return any(p.search(tag) for p in self._ignored_patterns)

    def _system_prompt(self, entry: dict) -> str:
        """Build the system prompt: role, vocabulary rule, focus, policy.

        Focus areas are per-entry: only groups the article
        actually matches within their configured fields are listed, reusing
        the keyword tagger's matching semantics.
        """
        existing_tags = [
            tag
            for tag in self.existing_tags
            if isinstance(tag, str)
            and tag.strip()
            and not (self.skip_ignored_tags and self._ignored(tag))
        ]
        lines = [
            "You are a tagging assistant for a personal read-it-later archive.",
            "Existing tag vocabulary: " + (", ".join(existing_tags) or "none") + ".",
        ]
        if self.tag_policy == "prefer-existing":
            # The vocabulary-preference rule is verbatim from the issue spec.
            # Only emitted under prefer-existing: under only-existing it would
            # contradict the "ONLY choose from the provided vocabulary" policy
            # line, and under "all" new tags are explicitly welcome.
            lines.append(
                "prefer the same tags that already exist; only add new ones if "
                "they really don't fit and are an important part of the text"
            )
        if self.use_focus_groups:
            # focus areas are per-entry — only groups the
            # article actually matches (within their configured fields,
            # keywords OR regexes) are listed, reusing the keyword tagger's
            # matching semantics. Groups with fields == () never match.
            focus_tags = [
                tag
                for name, group in self.focus_groups.items()
                if _group_fires(group, entry, self._group_regexes[name])
                for tag in group.tags
                if isinstance(tag, str) and tag.strip()
            ]
            lines.append("Focus areas: " + (", ".join(focus_tags) or "none") + ".")
        lines.append(f"Return at most {self._proposal_bound} tags.")
        if self.tag_policy == "only-existing":
            lines.append(
                "Tag policy: ONLY choose from the provided existing tag "
                "vocabulary; never invent new tags."
            )
        elif self.tag_policy == "prefer-existing":
            lines.append("Tag policy: prefer existing vocabulary tags.")
        else:  # "all"
            lines.append("Tag policy: new tags beyond the vocabulary are welcome.")
        lines.append(
            'Reply with ONLY a JSON array of objects with "tag" (string) and '
            '"confidence" (number between 0 and 1) fields. No prose, no '
            "markdown code fences."
        )
        return "\n".join(lines)

    def _user_prompt(self, entry: dict) -> str:
        """Build the user prompt: a plain-text digest of the article."""
        content = _clean_content(entry.get("content") or "")[:_MAX_CONTENT_CHARS]
        return "\n".join(
            [
                f"Title: {entry.get('title') or '(untitled)'}",
                f"URL: {entry.get('url') or '(none)'}",
                f"Domain: {entry.get('domain_name') or '(unknown)'}",
                f"Language: {entry.get('language') or 'unknown'}",
                f"Reading time: {entry.get('reading_time') or 0} minutes",
                f"Content: {content or '(no content)'}",
            ]
        )

    def _parse_response(self, body: str) -> list[dict]:
        """Parse the model's JSON array of {tag, confidence}, leniently.

        Malformed entries (missing/non-string tag, missing/non-numeric
        confidence) are SKIPPED rather than fatal, because a flaky model should
        cost us one tag, not the whole article. A body that is not a JSON
        array at all (or not JSON) raises LLMError: that is a protocol breach.
        Confidence values are clamped to [0, 1].
        """
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise LLMError(f"LLM returned invalid JSON: {body[:200]!r}") from exc
        if not isinstance(parsed, list):
            raise LLMError(f"LLM response is not a JSON array: {body[:200]!r}")
        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            tag = item.get("tag")
            confidence = item.get("confidence")
            if not isinstance(tag, str) or not tag.strip():
                continue
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                continue
            result.append(
                {
                    # Normalize: a padded model tag like "  python  " would
                    # otherwise be applied to wallabag verbatim.
                    "tag": tag.strip(),
                    "confidence": min(max(float(confidence), 0.0), 1.0),
                }
            )
        return result

    def _debug_dump(
        self,
        entry: dict,
        system_prompt: str,
        user_prompt: str,
        body: str,
    ) -> None:
        """Dump the prompts and the raw model response to stderr (verbose).

        Every line is ``[debug]``-prefixed so multi-line prompts stay
        readable, and nothing is truncated: the user sees exactly what was
        sent to and returned by the model. Only ever called when
        ``self.verbose`` is True.
        """
        header = f"[debug] LLM entry {entry.get('id')}"
        for label, content in (
            ("system prompt", system_prompt),
            ("user prompt", user_prompt),
            ("response", body),
        ):
            print(f"{header} {label}:", file=sys.stderr)
            for line in content.splitlines() or [""]:
                print(f"[debug]   {line}", file=sys.stderr)

    def suggest(self, entry: dict) -> list[TagSuggestion]:
        """Suggest tags for one entry; lets LLMError propagate to the caller."""
        system_prompt = self._system_prompt(entry)
        user_prompt = self._user_prompt(entry)
        body = self.client.complete(system_prompt, user_prompt)
        if self.verbose:
            # Dump BEFORE parsing so a malformed/error response is still
            # visible to an interactive verbose run.
            self._debug_dump(entry, system_prompt, user_prompt, body)
        suggestions = [
            TagSuggestion(
                tag=item["tag"],
                source="llm",
                confidence=item["confidence"],
            )
            for item in self._parse_response(body)
        ]

        # Headless apply is gated on confidence: drop below-threshold tags here
        # so the pipelines never see them.
        suggestions = [
            s for s in suggestions if s.confidence >= self.confidence_threshold
        ]
        # Issue db9f4c5: by default ignored tags are dropped here too, so a
        # model proposing an ignored tag despite the prompt can never apply it
        # (mirrors the vocabulary matcher's skip in KeywordTagger). Silently
        # dropped like the threshold filter, not an LLMError.
        if self.skip_ignored_tags and suggestions:
            suggestions = [s for s in suggestions if not self._ignored(s.tag)]
        if self.tag_policy == "only-existing":
            # Normalization shared with KeywordTagger: a tag counts as
            # existing iff its stripped+casefolded form is in the vocabulary
            # (_existing_tag_keys skips non-string/blank labels defensively).
            existing = _existing_tag_keys(self.existing_tags)
            suggestions = [
                s for s in suggestions if s.tag.strip().casefold() in existing
            ]

        # The model's array is NOT ranked (the system prompt never demands an
        # ordering), so rank the survivors by confidence DESCENDING before
        # truncation. Stable sort: equal-confidence candidates keep the model's
        # output order.
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        # Dedup case-insensitively keeping the FIRST occurrence. Ranking above
        # happens BEFORE this dedup, so the first occurrence of each casefolded
        # tag is automatically its highest-confidence one.
        seen: set[str] = set()
        unique: list[TagSuggestion] = []
        for suggestion in suggestions:
            key = suggestion.tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(suggestion)

        return unique[: self.max_applied_tags]
