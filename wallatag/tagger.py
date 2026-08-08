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
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from wallatag.config import VALID_TAG_POLICIES, FocusGroup
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
    """MVP tagger: keyword-rule groups plus existing-tag vocabulary.

    Pure logic: no I/O and no dependency on the wallabag client. Matching is a
    case-insensitive raw substring check, evaluated per field: a keyword or an
    existing tag label must be a substring of at least one of the entry's
    title, url, domain_name and content fields individually: a multi-word
    needle never matches across field boundaries. The content is HTML; raw
    substring matching against it is acceptable for the MVP.

    tag_policy semantics in the MVP: both "prefer-existing" and "all" produce
    the same ordering: vocabulary suggestions first, then rule suggestions,
    with case-insensitive first-wins dedup; the policies differ only in that
    "only-existing" drops rule suggestions entirely.
    """

    _FIELDS = ("title", "url", "domain_name", "content")

    def __init__(
        self,
        focus_groups: Mapping[str, FocusGroup],
        *,
        max_suggestions: int,
        tag_policy: str,
        existing_tags: Iterable[str] = (),
    ) -> None:
        if tag_policy not in VALID_TAG_POLICIES:
            raise ValueError(
                "invalid tag_policy %r; valid choices: %s"
                % (tag_policy, ", ".join(VALID_TAG_POLICIES))
            )
        if (
            not isinstance(max_suggestions, int)
            or isinstance(max_suggestions, bool)
            or max_suggestions < 0
        ):
            raise ValueError("max_suggestions must be a non-negative integer")
        # dict preserves config insertion order -> deterministic iteration.
        self.focus_groups = dict(focus_groups)
        self.max_suggestions = max_suggestions
        self.tag_policy = tag_policy
        # Labels are normalized for matching but suggested in their original
        # casing (wallabag labels as provided).
        self.existing_tags = list(existing_tags)

    def _field_needles(self, entry: dict) -> tuple[str, ...]:
        """Casefolded per-field text to match against, fields kept separate."""
        fields = []
        for key in self._FIELDS:
            value = entry.get(key)
            if isinstance(value, str) and value:
                fields.append(value.casefold())
        return tuple(fields)

    @staticmethod
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

    def _rule_suggestions(self, fields: tuple[str, ...]) -> list[TagSuggestion]:
        suggestions = []
        for group in self.focus_groups.values():
            if any(self._matches(kw, fields) for kw in group.keywords):
                for tag in group.tags:
                    if isinstance(tag, str) and tag.strip():
                        suggestions.append(
                            TagSuggestion(
                                tag=tag, source="rules", confidence=_RULE_CONFIDENCE
                            )
                        )
        return suggestions

    def suggest(self, entry: dict) -> list[TagSuggestion]:
        fields = self._field_needles(entry)

        vocabulary = self._vocabulary_suggestions(fields)

        # "only-existing" drops rule-derived suggestions entirely.
        rules = (
            self._rule_suggestions(fields)
            if self.tag_policy != "only-existing"
            else []
        )

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

        return unique[: self.max_suggestions]


class LLMTagger:
    """LLM tagger: the model ranks candidate tags and self-reports confidence.

    This is the ONE deliberate I/O exception in the tag engine: the model is
    reached through the injectable ``client`` (an ``LLMClientLike``) so tests
    can stub it and the pure filtering/parsing logic below stays deterministic.

    ``suggest()`` builds a system prompt that pins the model to the archive's
    existing tag vocabulary (the issue's rule is embedded verbatim), the union
    of all focus-group tags as preferred topics, and a strict JSON-only output
    format. The model returns a JSON array of ``{"tag", "confidence"}``;
    suggestions below ``confidence_threshold`` are dropped inside ``suggest``
    (headless apply only ever sees above-threshold tags) and then gated by
    ``tag_policy``. An ``LLMError`` from the client propagates to the caller
    (pipelines decide how to degrade), never swallowed here.
    """

    def __init__(
        self,
        client: LLMClientLike,
        *,
        focus_groups: Mapping[str, FocusGroup],
        max_suggestions: int,
        tag_policy: str,
        existing_tags: Iterable[str] = (),
        confidence_threshold: float = 0.7,
    ) -> None:
        if tag_policy not in VALID_TAG_POLICIES:
            raise ValueError(
                "invalid tag_policy %r; valid choices: %s"
                % (tag_policy, ", ".join(VALID_TAG_POLICIES))
            )
        if (
            not isinstance(max_suggestions, int)
            or isinstance(max_suggestions, bool)
            or max_suggestions < 0
        ):
            raise ValueError("max_suggestions must be a non-negative integer")
        # bool is a float subclass: 0 < True <= 1 passes a bare range check,
        # so the isinstance(bool) guard mirrors max_suggestions' trap.
        if (
            not isinstance(confidence_threshold, (int, float))
            or isinstance(confidence_threshold, bool)
            or not 0 < confidence_threshold <= 1
        ):
            raise ValueError(
                "confidence_threshold must be a number in the range (0, 1]"
            )
        self.client = client
        self.focus_groups = dict(focus_groups)
        self.max_suggestions = max_suggestions
        self.tag_policy = tag_policy
        self.existing_tags = list(existing_tags)
        self.confidence_threshold = float(confidence_threshold)

    def _system_prompt(self) -> str:
        """Build the system prompt: role, vocabulary rule, focus, policy."""
        focus_tags = [
            tag
            for group in self.focus_groups.values()
            for tag in group.tags
            if isinstance(tag, str) and tag.strip()
        ]
        existing_tags = [
            tag
            for tag in self.existing_tags
            if isinstance(tag, str) and tag.strip()
        ]
        lines = [
            "You are a tagging assistant for a personal read-it-later archive.",
            "Existing tag vocabulary: " + (", ".join(existing_tags) or "none") + ".",
            # The vocabulary-preference rule is verbatim from the issue spec.
            "prefer the same tags that already exist; only add new ones if they "
            "really don't fit and are an important part of the text",
            "Focus areas: " + (", ".join(focus_tags) or "none") + ".",
            f"Return at most {self.max_suggestions} tags.",
        ]
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

    def suggest(self, entry: dict) -> list[TagSuggestion]:
        """Suggest tags for one entry; lets LLMError propagate to the caller."""
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(entry)
        body = self.client.complete(system_prompt, user_prompt)
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
        if self.tag_policy == "only-existing":
            # Non-string vocabulary entries are filtered defensively: a
            # non-string element would crash the casefold() below (the system
            # prompt builder applies the same guard).
            existing = {
                tag.casefold() for tag in self.existing_tags if isinstance(tag, str)
            }
            suggestions = [
                s for s in suggestions if s.tag.casefold() in existing
            ]

        # Dedup case-insensitively keeping the FIRST occurrence and the model's
        # output order.
        seen: set[str] = set()
        unique: list[TagSuggestion] = []
        for suggestion in suggestions:
            key = suggestion.tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(suggestion)

        return unique[: self.max_suggestions]
