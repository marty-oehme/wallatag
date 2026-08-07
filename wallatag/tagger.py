"""Tag suggestion engine.

Public types: ``TagSuggestion`` and the ``Tagger`` protocol. The MVP
implementation is ``KeywordTagger`` (keyword-rule groups + existing-tag
vocabulary). The LLM tagger (git-bug issue 2ec50f7) will slot in behind the
same protocol later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from wallatag.config import VALID_TAG_POLICIES, FocusGroup

_VOCABULARY_CONFIDENCE = 1.0
_RULE_CONFIDENCE = 0.7


@dataclass(frozen=True)
class TagSuggestion:
    tag: str
    source: str
    confidence: float = 1.0


class Tagger(Protocol):
    def suggest(self, entry: dict) -> list[TagSuggestion]:
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
