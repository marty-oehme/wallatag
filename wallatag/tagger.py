"""Tag suggestion types.

Types only — no logic yet. The keyword-rule + vocabulary tagger logic lands in
git-bug issue 840ba68; the LLM tagger in issue 2ec50f7.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TagSuggestion:
    tag: str
    source: str
    confidence: float = 1.0


class Tagger(Protocol):
    def suggest(self, entry: dict) -> list[TagSuggestion]:
        ...
