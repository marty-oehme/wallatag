"""wallabag API client (placeholder).

The thin requests-based client (OAuth2 client-credentials flow, entries and
tags endpoints) is implemented in git-bug issue 007fc8f. This module stays a
stub until then; wallatag intentionally has no other HTTP dependency.
"""


class WallabagError(Exception):
    """Raised when a wallabag API request fails."""


class WallabagClient:
    """OAuth2 client-credentials client for the wallabag REST API."""

    def __init__(self) -> None:
        raise NotImplementedError("implemented in issue 007fc8f")
