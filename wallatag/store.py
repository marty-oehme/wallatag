"""Optional SQLite decision log (placeholder).

The decision log (seen-dedupe + tagging decisions) is implemented in git-bug
issue d3e386d. History-less mode means no database at all: untagged articles
may be re-presented and no learning data is accumulated.
"""


class Store:
    """Persistent decision log.

    ``path=None`` selects history-less mode: a valid no-op store that does not
    touch the filesystem.
    """

    def __init__(self, path: str | None) -> None:
        if path is None:
            self.path = None
            return
        raise NotImplementedError("implemented in issue d3e386d")
