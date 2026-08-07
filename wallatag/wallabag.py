"""Thin requests-based client for the wallabag v2 REST API.

Targets wallabag 2.x (endpoints verified against a 2.6.14 instance). Implements
the OAuth2 client-credentials token flow and the entries/tags endpoints used by
wallatag. There is no maintained external SDK; this client depends only on
``requests``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

TOKEN_ENDPOINT = "/oauth/v2/token"
ENTRIES_ENDPOINT = "/api/entries.json"
TAGS_ENDPOINT = "/api/tags.json"
ENTRY_TAGS_ENDPOINT = "/api/entries/{entry_id}/tags.json"

_MAX_PER_PAGE = 30
# Refresh the token this many seconds before the real expiry, to avoid races.
_TOKEN_SAFETY_MARGIN = 30.0


class WallabagError(Exception):
    """Raised when a wallabag API request fails.

    ``status`` holds the HTTP status code for HTTP-level failures, or ``None``
    for transport-level failures (connection error, timeout, ...).
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class EntryPage:
    items: list
    total: int
    page: int
    pages: int


class WallabagClient:
    """OAuth2 client-credentials client for the wallabag REST API.

    ``session`` is normally created internally; injecting one is how tests mock
    HTTP. ``timeout`` is applied to every request.
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0  # deadline on time.monotonic()

    # -- token handling --------------------------------------------------

    def _ensure_token(self) -> str:
        if self._access_token is None or time.monotonic() >= self._token_expires_at:
            self._fetch_token()
        assert self._access_token is not None
        return self._access_token

    def _fetch_token(self) -> None:
        url = f"{self.base_url}{TOKEN_ENDPOINT}"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        try:
            resp = self.session.post(url, data=data, timeout=self.timeout)
        except requests.RequestException as exc:
            raise WallabagError(f"token request failed: {exc}") from exc
        if not 200 <= resp.status_code < 300:
            raise WallabagError(
                self._http_error("token request", resp), status=resp.status_code
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise WallabagError(
                f"token response is not JSON: {resp.text[:200]!r}",
                status=resp.status_code,
            ) from exc
        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        if not access_token:
            raise WallabagError(
                f"token response missing access_token: {resp.text[:200]!r}",
                status=resp.status_code,
            )
        try:
            expires_in = float(payload.get("expires_in", 3600))
        except (TypeError, ValueError):
            expires_in = 3600.0
        self._access_token = access_token
        self._token_expires_at = time.monotonic() + max(
            expires_in - _TOKEN_SAFETY_MARGIN, 0.0
        )

    # -- low-level request -----------------------------------------------

    def _request(
        self, method: str, path: str, *, retried: bool = False, **kwargs: Any
    ) -> Any:
        token = self._ensure_token()
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {token}"
        url = f"{self.base_url}{path}"
        try:
            if method == "GET":
                resp = self.session.get(
                    url, headers=headers, timeout=self.timeout, **kwargs
                )
            elif method == "POST":
                resp = self.session.post(
                    url, headers=headers, timeout=self.timeout, **kwargs
                )
            else:
                raise ValueError(f"unsupported HTTP method: {method!r}")
        except requests.RequestException as exc:
            raise WallabagError(f"{method} {path} failed: {exc}") from exc

        if resp.status_code == 401 and not retried:
            # Token was rejected: force a refresh and retry exactly once.
            self._access_token = None
            self._token_expires_at = 0.0
            return self._request(method, path, retried=True, **kwargs)

        if not 200 <= resp.status_code < 300:
            raise WallabagError(
                self._http_error(f"{method} {path}", resp), status=resp.status_code
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise WallabagError(
                f"{method} {path} returned non-JSON body: {resp.text[:200]!r}",
                status=resp.status_code,
            ) from exc

    @staticmethod
    def _http_error(context: str, resp: Any) -> str:
        snippet = (resp.text or "").strip().replace("\n", " ")[:200]
        if snippet:
            return f"{context} failed with status {resp.status_code}: {snippet}"
        return f"{context} failed with status {resp.status_code}"

    # -- entries ---------------------------------------------------------

    def get_entries(self, page: int = 1, per_page: int = 30, **filters: Any) -> EntryPage:
        if page < 1:
            raise ValueError("page must be >= 1")
        if not 1 <= per_page <= _MAX_PER_PAGE:
            raise ValueError(f"per_page must be between 1 and {_MAX_PER_PAGE}")
        params = {"page": page, "perPage": per_page}
        params.update(filters)
        payload = self._request("GET", ENTRIES_ENDPOINT, params=params)
        if not isinstance(payload, dict):
            raise WallabagError("entries response is not a JSON object")
        embedded = payload.get("_embedded", {}) or {}
        return EntryPage(
            items=embedded.get("items", []),
            total=payload.get("total", 0),
            page=payload.get("page", page),
            pages=payload.get("pages", 0),
        )

    def untagged_entries(self, page: int = 1, per_page: int = 30) -> EntryPage:
        """Like get_entries, but only items whose ``tags`` is empty.

        The API has no server-side "no tags" filter, so this filters
        client-side on ``tags == []`` while preserving total/page/pages.
        """
        full = self.get_entries(page=page, per_page=per_page)
        items = [item for item in full.items if not item.get("tags")]
        return EntryPage(
            items=items, total=full.total, page=full.page, pages=full.pages
        )

    def iter_untagged(self, per_page: int = 30) -> Iterator[dict]:
        """Yield untagged entries across all pages, in order."""
        page_num = 1
        while True:
            page = self.untagged_entries(page=page_num, per_page=per_page)
            yield from page.items
            if page.pages == 0 or page_num >= page.pages:
                return
            page_num += 1

    # -- tags ------------------------------------------------------------

    def get_tags(self) -> list[dict]:
        payload = self._request("GET", TAGS_ENDPOINT)
        if not isinstance(payload, list):
            raise WallabagError("tags response is not a JSON array")
        return payload

    def add_tags(self, entry_id: int, tags: list[str]) -> dict:
        cleaned = [tag.strip() for tag in tags]
        cleaned = [tag for tag in cleaned if tag]
        if not cleaned:
            raise ValueError("at least one non-empty tag is required")
        path = ENTRY_TAGS_ENDPOINT.format(entry_id=entry_id)
        # Form-urlencoded on purpose: a dict body is encoded as form data by
        # requests (do NOT use json=); wallabag rejects JSON tag bodies.
        payload = self._request("POST", path, data={"tags": ",".join(cleaned)})
        if not isinstance(payload, dict):
            raise WallabagError("add_tags response is not a JSON object")
        return payload

    def close(self) -> None:
        self.session.close()
