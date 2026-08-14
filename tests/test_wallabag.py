"""Tests for wallatag.wallabag: the thin wallabag REST API client.

HTTP is mocked by injecting a FakeSession (a lightweight stub with ``get`` /
``post`` returning FakeResponse objects) into WallabagClient. Every call is
recorded for assertions.
"""

import time
import unittest

import requests

from wallatag.wallabag import WallabagClient, WallabagError

BASE_URL = "https://wallabag.example.com"
TOKEN_PAYLOAD = {
    "access_token": "tok123",
    "expires_in": 3600,
    "refresh_token": "refresh123",
    "token_type": "bearer",
    "scope": "",
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", bad_json=False):
        self.status_code = status_code
        self._payload = payload
        self._bad_json = bad_json
        if text:
            self.text = text
        elif bad_json:
            self.text = "<html>not json</html>"
        else:
            self.text = ""

    def json(self):
        if self._bad_json:
            raise ValueError("no JSON here")
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.closed = False

    def _record(self, method, url, kwargs):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": kwargs.get("params"),
                "data": kwargs.get("data"),
                "headers": kwargs.get("headers"),
                "timeout": kwargs.get("timeout"),
            }
        )

    def _next(self):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        self._record("GET", url, kwargs)
        return self._next()

    def post(self, url, **kwargs):
        self._record("POST", url, kwargs)
        return self._next()

    def close(self):
        self.closed = True


def entry(item_id, tags):
    return {
        "id": item_id,
        "title": f"title-{item_id}",
        "url": f"https://example.com/{item_id}",
        "content": f"<p>{item_id}</p>",
        "reading_time": 1,
        "domain_name": "example.com",
        "tags": tags,
        "is_archived": 0,
        "is_starred": 0,
        "language": "en",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }


def entries_payload(items, total=None, page=1, pages=1):
    return {
        "_embedded": {"items": items},
        "total": total if total is not None else len(items),
        "page": page,
        "pages": pages,
        "limit": 30,
    }


def make_client(session, username="alice", password="wonderland"):
    # Trailing slash in base_url must be normalized away.
    return WallabagClient(
        base_url=BASE_URL + "/",
        client_id="cid",
        client_secret="secret",
        username=username,
        password=password,
        session=session,
    )


class WallabagClientTestCase(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.client = make_client(self.session)

    def queue(self, *responses):
        self.session._responses.extend(responses)

    def token_calls(self):
        return [c for c in self.session.calls if c["url"].endswith("/oauth/v2/token")]

    def api_calls(self, path):
        return [c for c in self.session.calls if c["url"].endswith(path)]


class TokenFlowTest(WallabagClientTestCase):
    def test_no_request_on_construction(self):
        # Token fetching is lazy: constructing the client does no HTTP.
        self.assertEqual(self.session.calls, [])

    def test_token_fetched_lazily_and_attached(self):
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(200, entries_payload([entry(1, [])])),
        )
        self.client.get_entries()

        self.assertEqual(len(self.session.calls), 2)
        token_call = self.session.calls[0]
        self.assertEqual(token_call["method"], "POST")
        self.assertEqual(token_call["url"], f"{BASE_URL}/oauth/v2/token")
        # wallabag uses the OAuth2 password grant, not client_credentials.
        self.assertEqual(
            token_call["data"],
            {
                "grant_type": "password",
                "client_id": "cid",
                "client_secret": "secret",
                "username": "alice",
                "password": "wonderland",
            },
        )
        # The refresh_token from the password response is stored for later.
        self.assertEqual(self.client._refresh_token, "refresh123")
        api_call = self.session.calls[1]
        self.assertEqual(api_call["url"], f"{BASE_URL}/api/entries.json")
        self.assertEqual(api_call["headers"]["Authorization"], "Bearer tok123")

    def test_token_cached_within_expiry(self):
        page_payload = entries_payload([entry(1, [])])
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(200, page_payload),
            FakeResponse(200, page_payload),
        )
        self.client.get_entries()
        self.client.get_entries()
        self.assertEqual(len(self.token_calls()), 1)

    def test_expired_token_refreshed_with_stored_refresh_token(self):
        page_payload = entries_payload([entry(1, [])])
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, page_payload))
        self.client.get_entries()
        self.assertEqual(len(self.token_calls()), 1)

        # Simulate expiry by pushing the cached deadline into the past.
        self.client._token_expires_at = time.monotonic() - 10
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, page_payload))
        self.client.get_entries()

        calls = self.token_calls()
        self.assertEqual(len(calls), 2)
        # The next token call uses the refresh grant with the stored token.
        self.assertEqual(calls[1]["data"]["grant_type"], "refresh_token")
        self.assertEqual(calls[1]["data"]["refresh_token"], "refresh123")


class RetryTest(WallabagClientTestCase):
    def test_401_retries_via_fresh_password_fetch(self):
        # A mid-request 401 invalidates the cached token; the retry re-auths
        # with a fresh password grant (not a refresh) exactly once.
        page_payload = entries_payload([entry(1, [])])
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, page_payload))
        self.client.get_entries()  # cache a token first

        self.queue(
            FakeResponse(401, text="unauthorized"),
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(200, page_payload),
        )
        page = self.client.get_entries()

        self.assertEqual([e["id"] for e in page.items], [1])
        calls = self.token_calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["data"]["grant_type"], "password")
        self.assertEqual(calls[1]["data"]["grant_type"], "password")
        # 1 priming call + the 401 attempt and its retry.
        self.assertEqual(len(self.api_calls("/api/entries.json")), 3)

    def test_persistent_401_raises(self):
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(401, text="unauthorized"),
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(401, text="unauthorized"),
        )
        with self.assertRaises(WallabagError) as ctx:
            self.client.get_entries()
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(len(self.token_calls()), 2)


class RefreshTest(WallabagClientTestCase):
    def _prime_token(self):
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, entries_payload([entry(1, [])])))
        self.client.get_entries()
        self.client._token_expires_at = time.monotonic() - 10

    def test_refresh_failure_falls_back_to_password_fetch(self):
        self._prime_token()
        self.queue(
            FakeResponse(400, text="invalid_grant"),  # refresh is rejected
            FakeResponse(200, TOKEN_PAYLOAD),          # password-grant fallback
            FakeResponse(200, entries_payload([entry(1, [])])),
        )
        page = self.client.get_entries()
        self.assertEqual([e["id"] for e in page.items], [1])

        calls = self.token_calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [c["data"]["grant_type"] for c in calls],
            ["password", "refresh_token", "password"],
        )

    def test_refresh_and_password_fallback_both_fail_raises_password_error(self):
        # Double failure: refresh grant is rejected AND the password-grant
        # fallback is also rejected -> the password error surfaces (status
        # 400), it is not swallowed.
        self._prime_token()
        self.queue(
            FakeResponse(400, text="invalid_grant"),   # refresh is rejected
            FakeResponse(400, text="invalid_client"),  # fallback also fails
        )
        with self.assertRaises(WallabagError) as ctx:
            self.client.get_entries()
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("invalid_client", str(ctx.exception))

        calls = self.token_calls()
        self.assertEqual(len(calls), 3)  # prime + refresh + password fallback
        self.assertEqual(
            [c["data"]["grant_type"] for c in calls],
            ["password", "refresh_token", "password"],
        )

    def test_refresh_rotates_stored_refresh_token(self):
        self._prime_token()
        self.assertEqual(self.client._refresh_token, "refresh123")
        rotated = dict(TOKEN_PAYLOAD, access_token="tok456", refresh_token="refresh456")
        self.queue(FakeResponse(200, rotated), FakeResponse(200, entries_payload([entry(1, [])])))
        self.client.get_entries()
        self.assertEqual(self.client._refresh_token, "refresh456")

    def test_refresh_keeps_existing_refresh_token_when_absent(self):
        self._prime_token()
        no_refresh = {k: v for k, v in TOKEN_PAYLOAD.items() if k != "refresh_token"}
        self.queue(FakeResponse(200, no_refresh), FakeResponse(200, entries_payload([entry(1, [])])))
        self.client.get_entries()
        self.assertEqual(self.client._refresh_token, "refresh123")

    def test_no_refresh_token_available_falls_back_to_password_fetch(self):
        # Password response without a refresh_token: on expiry, refresh is
        # unavailable, so the client falls back to a fresh password fetch.
        no_refresh = {k: v for k, v in TOKEN_PAYLOAD.items() if k != "refresh_token"}
        self.queue(FakeResponse(200, no_refresh), FakeResponse(200, entries_payload([entry(1, [])])))
        self.client.get_entries()
        self.assertIsNone(self.client._refresh_token)
        self.client._token_expires_at = time.monotonic() - 10
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, entries_payload([entry(1, [])])))
        self.client.get_entries()
        calls = self.token_calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual([c["data"]["grant_type"] for c in calls], ["password", "password"])


class ConstructionValidationTest(unittest.TestCase):
    def test_empty_username_raises(self):
        with self.assertRaises(ValueError):
            WallabagClient(
                BASE_URL, "cid", "secret", username="", password="wonderland",
                session=FakeSession(),
            )

    def test_empty_password_raises(self):
        with self.assertRaises(ValueError):
            WallabagClient(
                BASE_URL, "cid", "secret", username="alice", password="",
                session=FakeSession(),
            )


class EntriesTest(WallabagClientTestCase):
    def test_get_entries_url_params_and_page(self):
        payload = entries_payload([entry(1, [])], total=2, page=2, pages=3)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.get_entries(page=2, per_page=10, sort="created")

        api_call = self.session.calls[1]
        self.assertEqual(api_call["url"], f"{BASE_URL}/api/entries.json")
        self.assertEqual(
            api_call["params"], {"page": 2, "perPage": 10, "sort": "created"}
        )
        self.assertEqual(page.items, payload["_embedded"]["items"])
        self.assertEqual(page.total, 2)
        self.assertEqual(page.page, 2)
        self.assertEqual(page.pages, 3)

    def test_untagged_entries_filters_client_side(self):
        items = [entry(1, []), entry(2, ["a"]), entry(3, [])]
        payload = entries_payload(items, total=3, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries()

        self.assertEqual([e["id"] for e in page.items], [1, 3])
        # Filtering happens client-side; the API-level counts are preserved.
        self.assertEqual(page.total, 3)
        self.assertEqual(page.pages, 1)

    def test_iter_untagged_pages_through_all(self):
        page1 = entries_payload(
            [entry(1, []), entry(2, ["x"])], total=3, page=1, pages=2
        )
        page2 = entries_payload(
            [entry(3, []), entry(4, ["y"])], total=3, page=2, pages=2
        )
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(200, page1),
            FakeResponse(200, page2),
        )

        entries = list(self.client.iter_untagged(per_page=2))

        self.assertEqual([e["id"] for e in entries], [1, 3])
        self.assertEqual(len(self.api_calls("/api/entries.json")), 2)

    def test_untagged_entries_ignored_tags_matrix(self):
        # The user A-E matrix: an article is fetched iff it has no tags OR
        # every one of its tags is in the ignore-any list.
        items = [
            entry(1, ["fix", "_frigo"]),    # A: all ignored -> fetched
            entry(2, ["fix"]),              # B: all ignored -> fetched
            entry(3, ["fix", "something"]),  # C: non-ignored present -> dropped
            entry(4, ["something"]),        # D: non-ignored present -> dropped
            entry(5, []),                   # E: untagged -> fetched, as before
        ]
        payload = entries_payload(items, total=5, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(ignored_tags=["fix", "_frigo"])

        self.assertEqual([e["id"] for e in page.items], [1, 2, 5])
        # Client-side filtering; the API-level counts are preserved.
        self.assertEqual(page.total, 5)
        self.assertEqual(page.pages, 1)

    def test_iter_untagged_default_ignored_tags_unchanged(self):
        # With the default empty ignore list, only fully untagged entries are
        # yielded (existing behavior unchanged).
        page = entries_payload(
            [entry(1, []), entry(2, ["a"]), entry(3, ["fix"])],
            total=3, pages=1,
        )
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, page))

        entries = list(self.client.iter_untagged(per_page=30))

        self.assertEqual([e["id"] for e in entries], [1])

    def test_untagged_entries_dict_shaped_tags_matched(self):
        # Wallabag 2.x may return tag objects; label is used, slug as fallback.
        items = [
            entry(1, [{"label": "fix", "slug": "fix"}]),   # label ignored -> fetched
            entry(2, [{"slug": "fix"}]),                   # no label, slug ignored -> fetched
            entry(3, [{"label": "something"}]),            # non-ignored label -> dropped
            entry(4, [{"foo": "bar"}]),                    # no label/slug -> untagged -> fetched
        ]
        payload = entries_payload(items, total=4, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(ignored_tags=["fix"])

        self.assertEqual([e["id"] for e in page.items], [1, 2, 4])
        self.assertEqual(page.total, 4)

    def test_untagged_entries_ignored_tags_case_insensitive(self):
        # Case-insensitive exact match: ignore ["Fix"] also covers tag "fix".
        items = [
            entry(1, ["fix"]),                # casefold matches "Fix" -> fetched
            entry(2, ["FIX", "something"]),   # still carries a non-ignored tag
        ]
        payload = entries_payload(items, total=2, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(ignored_tags=["Fix"])

        self.assertEqual([e["id"] for e in page.items], [1])
        self.assertEqual(page.total, 2)

    def test_untagged_entries_ignored_regex_matrix(self):
        # The ignore-any matrix with regex patterns: an article is fetched iff
        # it has no tags OR every tag matches at least one pattern.
        items = [
            entry(1, ["todo"]),          # matches "todo|fix" -> fetched
            entry(2, ["fix"]),           # matches -> fetched
            entry(3, ["todo", "other"]),  # non-matching tag present -> dropped
            entry(4, ["other"]),         # non-matching -> dropped
        ]
        payload = entries_payload(items, total=4, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(ignored_regex=("todo|fix",))

        self.assertEqual([e["id"] for e in page.items], [1, 2])
        # Client-side filtering; the API-level counts are preserved.
        self.assertEqual(page.total, 4)
        self.assertEqual(page.pages, 1)

    def test_untagged_entries_ignored_regex_case_insensitive(self):
        # re.IGNORECASE is applied by the caller: pattern "TODO" matches the
        # raw tag "todo".
        items = [entry(1, ["todo"]), entry(2, ["todo", "other"])]
        payload = entries_payload(items, total=2, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(ignored_regex=("TODO",))

        self.assertEqual([e["id"] for e in page.items], [1])
        self.assertEqual(page.total, 2)

    def test_untagged_entries_ignored_regex_inline_flag_override(self):
        # Inline (?-i:...) survives: the caller compiles with re.IGNORECASE
        # but the inline case-sensitive section wins, so "(?-i:todo)" does NOT
        # match the tag "TODO".
        items = [entry(1, ["TODO"]), entry(2, ["todo"])]
        payload = entries_payload(items, total=2, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(ignored_regex=("(?-i:todo)",))

        self.assertEqual([e["id"] for e in page.items], [2])
        self.assertEqual(page.total, 2)

    def test_untagged_entries_ignored_tags_and_regex_combined(self):
        # Literal ignore_tags and ignored_regex compose: a tag counts as
        # ignored iff it equals a literal entry OR matches a pattern.
        items = [
            entry(1, ["fix", "todo"]),   # fix literal + todo regex -> fetched
            entry(2, ["fix", "other"]),  # other matches neither -> dropped
        ]
        payload = entries_payload(items, total=2, pages=1)
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        page = self.client.untagged_entries(
            ignored_tags=["fix"], ignored_regex=("^todo$",)
        )

        self.assertEqual([e["id"] for e in page.items], [1])
        self.assertEqual(page.total, 2)

    def test_iter_untagged_with_ignored_regex_pages(self):
        page1 = entries_payload(
            [entry(1, ["todo"]), entry(2, ["other"])], total=3, page=1, pages=2
        )
        page2 = entries_payload(
            [entry(3, ["fix"]), entry(4, ["other"])], total=3, page=2, pages=2
        )
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(200, page1),
            FakeResponse(200, page2),
        )

        entries = list(
            self.client.iter_untagged(per_page=2, ignored_regex=("todo|fix",))
        )

        self.assertEqual([e["id"] for e in entries], [1, 3])
        self.assertEqual(len(self.api_calls("/api/entries.json")), 2)

    def test_per_page_too_large_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.client.get_entries(per_page=31)
        self.assertEqual(self.session.calls, [])

    def test_iter_untagged_terminates_when_pages_is_null(self):
        # Server returning pages=null must not cause a TypeError in the loop.
        payload = {
            "_embedded": {"items": [entry(1, []), entry(2, ["x"])]},
            "total": 2,
            "page": 1,
            "pages": None,
            "limit": 30,
        }
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, payload))

        entries = list(self.client.iter_untagged(per_page=30))

        self.assertEqual([e["id"] for e in entries], [1])
        self.assertEqual(len(self.api_calls("/api/entries.json")), 1)


class TagsTest(WallabagClientTestCase):
    def test_get_tags_returns_list(self):
        tags = [{"id": 1, "label": "a", "slug": "a", "nbEntries": 3}]
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, tags))

        self.assertEqual(self.client.get_tags(), tags)

    def test_add_tags_form_encoded(self):
        updated = entry(5, ["a", "b"])
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, updated))

        result = self.client.add_tags(5, [" a ", "", "b"])

        api_call = self.api_calls("/api/entries/5/tags.json")[0]
        self.assertEqual(api_call["method"], "POST")
        self.assertEqual(api_call["url"], f"{BASE_URL}/api/entries/5/tags.json")
        # Form-urlencoded body: a plain data dict, not json=.
        self.assertEqual(api_call["data"], {"tags": "a,b"})
        self.assertIsNone(api_call["params"])
        self.assertEqual(result, updated)

    def test_add_tags_empty_after_clean_raises(self):
        with self.assertRaises(ValueError):
            self.client.add_tags(5, ["   ", ""])
        self.assertEqual(self.session.calls, [])


class ErrorHandlingTest(WallabagClientTestCase):
    def test_non_2xx_raises_with_status_and_snippet(self):
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(404, text="entry not found"),
        )
        with self.assertRaises(WallabagError) as ctx:
            self.client.get_entries()
        self.assertEqual(ctx.exception.status, 404)
        self.assertIn("404", str(ctx.exception))
        self.assertIn("entry not found", str(ctx.exception))

    def test_token_fetch_failure_raises(self):
        self.queue(FakeResponse(401, text="invalid_client"))
        with self.assertRaises(WallabagError) as ctx:
            self.client.get_entries()
        self.assertEqual(ctx.exception.status, 401)

    def test_token_response_missing_token_raises(self):
        self.queue(FakeResponse(200, {"expires_in": 3600}))
        with self.assertRaises(WallabagError):
            self.client.get_entries()

    def test_non_json_success_body_raises(self):
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, bad_json=True))
        with self.assertRaises(WallabagError):
            self.client.get_entries()

    def test_connection_error_wrapped(self):
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), requests.ConnectionError("boom"))
        with self.assertRaises(WallabagError) as ctx:
            self.client.get_entries()
        self.assertIsNone(ctx.exception.status)
        self.assertIsInstance(ctx.exception.__cause__, requests.ConnectionError)


class LifecycleTest(WallabagClientTestCase):
    def test_close_closes_session(self):
        self.client.close()
        self.assertTrue(self.session.closed)


if __name__ == "__main__":
    unittest.main()
