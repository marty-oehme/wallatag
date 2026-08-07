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
TOKEN_PAYLOAD = {"access_token": "tok123", "expires_in": 3600}


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


def make_client(session):
    # Trailing slash in base_url must be normalized away.
    return WallabagClient(
        base_url=BASE_URL + "/",
        client_id="cid",
        client_secret="secret",
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
        self.assertEqual(
            token_call["data"],
            {
                "grant_type": "client_credentials",
                "client_id": "cid",
                "client_secret": "secret",
            },
        )
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

    def test_expired_token_refetched(self):
        page_payload = entries_payload([entry(1, [])])
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, page_payload))
        self.client.get_entries()
        self.assertEqual(len(self.token_calls()), 1)

        # Simulate expiry by pushing the cached deadline into the past.
        self.client._token_expires_at = time.monotonic() - 10
        self.queue(FakeResponse(200, TOKEN_PAYLOAD), FakeResponse(200, page_payload))
        self.client.get_entries()

        self.assertEqual(len(self.token_calls()), 2)


class RetryTest(WallabagClientTestCase):
    def test_401_retries_with_refreshed_token(self):
        self.queue(
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(401, text="unauthorized"),
            FakeResponse(200, TOKEN_PAYLOAD),
            FakeResponse(200, entries_payload([entry(1, [])])),
        )
        page = self.client.get_entries()

        self.assertEqual([e["id"] for e in page.items], [1])
        self.assertEqual(len(self.token_calls()), 2)
        self.assertEqual(len(self.api_calls("/api/entries.json")), 2)

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

    def test_per_page_too_large_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.client.get_entries(per_page=31)
        self.assertEqual(self.session.calls, [])


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
