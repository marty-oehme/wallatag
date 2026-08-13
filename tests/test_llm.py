"""Tests for wallatag.llm: LLMClient URL building, payload shape and errors.

HTTP is faked with a small FakeSession that records ``post(url, json,
headers, timeout)`` calls and returns a canned FakeResponse (or raises a
canned exception). A ``responses`` sequence lets the retry tests script a run
of transient failures followed by success.
"""

import json
import unittest
from unittest.mock import patch

import requests

from wallatag.llm import LLMClient, LLMError

SYSTEM = "system prompt"
USER = "user prompt"


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else (
            json.dumps(body) if body is not None else ""
        )

    def json(self):
        if self._body is None:
            raise ValueError("response body is not JSON")
        return self._body


class FakeSession:
    def __init__(self, response=None, error=None, responses=None):
        self.response = response
        self.error = error
        self.responses = list(responses) if responses is not None else None
        self.posts = []
        self.closed = False

    def post(self, url, json=None, timeout=None, headers=None, **kwargs):
        self.posts.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if self.responses is not None:
            if not self.responses:
                raise AssertionError(
                    "FakeSession.post called more times than responses provided"
                )
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


def client_for(response=None, error=None, responses=None, **kwargs):
    session = FakeSession(response=response, error=error, responses=responses)
    client = LLMClient(
        kwargs.pop("provider", "ollama"),
        kwargs.pop("base_url", "http://localhost:11434"),
        kwargs.pop("model", "qwen2.5:3b"),
        session=session,
        **kwargs,
    )
    return client, session


def ok_response(content="[{\"tag\": \"python\"}]"):
    return FakeResponse(
        status_code=200,
        body={"choices": [{"message": {"content": content}}]},
    )


class UrlTest(unittest.TestCase):
    def test_ollama_url_has_v1_prefix(self):
        client, session = client_for(ok_response())
        client.complete(SYSTEM, USER)
        self.assertEqual(
            session.posts[0]["url"], "http://localhost:11434/v1/chat/completions"
        )

    def test_openai_compatible_url_uses_base_url_as_is(self):
        client, session = client_for(
            ok_response(),
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
        )
        client.complete(SYSTEM, USER)
        self.assertEqual(
            session.posts[0]["url"], "https://api.example.com/v1/chat/completions"
        )

    def test_trailing_slash_stripped(self):
        client, session = client_for(ok_response(), base_url="http://localhost:11434/")
        client.complete(SYSTEM, USER)
        self.assertEqual(
            session.posts[0]["url"], "http://localhost:11434/v1/chat/completions"
        )

    def test_ollama_base_url_with_v1_prefix_no_double_prefix(self):
        # Ollama docs commonly give http://host:11434/v1 as the
        # OpenAI-compatible base_url; the client must not double the /v1.
        client, session = client_for(ok_response(), base_url="http://host:11434/v1")
        client.complete(SYSTEM, USER)
        self.assertEqual(
            session.posts[0]["url"], "http://host:11434/v1/chat/completions"
        )

    def test_ollama_base_url_with_v1_prefix_and_trailing_slash(self):
        # The /v1/ form survives __init__ rstrip as /v1 and is then de-duped.
        client, session = client_for(ok_response(), base_url="http://host:11434/v1/")
        client.complete(SYSTEM, USER)
        self.assertEqual(
            session.posts[0]["url"], "http://host:11434/v1/chat/completions"
        )

    def test_ollama_bare_host_still_gets_v1_prefix(self):
        # Bare-host form must keep its documented behavior.
        client, session = client_for(ok_response(), base_url="http://host:11434")
        client.complete(SYSTEM, USER)
        self.assertEqual(
            session.posts[0]["url"], "http://host:11434/v1/chat/completions"
        )


class PayloadTest(unittest.TestCase):
    def test_body_shape(self):
        client, session = client_for(ok_response())
        client.complete(SYSTEM, USER)

        payload = session.posts[0]["json"]
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["stream"], False)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
        )
        self.assertEqual(session.posts[0]["timeout"], 30.0)

    def test_returns_choices_message_content(self):
        client, _ = client_for(ok_response(content="a tag"))
        self.assertEqual(client.complete(SYSTEM, USER), "a tag")


class AuthHeaderTest(unittest.TestCase):
    """api_key controls the Authorization header (and nothing else)."""

    def test_api_key_sends_bearer_header(self):
        client, session = client_for(
            ok_response(),
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            api_key="sk-secret-key-123",
        )
        client.complete(SYSTEM, USER)
        headers = session.posts[0]["headers"]
        self.assertIsNotNone(headers)
        self.assertEqual(headers["Authorization"], "Bearer sk-secret-key-123")

    def test_no_api_key_sends_no_authorization_header(self):
        client, session = client_for(ok_response())
        client.complete(SYSTEM, USER)
        headers = session.posts[0]["headers"]
        # Production always sends a headers dict (possibly empty); there is
        # never a None, so assert unconditionally on the dict.
        self.assertNotIn("Authorization", headers)

    def test_empty_api_key_sends_no_authorization_header(self):
        client, session = client_for(ok_response(), api_key="")
        client.complete(SYSTEM, USER)
        headers = session.posts[0]["headers"]
        self.assertNotIn("Authorization", headers)

    def test_api_key_stored_on_client(self):
        client, _ = client_for(ok_response(), api_key="sk-abc")
        self.assertEqual(client.api_key, "sk-abc")

    def test_default_api_key_is_empty(self):
        client, _ = client_for(ok_response())
        self.assertEqual(client.api_key, "")

    def test_401_error_does_not_leak_api_key(self):
        # A hostile/misconfigured gateway echoes the bearer key back in the
        # error body; it must be scrubbed before reaching the LLMError text
        # (and from there logger.error / print output downstream).
        key = "sk-super-secret-value"
        client, _ = client_for(
            FakeResponse(
                status_code=401,
                body={"error": f"invalid api key {key} for model x"},
            ),
            api_key=key,
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 401)
        self.assertNotIn(key, str(ctx.exception))
        self.assertIn("***", str(ctx.exception))

    def test_connection_error_does_not_leak_api_key(self):
        client, _ = client_for(
            None,
            error=requests.ConnectionError("boom"),
            api_key="sk-secret-2",
            backoff_base=0.01,  # transient: retried; keep the sleep negligible
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertNotIn("sk-secret-2", str(ctx.exception))


class ErrorTest(unittest.TestCase):
    def test_http_error_raises_llm_error_with_status(self):
        # Keyless client (default api_key=""): the error body must be
        # preserved intact and never corrupted by a redaction pass (a bogus
        # unconditional replace of an empty key would corrupt every message).
        # 500 is transient, so this also exercises the exhausted-retry path;
        # the final LLMError still carries the status and the body snippet.
        client, _ = client_for(
            FakeResponse(status_code=500, body={"error": "boom"}),
            backoff_base=0.01,  # transient: retried; keep the sleep negligible
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("500", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))
        self.assertNotIn("***", str(ctx.exception))

    def test_transport_error_raises_llm_error(self):
        client, _ = client_for(
            None,
            error=requests.ConnectionError("connection refused"),
            backoff_base=0.01,  # transient: retried; keep the sleep negligible
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertIsNone(ctx.exception.status)

    def test_non_json_body_raises_llm_error(self):
        client, _ = client_for(FakeResponse(status_code=200, body=None, text="<html>"))
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 200)
        self.assertIn("not JSON", str(ctx.exception))

    def test_missing_content_raises_llm_error(self):
        client, _ = client_for(
            FakeResponse(status_code=200, body={"choices": [{"message": {}}]})
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 200)
        self.assertIn("content", str(ctx.exception))

    def test_empty_choices_raises_llm_error(self):
        client, _ = client_for(
            FakeResponse(status_code=200, body={"choices": []})
        )
        with self.assertRaises(LLMError):
            client.complete(SYSTEM, USER)

    def test_non_string_content_raises_llm_error(self):
        client, _ = client_for(
            FakeResponse(
                status_code=200, body={"choices": [{"message": {"content": 42}}]}
            )
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertIn("not a string", str(ctx.exception))


class RetryTest(unittest.TestCase):
    """Bounded retry with exponential backoff + jitter (issue 2b3f9e8).

    Only TRANSIENT failures (transport errors, _TRANSIENT_STATUSES HTTP
    codes) are retried; config/protocol errors never are. Retry tests use a
    tiny backoff_base so the real sleeps stay negligible.
    """

    def test_transport_error_twice_then_success(self):
        # (a) two transport failures, then a 200: complete() succeeds and the
        # session was called exactly 3 times (1 initial + 2 retries).
        client, session = client_for(
            None,
            responses=[
                requests.ConnectionError("boom"),
                requests.ConnectionError("boom"),
                ok_response(content="a tag"),
            ],
            backoff_base=0.01,
        )
        self.assertEqual(client.complete(SYSTEM, USER), "a tag")
        self.assertEqual(len(session.posts), 3)

    def test_http_503_twice_then_success(self):
        # (b) transient 503s are retried; a later 200 succeeds.
        client, session = client_for(
            None,
            responses=[
                FakeResponse(status_code=503, body={"error": "down"}),
                FakeResponse(status_code=503, body={"error": "down"}),
                ok_response(content="a tag"),
            ],
            backoff_base=0.01,
        )
        self.assertEqual(client.complete(SYSTEM, USER), "a tag")
        self.assertEqual(len(session.posts), 3)

    def test_http_429_is_retried(self):
        # (c) 429 (rate limit) is transient and gets retried.
        client, session = client_for(
            None,
            responses=[
                FakeResponse(status_code=429, body={"error": "slow down"}),
                ok_response(content="a tag"),
            ],
            backoff_base=0.01,
        )
        self.assertEqual(client.complete(SYSTEM, USER), "a tag")
        self.assertEqual(len(session.posts), 2)

    def test_http_408_502_504_are_retried(self):
        # (i) 408 (request timeout) and the 5xx gateway hiccups (502, 504)
        # are transient: with retries=1 each is retried once, then a 200 wins
        # (exactly 1 initial attempt + 1 retry = 2 posts).
        for status in (408, 502, 504):
            with self.subTest(status=status):
                client, session = client_for(
                    None,
                    responses=[
                        FakeResponse(status_code=status, body={"error": "transient"}),
                        ok_response(content="a tag"),
                    ],
                    retries=1,
                    backoff_base=0.01,
                )
                self.assertEqual(client.complete(SYSTEM, USER), "a tag")
                self.assertEqual(len(session.posts), 2)

    def test_http_400_is_not_retried(self):
        # (d) a 400 is a protocol/config error: exactly one attempt, no sleep.
        client, session = client_for(
            None,
            responses=[FakeResponse(status_code=400, body={"error": "bad request"})],
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(session.posts), 1)

    def test_http_401_403_404_are_not_retried(self):
        # (j) 401/403/404 are auth/config errors, not transient: exactly one
        # attempt even with retries=2, and the final LLMError preserves the
        # HTTP status.
        for status in (401, 403, 404):
            with self.subTest(status=status):
                client, session = client_for(
                    None,
                    responses=[FakeResponse(status_code=status, body={"error": "no"})],
                    retries=2,
                )
                with self.assertRaises(LLMError) as ctx:
                    client.complete(SYSTEM, USER)
                self.assertEqual(ctx.exception.status, status)
                self.assertEqual(len(session.posts), 1)

    def test_non_json_body_is_not_retried(self):
        # (e) response parsing happens after the retry loop and is never
        # retried, even though the status is 200.
        client, session = client_for(
            None,
            responses=[FakeResponse(status_code=200, body=None, text="<html>")],
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertIn("not JSON", str(ctx.exception))
        self.assertEqual(len(session.posts), 1)

    def test_transport_retries_exhausted_mentions_attempts(self):
        # (f) retries=2 with 3 transport failures: the final LLMError notes the
        # attempts and keeps a None status.
        client, session = client_for(
            None,
            responses=[
                requests.ConnectionError("boom"),
                requests.ConnectionError("boom"),
                requests.ConnectionError("boom"),
            ],
            backoff_base=0.01,
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertIsNone(ctx.exception.status)
        self.assertIn("after 3 attempts", str(ctx.exception))
        self.assertEqual(len(session.posts), 3)

    def test_http_retries_exhausted_keeps_status(self):
        # Transient HTTP failures exhausted: the final LLMError still carries
        # the HTTP status attribute (probe: status preserved for the HTTP case).
        client, session = client_for(
            None,
            responses=[
                FakeResponse(status_code=500, body={"error": "boom"}),
                FakeResponse(status_code=500, body={"error": "boom"}),
                FakeResponse(status_code=500, body={"error": "boom"}),
            ],
            backoff_base=0.01,
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("500", str(ctx.exception))
        self.assertEqual(len(session.posts), 3)

    def test_sleeps_between_attempts_with_backoff(self):
        # (g) time.sleep is called between attempts with the backoff delays.
        client, session = client_for(
            None,
            responses=[
                requests.ConnectionError("boom"),
                requests.ConnectionError("boom"),
                ok_response(content="a tag"),
            ],
            backoff_base=0.01,
        )
        sleeps = []
        with patch("wallatag.llm.time.sleep", side_effect=sleeps.append):
            content = client.complete(SYSTEM, USER)
        self.assertEqual(content, "a tag")
        self.assertEqual(len(session.posts), 3)
        self.assertEqual(len(sleeps), 2)
        # attempt 0: 0.01 * 2**0 * [0.5, 1.5]; attempt 1: 0.01 * 2**1 * [0.5, 1.5].
        self.assertTrue(0.005 <= sleeps[0] <= 0.015, sleeps[0])
        self.assertTrue(0.01 <= sleeps[1] <= 0.03, sleeps[1])

    def test_backoff_delay_formula(self):
        # Exponential + jitter: with the jitter pinned to 1.0 the delays are
        # exactly backoff_base * 2**attempt.
        client, _ = client_for(ok_response(), backoff_base=0.5)
        with patch("wallatag.llm.random.uniform", return_value=1.0):
            self.assertEqual(client._backoff_delay(0), 0.5)
            self.assertEqual(client._backoff_delay(1), 1.0)
            self.assertEqual(client._backoff_delay(2), 2.0)

    def test_retries_zero_single_attempt(self):
        # (h) retries=0 means one attempt, no retries and no sleeping.
        client, session = client_for(
            None,
            responses=[requests.ConnectionError("boom")],
            retries=0,
        )
        with patch("wallatag.llm.time.sleep") as sleep_mock:
            with self.assertRaises(LLMError):
                client.complete(SYSTEM, USER)
        self.assertEqual(len(session.posts), 1)
        sleep_mock.assert_not_called()

    def test_retries_zero_single_attempt_success(self):
        client, session = client_for(
            None, responses=[ok_response(content="a tag")], retries=0
        )
        self.assertEqual(client.complete(SYSTEM, USER), "a tag")
        self.assertEqual(len(session.posts), 1)


class RetryValidationTest(unittest.TestCase):
    """Constructor validation for the retry knobs (issue 2b3f9e8)."""

    def test_negative_retries_rejected(self):
        with self.assertRaises(ValueError):
            LLMClient("ollama", "http://localhost:11434", "qwen2.5:3b", retries=-1)

    def test_string_retries_rejected(self):
        with self.assertRaises(ValueError):
            LLMClient("ollama", "http://localhost:11434", "qwen2.5:3b", retries="2")

    def test_bool_retries_rejected(self):
        with self.assertRaises(ValueError):
            LLMClient("ollama", "http://localhost:11434", "qwen2.5:3b", retries=True)

    def test_zero_backoff_rejected(self):
        with self.assertRaises(ValueError):
            LLMClient("ollama", "http://localhost:11434", "qwen2.5:3b", backoff_base=0)

    def test_defaults_retries_two_backoff_one(self):
        client, _ = client_for(ok_response())
        self.assertEqual(client.retries, 2)
        self.assertEqual(client.backoff_base, 1.0)


class ValidationTest(unittest.TestCase):
    def test_invalid_provider_raises_value_error(self):
        with self.assertRaises(ValueError):
            LLMClient("openai", "http://localhost:11434", "qwen2.5:3b")

    def test_empty_base_url_raises_value_error(self):
        with self.assertRaises(ValueError):
            LLMClient("ollama", "", "qwen2.5:3b")

    def test_empty_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            LLMClient("ollama", "http://localhost:11434", "")


class LifecycleTest(unittest.TestCase):
    def test_close_closes_session(self):
        client, session = client_for(ok_response())
        client.close()
        self.assertTrue(session.closed)

    def test_close_idempotent(self):
        client, session = client_for(ok_response())
        client.close()
        client.close()  # must not raise
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
