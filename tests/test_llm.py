"""Tests for wallatag.llm: LLMClient URL building, payload shape and errors.

HTTP is faked with a small FakeSession that records ``post(url, json,
headers, timeout)`` calls and returns a canned FakeResponse.
"""

import json
import unittest

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
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
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
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


def client_for(response, error=None, **kwargs):
    session = FakeSession(response=response, error=error)
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
            None, error=requests.ConnectionError("boom"), api_key="sk-secret-2"
        )
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertNotIn("sk-secret-2", str(ctx.exception))


class ErrorTest(unittest.TestCase):
    def test_http_error_raises_llm_error_with_status(self):
        # Keyless client (default api_key=""): the error body must be
        # preserved intact and never corrupted by a redaction pass (a bogus
        # unconditional replace of an empty key would corrupt every message).
        client, _ = client_for(FakeResponse(status_code=500, body={"error": "boom"}))
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("500", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))
        self.assertNotIn("***", str(ctx.exception))

    def test_transport_error_raises_llm_error(self):
        client, _ = client_for(
            None, error=requests.ConnectionError("connection refused")
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
