"""Tests for wallatag.llm: LLMClient URL building, payload shape and errors.

HTTP is faked with a small FakeSession that records ``post(url, json,
timeout)`` calls and returns a canned FakeResponse.
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

    def post(self, url, json=None, timeout=None, **kwargs):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
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


class ErrorTest(unittest.TestCase):
    def test_http_error_raises_llm_error_with_status(self):
        client, _ = client_for(FakeResponse(status_code=500, body={"error": "boom"}))
        with self.assertRaises(LLMError) as ctx:
            client.complete(SYSTEM, USER)
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("500", str(ctx.exception))

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
