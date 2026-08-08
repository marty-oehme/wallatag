"""LLM provider client for the OpenAI-compatible chat completions API.

Thin requests-based client with ZERO extra dependencies: the project's single
runtime dep (``requests``) talks to the provider. Two provider flavors share
one wire format (the OpenAI-compatible chat completions endpoint); they differ
only in the URL. Ollama's OpenAI shim hangs off ``/v1`` (base_url like
``http://localhost:11434`` has no version prefix of its own), while an
openai-compatible gateway is expected to include the version prefix in its
``base_url`` (e.g. ``https://api.example.com/v1``).

Auth: when ``api_key`` is set (non-empty), every request carries an
``Authorization: Bearer <api_key>`` header for hosted openai-compatible
gateways that require a key (OpenAI, OpenRouter, ...); with no api_key the
client sends no Authorization header at all (keyless servers such as a local
ollama instance). The key is never included in any LLMError message, exception
text or logged data.

This is the ONLY module in the tagger stack that performs I/O; everything
above it (config, tagger, pipelines) talks to the injectable client. It
imports only from wallatag.config, so there are no import cycles with
wallatag.tagger or the pipeline modules.
"""

from __future__ import annotations

import requests

from wallatag.config import VALID_AI_PROVIDERS


class LLMError(Exception):
    """Raised when an LLM provider request fails.

    ``status`` holds the HTTP status code for HTTP-level failures, or ``None``
    for transport-level failures (connection error, timeout, ...).
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LLMClient:
    """Raw HTTP client for the OpenAI-compatible chat completions endpoint.

    ``session`` is normally created internally; injecting one is how tests mock
    HTTP (same pattern as WallabagClient). ``timeout`` is applied to every
    request. ``api_key`` (optional) enables ``Authorization: Bearer`` auth for
    keyed openai-compatible gateways; it is only stored here and never appears
    in any exception or log output. ``complete`` wraps ALL failures (transport
    errors, non-2xx HTTP status, non-JSON bodies, missing/malformed content) in
    ``LLMError`` so the tagger and pipelines only ever have to handle one
    exception type.
    """

    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if provider not in VALID_AI_PROVIDERS:
            raise ValueError(
                "invalid ai provider %r; valid choices: %s"
                % (provider, ", ".join(VALID_AI_PROVIDERS))
            )
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        self.provider: str = provider
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.api_key: str = api_key
        self.timeout: float = timeout
        self.session: requests.Session = (
            session if session is not None else requests.Session()
        )

    def _endpoint(self) -> str:
        """Chat completions URL for this provider flavor."""
        if self.provider == "ollama":
            # Ollama's OpenAI-compatible shim lives under /v1; base_url like
            # http://localhost:11434 must NOT be expected to carry a prefix.
            return f"{self.base_url}/v1/chat/completions"
        # openai-compatible: base_url already includes the /v1 prefix.
        return f"{self.base_url}/chat/completions"

    def _redact(self, text: str) -> str:
        """Redact the api_key from gateway-provided text embedded in errors.

        A misconfigured/hostile gateway may echo the bearer key back in an
        error body; scrub it before the text can reach an LLMError (and from
        there logger/print output). No-op for keyless clients: an empty
        api_key must never trigger a replace (``"".replace`` would corrupt
        every message).
        """
        if not self.api_key:
            return text
        return text.replace(self.api_key, "***")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """POST a chat completion and return the assistant message content.

        The payload is OpenAI-shaped: model, system+user messages, and a hard
        ``temperature`` of 0 with streaming disabled so the model's output is
        deterministic and a single parseable JSON body.
        """
        url = self._endpoint()
        headers: dict[str, str] = {}
        if self.api_key:
            # Bearer auth for hosted openai-compatible gateways (OpenAI,
            # OpenRouter, ...). Keyless servers (local ollama) get no header.
            # SECURITY: headers are never included in LLMError messages.
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "stream": False,
        }
        try:
            resp = self.session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if not 200 <= resp.status_code < 300:
            raise LLMError(self._http_error(resp), status=resp.status_code)
        try:
            body = resp.json()
        except ValueError as exc:
            raise LLMError(
                f"LLM response is not JSON: {self._redact(resp.text)[:200]!r}",
                status=resp.status_code,
            ) from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"LLM response missing choices[0].message.content: "
                f"{self._redact(resp.text)[:200]!r}",
                status=resp.status_code,
            ) from exc
        if not isinstance(content, str):
            raise LLMError(
                f"LLM response content is not a string: {self._redact(repr(content))}",
                status=resp.status_code,
            )
        return content

    def _http_error(self, resp: requests.Response) -> str:
        snippet = self._redact(resp.text or "").strip().replace("\n", " ")[:200]
        if snippet:
            return f"LLM request failed with status {resp.status_code}: {snippet}"
        return f"LLM request failed with status {resp.status_code}"

    def close(self) -> None:
        """Close the session. Safe to call more than once."""
        self.session.close()
