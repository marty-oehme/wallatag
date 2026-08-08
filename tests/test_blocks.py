"""Tests for blocks.py (skipped when the prefect group isn't installed).

Block.load/save talk to a live Prefect server, so they are always patched
here; only the pure model behaviour is exercised for real.
"""

from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class BlocksModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import blocks

        cls.blocks = blocks

    def test_block_metadata(self) -> None:
        cls = self.blocks.LLMCredentials
        self.assertEqual(cls._block_type_name, "Wallatag LLM Credentials")
        self.assertEqual(cls._block_type_slug, "wallatag-llm-credentials")

    def test_block_name_constant(self) -> None:
        self.assertEqual(self.blocks.BLOCK_NAME, "wallatag-llm")

    def test_default_fields(self) -> None:
        block = self.blocks.LLMCredentials()
        self.assertEqual(block.provider, "")
        self.assertEqual(block.base_url, "")
        self.assertEqual(block.model, "")
        self.assertEqual(block.api_key.get_secret_value(), "")
        self.assertIsNone(block.confidence_threshold)

    def test_empty_block_yields_empty_env(self) -> None:
        block = self.blocks.LLMCredentials()
        self.assertEqual(block.llm_env(), {})

    def test_llm_env_maps_populated_fields(self) -> None:
        block = self.blocks.LLMCredentials(
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            model="gpt-4o-mini",
            api_key="sk-secret",
            confidence_threshold=0.7,
        )
        self.assertEqual(
            block.llm_env(),
            {
                "WALLATAG_AI_PROVIDER": "openai-compatible",
                "WALLATAG_AI_BASE_URL": "https://api.example.com/v1",
                "WALLATAG_AI_MODEL": "gpt-4o-mini",
                "WALLATAG_AI_API_KEY": "sk-secret",
                "WALLATAG_AI_CONFIDENCE_THRESHOLD": "0.7",
            },
        )

    def test_llm_env_omits_empty_api_key_and_none_threshold(self) -> None:
        block = self.blocks.LLMCredentials(
            provider="ollama",
            base_url="http://localhost:11434",
            model="qwen2.5:3b",
        )
        self.assertEqual(
            block.llm_env(),
            {
                "WALLATAG_AI_PROVIDER": "ollama",
                "WALLATAG_AI_BASE_URL": "http://localhost:11434",
                "WALLATAG_AI_MODEL": "qwen2.5:3b",
            },
        )


class EnsureBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import blocks

        cls.blocks = blocks

    def test_seeds_from_env_when_trio_set(self) -> None:
        saved = {}

        def fake_save(self, name, overwrite=False, client=None):
            saved["block"] = self
            saved["name"] = name

        messages = []
        with patch.object(
            self.blocks.LLMCredentials, "load", side_effect=Exception("missing")
        ), patch.object(self.blocks.LLMCredentials, "save", fake_save), patch.dict(
            os.environ,
            {
                "WALLATAG_AI_PROVIDER": "ollama",
                "WALLATAG_AI_BASE_URL": "http://localhost:11434",
                "WALLATAG_AI_MODEL": "qwen2.5:3b",
                "WALLATAG_AI_API_KEY": "sk-123",
                "WALLATAG_AI_CONFIDENCE_THRESHOLD": "0.85",
            },
            clear=True,
        ):
            self.blocks.ensure_wallatag_llm_credentials_block(log=messages.append)

        block = saved["block"]
        self.assertEqual(saved["name"], "wallatag-llm")
        self.assertEqual(block.provider, "ollama")
        self.assertEqual(block.base_url, "http://localhost:11434")
        self.assertEqual(block.model, "qwen2.5:3b")
        self.assertEqual(block.api_key.get_secret_value(), "sk-123")
        self.assertEqual(block.confidence_threshold, 0.85)
        self.assertTrue(any("seeding" in m for m in messages))
        self.assertTrue(any("created" in m for m in messages))

    def test_creates_empty_block_without_env(self) -> None:
        saved = {}

        def fake_save(self, name, overwrite=False, client=None):
            saved["block"] = self

        messages = []
        with patch.object(
            self.blocks.LLMCredentials, "load", side_effect=Exception("missing")
        ), patch.object(self.blocks.LLMCredentials, "save", fake_save), patch.dict(
            os.environ, {}, clear=True
        ):
            self.blocks.ensure_wallatag_llm_credentials_block(log=messages.append)

        block = saved["block"]
        self.assertEqual(block.provider, "")
        self.assertEqual(block.base_url, "")
        self.assertEqual(block.model, "")
        self.assertEqual(block.api_key.get_secret_value(), "")
        self.assertIsNone(block.confidence_threshold)
        self.assertTrue(any("creating empty" in m for m in messages))
        self.assertTrue(any("trio" in m for m in messages))

    def test_partial_env_trio_creates_empty_block(self) -> None:
        # Seeding requires the full provider/base_url/model trio.
        saved = {}

        def fake_save(self, name, overwrite=False, client=None):
            saved["block"] = self

        with patch.object(
            self.blocks.LLMCredentials, "load", side_effect=Exception("missing")
        ), patch.object(self.blocks.LLMCredentials, "save", fake_save), patch.dict(
            os.environ, {"WALLATAG_AI_PROVIDER": "ollama"}, clear=True
        ):
            self.blocks.ensure_wallatag_llm_credentials_block()

        self.assertEqual(saved["block"].provider, "")
        self.assertEqual(saved["block"].base_url, "")
        self.assertEqual(saved["block"].model, "")

    def test_existing_block_not_overwritten(self) -> None:
        with patch.object(
            self.blocks.LLMCredentials,
            "load",
            return_value=self.blocks.LLMCredentials(provider="ollama"),
        ), patch.object(self.blocks.LLMCredentials, "save") as mock_save, \
            redirect_stdout(io.StringIO()) as out:
            self.blocks.ensure_wallatag_llm_credentials_block()
        mock_save.assert_not_called()
        self.assertIn("already exists", out.getvalue())

    def test_save_failure_logs_warning_and_returns(self) -> None:
        messages = []
        with patch.object(
            self.blocks.LLMCredentials, "load", side_effect=Exception("missing")
        ), patch.object(
            self.blocks.LLMCredentials,
            "save",
            side_effect=RuntimeError("prefect server unreachable"),
        ), patch.dict(
            os.environ,
            {
                "WALLATAG_AI_PROVIDER": "ollama",
                "WALLATAG_AI_BASE_URL": "http://localhost:11434",
                "WALLATAG_AI_MODEL": "qwen2.5:3b",
            },
            clear=True,
        ):
            # Must not raise.
            self.blocks.ensure_wallatag_llm_credentials_block(log=messages.append)

        self.assertTrue(
            any("warning" in m and "could not save" in m for m in messages)
        )

    def test_invalid_threshold_env_skipped_with_warning(self) -> None:
        saved = {}
        messages = []

        def fake_save(self, name, overwrite=False, client=None):
            saved["block"] = self

        with patch.object(
            self.blocks.LLMCredentials, "load", side_effect=Exception("missing")
        ), patch.object(self.blocks.LLMCredentials, "save", fake_save), patch.dict(
            os.environ,
            {
                "WALLATAG_AI_PROVIDER": "ollama",
                "WALLATAG_AI_BASE_URL": "http://localhost:11434",
                "WALLATAG_AI_MODEL": "qwen2.5:3b",
                "WALLATAG_AI_CONFIDENCE_THRESHOLD": "not-a-number",
            },
            clear=True,
        ):
            self.blocks.ensure_wallatag_llm_credentials_block(log=messages.append)

        self.assertIsNone(saved["block"].confidence_threshold)
        self.assertTrue(
            any("warning" in m and "not a number" in m for m in messages)
        )


if __name__ == "__main__":
    unittest.main()
