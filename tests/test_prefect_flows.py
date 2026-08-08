"""Tests for prefect_flows (skipped when the prefect group isn't installed)."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch


class PrefectFlowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import prefect_flows

        cls.prefect_flows = prefect_flows

    def test_command_build_defaults(self) -> None:
        self.assertEqual(
            self.prefect_flows.build_wallatag_command(50, None, None),
            ["wallatag", "run", "--max", "50"],
        )

    def test_command_build_full(self) -> None:
        self.assertEqual(
            self.prefect_flows.build_wallatag_command(10, "all", "methods"),
            [
                "wallatag",
                "run",
                "--max",
                "10",
                "--tag-policy",
                "all",
                "--focus",
                "methods",
            ],
        )

    def test_flow_defined(self) -> None:
        self.assertTrue(hasattr(self.prefect_flows, "wallatag_batch"))
        self.assertEqual(self.prefect_flows.wallatag_batch.name, "wallatag-batch")

    def test_missing_console_script_raises(self) -> None:
        with patch("prefect_flows.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                self.prefect_flows.wallatag_batch(max_articles=50)

    def test_nonzero_exit_raises_with_stderr(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=2, stdout="", stderr="boom"
        )
        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.subprocess.run", return_value=completed):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.wallatag_batch(max_articles=50)
        self.assertIn("boom", str(ctx.exception))

    def test_success_returns_stdout(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="tagged 3 articles", stderr=""
        )
        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.subprocess.run", return_value=completed):
            result = self.prefect_flows.wallatag_batch(max_articles=50)
        self.assertEqual(result, "tagged 3 articles")

    def test_timeout_raises(self) -> None:
        with patch("prefect_flows.shutil.which", return_value="/usr/local/bin/wallatag"), \
             patch("prefect_flows.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="wallatag", timeout=1800)):
            with self.assertRaises(RuntimeError) as ctx:
                self.prefect_flows.wallatag_batch(max_articles=50)
        self.assertIn("timed out after 1800s", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
