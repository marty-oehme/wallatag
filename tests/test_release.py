"""Tests for deploy/release.py (skipped when the prefect group isn't installed).

The release script talks to a live Prefect API, so the API path is not unit
tested; these tests only verify the module imports cleanly, its default work
pool name, and that main() runs the block-ensure step before `prefect deploy`
mirroring tests/test_prefect_flows.py's skip guard.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch


class ReleaseScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "prefect not installed (uv sync --group prefect)"
            )
        import deploy.release

        cls.release = deploy.release

    def test_module_imports(self) -> None:
        self.assertTrue(callable(self.release.main))
        self.assertTrue(callable(self.release.ensure_work_pool))
        self.assertTrue(callable(self.release.log))

    def test_default_work_pool_name(self) -> None:
        self.assertEqual(self.release.WORK_POOL_NAME, "wallatag-pool")

    def test_main_ensures_block_before_deploy(self) -> None:
        calls = []
        with (
            patch(
                "deploy.release.ensure_work_pool",
                side_effect=lambda: calls.append("pool"),
            ),
            patch(
                "blocks.ensure_wallatag_llm_credentials_block",
                side_effect=lambda *a, **k: calls.append("block"),
            ),
            patch(
                "blocks.ensure_wallabag_credentials_block",
                side_effect=lambda *a, **k: calls.append("wallabag-block"),
            ),
            patch(
                "deploy.release.subprocess.run",
                side_effect=lambda *a, **k: calls.append("deploy"),
            ),
        ):
            self.release.main()
        self.assertEqual(calls, ["pool", "block", "wallabag-block", "deploy"])


if __name__ == "__main__":
    unittest.main()
