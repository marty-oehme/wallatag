"""Tests for deploy/release.py (skipped when the prefect group isn't installed).

The release script talks to a live Prefect API, so the API path is not unit
tested; these tests only verify the module imports cleanly and its default
work pool name, mirroring tests/test_prefect_flows.py's skip guard.
"""

from __future__ import annotations

import unittest


class ReleaseScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import prefect  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("prefect not installed (uv sync --group prefect)")
        import deploy.release

        cls.release = deploy.release

    def test_module_imports(self) -> None:
        self.assertTrue(callable(self.release.main))
        self.assertTrue(callable(self.release.ensure_work_pool))
        self.assertTrue(callable(self.release.log))

    def test_default_work_pool_name(self) -> None:
        self.assertEqual(self.release.WORK_POOL_NAME, "wallatag-pool")


if __name__ == "__main__":
    unittest.main()
