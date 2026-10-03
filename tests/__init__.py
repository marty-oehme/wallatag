"""wallatag test suite.

Tests are split into tiers by ``WALLATAG_TEST_TIERS`` (unit / integration /
e2e); see ``tests/_tags.py`` for the mechanism and the default. This package
marker exists so that ``python -m unittest discover -s tests`` sees a package
and honours the ``load_tests`` hook defined here.
"""

from tests._tags import load_tests  # noqa: F401  (re-exported for unittest)

__all__ = ["load_tests"]
