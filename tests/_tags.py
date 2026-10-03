"""Test tier tags for the stdlib unittest suite.

The suite runs with ``python -m unittest discover -s tests -t .`` and unittest
has no built-in marker mechanism (unlike pytest). These helpers provide one with
no new dependency, by combining two features that do exist:

* each tier is a subclass of :class:`unittest.TestCase`, and
* ``unittest``'s loader only discovers subclasses of ``TestCase``.

A test belongs to a tier simply by inheriting from the matching base class.
Because the bases are plain ``TestCase`` subclasses they are still discovered and
run by default, which keeps the "run everything in CI" behaviour.

Tiers
-----
``UnitTest``
    The default. No real external process, server, database engine or network
    call. Optional packages may be *imported* (and skipped when absent), but the
    test does not require anything outside the process.

``IntegrationTest``
    Requires an external dependency but no outside service: an optional package
    that must actually work (e.g. Prefect spawning its ephemeral API server
    subprocess), a separate interpreter/process, or the real SQLite engine
    through concurrent connections. Deterministic and offline.

``E2ETest``
    Requires a real external service or process: a live Wallabag instance, a
    live Prefect API, or a real network call. Not expected to run everywhere.

Selecting tiers
---------------
``WALLATAG_TEST_TIERS`` is a comma-separated list of the tiers to run
(``unit``, ``integration``, ``e2e``); names may be abbreviated to any unique
prefix and ``all`` selects everything. Unset means only ``unit``, so the quick
local default stays fast while CI sets ``all``.

    WALLATAG_TEST_TIERS=unit              # unit only (default)
    WALLATAG_TEST_TIERS=unit,integration  # skip e2e
    WALLATAG_TEST_TIERS=all               # everything

The filtering happens in :func:`load_tests`. When unittest descends into the
``tests`` package and finds that hook, it stops recursing on its own and calls
it with a suite built only from ``tests/__init__.py`` (which holds no tests).
The hook therefore re-runs discovery over the package directory and then drops
the tests whose tier was not requested. Discovery must use ``-t .`` so that
``tests`` is imported as a package; otherwise unittest globs the modules
directly, never calls the hook, and runs every tier regardless of the variable.
"""

from __future__ import annotations

import os
import sys
import unittest

__all__ = [
    "UnitTest",
    "IntegrationTest",
    "E2ETest",
    "selected_tiers",
    "TIER_ENV",
]

TIER_ENV = "WALLATAG_TEST_TIERS"

# Tier name -> the base class that marks a test as belonging to it. Order is the
# canonical ordering used in help/error messages.
_TIERS = ("unit", "integration", "e2e")


class UnitTest(unittest.TestCase):
    """Default tier: no external process, server, database engine or network."""

    tier = "unit"


class IntegrationTest(unittest.TestCase):
    """Needs an optional package or a child process, but no outside service."""

    tier = "integration"


class E2ETest(unittest.TestCase):
    """Needs a live external service or a real network call."""

    tier = "e2e"


def selected_tiers(environ: dict[str, str] | None = None) -> set[str]:
    """The tiers to run, from ``WALLATAG_TEST_TIERS`` (default: ``{"unit"}``).

    Names may be abbreviated to any unique prefix and ``all`` expands to every
    tier. An unknown or ambiguous name raises ``ValueError`` so a typo in CI
    cannot silently skip the whole suite.
    """
    env = os.environ if environ is None else environ
    raw = env.get(TIER_ENV)
    if raw is None or not raw.strip():
        return {"unit"}

    chosen: set[str] = set()
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name == "all":
            chosen.update(_TIERS)
            continue
        matches = [t for t in _TIERS if t.startswith(name)]
        if len(matches) != 1:
            raise ValueError(
                f"{TIER_ENV}: {name!r} matches {matches or 'no tier'}; "
                f"expected one of {', '.join(_TIERS)}"
            )
        chosen.add(matches[0])
    return chosen or {"unit"}


def _tier_of(test: unittest.TestCase) -> str:
    return getattr(type(test), "tier", "unit")


def _discover_package(loader, package, pattern):  # noqa: ARG001
    """Recurse into a package's submodules, bypassing this filter.

    ``load_tests`` runs *instead of* discovery recursing into the package, so
    the hook receives only the (empty) suite built from the package's
    ``__init__``. It has to drive discovery itself.
    """
    start = os.path.dirname(os.path.abspath(package.__file__))
    top = os.path.dirname(start)
    return loader.discover(start, pattern=pattern, top_level_dir=top)


def load_tests(loader, tests, pattern):  # noqa: ARG001 - unittest protocol
    """Filter discovered tests down to the requested tiers.

    unittest calls ``load_tests(loader, tests, pattern)`` when it finds the
    function in a package it is descending into (here: ``tests``). Because the
    presence of the hook suppresses unittest's own recursion, we re-run
    discovery over this module's directory and then drop the tests whose tier
    was not requested. Returning ``tests`` unchanged keeps default behaviour
    for any module that does not define its own hook.
    """
    wanted = selected_tiers()

    module = sys.modules.get(__name__)
    if module is not None and getattr(module, "__file__", None):
        tests = _discover_package(loader, module, pattern)

    if wanted >= set(_TIERS):
        return tests

    def keep(obj):
        if isinstance(obj, unittest.TestSuite):
            result = unittest.TestSuite()
            for child in obj:
                result.addTests(keep(child))
            return result
        if isinstance(obj, unittest.TestCase):
            if _tier_of(obj) not in wanted:
                return unittest.TestSuite()
            return unittest.TestSuite([obj])
        return unittest.TestSuite()

    return keep(tests)
