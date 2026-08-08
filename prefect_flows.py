"""Prefect flows for wallatag.

Prefect is the sole scheduler for wallatag; this module is never imported by
the wallatag package, which keeps zero Prefect dependency. The flow is executed
by a Prefect worker that runs as a process inside the Dokku container where
wallatag is installed and configured (git-bug issue 3d0b22f).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from prefect import flow

if TYPE_CHECKING:
    from prefect.flows import Flow


def build_wallatag_command(
    max_articles: int,
    tag_policy: str | None,
    focus: str | None,
) -> list[str]:
    """Build the `wallatag run` command line for a batch."""
    cmd = ["wallatag", "run", "--max", str(max_articles)]
    if tag_policy is not None:
        cmd += ["--tag-policy", tag_policy]
    if focus is not None:
        cmd += ["--focus", focus]
    return cmd


@flow(log_prints=True)
def wallatag_batch(
    max_articles: int = 50,
    tag_policy: str | None = None,
    focus: str | None = None,
) -> str:
    """Run one headless wallatag batch against the wallabag API.

    The flow executes inside the Dokku container, so it shells out to the
    installed `wallatag` console script (which reads WALLATAG_* env vars).
    Returns the captured stdout on success. Raises on a non-zero exit so
    Prefect marks the run Failed and can notify on problems.
    """
    if shutil.which("wallatag") is None:
        raise RuntimeError(
            "wallatag console script not found on PATH; is the package installed?"
        )
    try:
        completed = subprocess.run(
            build_wallatag_command(max_articles, tag_policy, focus),
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("wallatag run timed out after 1800s") from exc
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"wallatag run exited with code {completed.returncode}: {detail}"
        )
    return completed.stdout.strip()
