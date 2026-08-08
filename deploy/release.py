"""Release-phase script for the wallatag Dokku app.

Runs on every dokku deploy (Procfile `release:`), BEFORE the worker starts.
Responsibilities:

1. Idempotently ensure the `wallatag-pool` work pool (type `process`) exists:
   `prefect deploy` hard-fails if the pool is missing, and the worker would not
   have created it yet at release time.
2. Idempotently ensure the `wallatag-llm` LLMCredentials block exists
   (auto-created on the first release, seeded from the WALLATAG_AI_* env vars
   when the provider/base_url/model trio is set, else empty): so the user
   does not have to create the credentials block by hand in the Prefect UI.
3. Run `prefect deploy --all`, which registers the single committed
   `prefect.yaml` deployment (wallatag-batch). `prefect deploy --all` is
   idempotent, so re-runs on every deploy are harmless.

Any failure exits non-zero so dokku fails the deploy (fail loud).

The release phase normally runs from the deployed image with cwd=/app, but the
script pins its own working directory to the repo root so `prefect deploy`
finds `prefect.yaml` regardless of where it is invoked from. The repo root is
also put on sys.path: run as `python deploy/release.py`, only the script's own
directory is on sys.path by default, so the sibling `blocks` module would not
be importable otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORK_POOL_NAME = "wallatag-pool"


def log(message: str) -> None:
    print(f"[release] {message}", flush=True)


def ensure_work_pool() -> None:
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.actions import WorkPoolCreate
    from prefect.exceptions import ObjectAlreadyExists

    try:
        with get_client(sync_client=True) as client:
            client.create_work_pool(
                WorkPoolCreate(name=WORK_POOL_NAME, type="process")
            )
    except ObjectAlreadyExists:
        log(f"work pool {WORK_POOL_NAME!r} already exists")
    else:
        log(f"created work pool {WORK_POOL_NAME!r} (type process)")


def main() -> None:
    log(f"work pool: {WORK_POOL_NAME}")
    ensure_work_pool()
    from blocks import ensure_wallatag_llm_credentials_block

    ensure_wallatag_llm_credentials_block()
    subprocess.run(
        ["prefect", "deploy", "--all"],
        check=True,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loud: dokku must see the deploy fail
        print(f"[release] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
