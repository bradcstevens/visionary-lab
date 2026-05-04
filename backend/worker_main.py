"""``backend.worker_main`` — production entrypoint for the JobWorker.

The same Docker image runs both Container Apps; this module is what
``python -m backend.worker_main`` invokes inside the worker container
(``infra/modules/containerAppWorker.bicep`` sets ``ROLE=worker`` and
overrides the container command). The API container keeps its
``uvicorn backend.main:app`` entrypoint untouched.

Bootstrap responsibilities (per PRD § "Production worker entrypoint"):

  1. Construct ``JobStore`` and ``JobQueue`` (managed-identity backed).
  2. Construct ``ProgressEstimator`` against the JobStore container so
     the synthetic-progress heartbeat in ``JobWorker`` has historical
     p50 data to read from.
  3. Construct the heavy staging dependencies (LLM client, image
     analyzer, image pipeline, blob storage, staging storage) and a
     single ``StagingPipeline`` instance — sharing one pipeline across
     all dispatch calls keeps construction cost O(1) per worker
     replica instead of O(N) per dispatch.
  4. Wire the dispatcher's module-level dependency factories via
     ``configure_dispatcher_dependencies(...)``. Both factories return
     the SAME instance every call (lambda closures over the
     constructed singletons).
  5. Construct the ``JobWorker`` with ``dispatcher=staging_dispatcher``
     and run its poll loop forever.

The actual construction wiring lives in
``backend.core.worker_factory.build_worker`` so the FastAPI lifespan
embedded-worker path (``backend.core.embedded_worker``) shares the
same factory and the two paths cannot drift. ``build_worker`` is
re-exported here for backward compatibility with anything importing
``backend.worker_main.build_worker`` directly.
"""
from __future__ import annotations

import asyncio
import logging

# Re-exported for backward compatibility — older callers and tests
# imported ``build_worker`` from this module before it was extracted
# into ``worker_factory`` per issue 001 of the
# active-and-queued-jobs-ux-redesign PRD.
from backend.core.worker_factory import build_worker  # noqa: F401

logger = logging.getLogger(__name__)


async def main() -> None:
    """Entry point. Builds the worker and runs its poll loop forever.

    The poll loop only exits on ``stop()`` (currently never called
    from inside the process — Container App lifecycle is in charge of
    termination, and visibility-timeout re-delivery covers in-flight
    work that gets killed mid-dispatch).
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("worker_main.start")
    worker = build_worker()
    logger.info("worker_main.run")
    await worker.run()


if __name__ == "__main__":  # pragma: no cover — entry-point glue
    asyncio.run(main())
