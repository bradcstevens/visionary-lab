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

Wiring order is deliberate: ``configure_dispatcher_dependencies`` runs
BEFORE the ``JobWorker`` is constructed so a defensive immediate-pickup
path inside ``JobWorker.__init__`` (none today, but cheap insurance)
would already see configured factories. Pinned by a unit test.
"""
from __future__ import annotations

import asyncio
import logging

from backend.core.job_queue import JobQueue
from backend.core.job_store import JobStore
from backend.core.job_worker import JobWorker
from backend.core.progress_estimator import ProgressEstimator
from backend.core.staging_dispatcher import (
    configure_dispatcher_dependencies,
    staging_dispatcher,
)
from backend.core.staging_pipeline import StagingPipeline
from backend.core.staging_storage import StagingStorageService

logger = logging.getLogger(__name__)


def _build_staging_pipeline() -> StagingPipeline:
    """Mirror of ``get_staging_pipeline`` from ``backend/api/endpoints/staging.py``.

    Local imports for the heavy dependencies (LLM client, image
    analyzer, image pipeline, blob service) so this module's top-level
    import path stays light — tests of ``build_worker`` patch this
    function out and never trigger Foundry/Cosmos client construction.
    """
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    from backend.core.azure_storage import AzureBlobStorageService
    from backend.core.config import settings
    from backend.core.image_pipeline import ImagePipelineService

    analyzer = ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )
    image_pipeline = ImagePipelineService()
    blob_service = AzureBlobStorageService()
    storage = StagingStorageService()

    return StagingPipeline(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyzer=analyzer,
        image_pipeline=image_pipeline,
        storage_service=storage,
        blob_service=blob_service,
    )


def build_worker() -> JobWorker:
    """Construct the production ``JobWorker``.

    Pure factory — testable in isolation by patching the module-level
    constructors (``JobStore``, ``JobQueue``, ``ProgressEstimator``,
    ``StagingStorageService``, ``_build_staging_pipeline``,
    ``JobWorker``, ``configure_dispatcher_dependencies``) via
    ``monkeypatch.setattr``. See ``tests/test_worker_main.py``.
    """
    store = JobStore()
    queue = JobQueue()

    # ProgressEstimator reads historical p50 data from the JobStore
    # container. Sharing the container reference keeps a single Cosmos
    # read path; passing False here would silence synthetic progress
    # entirely, which we don't want in production.
    estimator = ProgressEstimator(container=store.container)

    # Build the heavy services once at startup and capture them in
    # closures so the dispatcher's per-call factory invocation is
    # essentially free. ``configure_dispatcher_dependencies`` accepts
    # zero-arg callables that return services; ``lambda: pipeline``
    # is the cheapest possible factory.
    storage = StagingStorageService()
    pipeline = _build_staging_pipeline()

    configure_dispatcher_dependencies(
        storage_factory=lambda: storage,
        pipeline_factory=lambda: pipeline,
        store_factory=lambda: store,
    )

    return JobWorker(
        queue=queue,
        store=store,
        dispatcher=staging_dispatcher,
        estimator=estimator,
    )


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
