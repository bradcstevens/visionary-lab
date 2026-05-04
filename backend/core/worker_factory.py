"""``backend.core.worker_factory`` — shared JobWorker construction logic.

Both the standalone worker entry point (``backend.worker_main``, the
``ROLE=worker`` Container App) and the embedded worker
(``backend.core.embedded_worker``, the dev API process) construct a
``JobWorker`` the same way: same JobStore + JobQueue + ProgressEstimator
wiring, same staging-pipeline construction, same dispatcher
configuration. Keeping that wiring in ONE place prevents the two paths
from drifting (e.g. one wires an estimator and the other doesn't, so
synthetic-progress works in prod but not in dev — exactly the kind of
quietly-different bug a shared factory eliminates).

Public surface:

  - ``build_staging_pipeline()`` — heavy: instantiates the Foundry
    LLM client, image analyzer, image pipeline, blob storage, and
    staging storage services. Tests typically patch this out.

  - ``build_worker()`` — pure factory; returns a constructed
    ``JobWorker`` with the dispatcher already wired and the estimator
    wired against the JobStore container.

Wiring order is deliberate: ``configure_dispatcher_dependencies`` runs
BEFORE the ``JobWorker`` is constructed so any defensive
immediate-pickup path inside ``JobWorker.__init__`` would already see
configured factories. Pinned by a unit test in
``tests/test_worker_factory.py``.
"""
from __future__ import annotations

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


def build_staging_pipeline() -> StagingPipeline:
    """Construct the heavy ``StagingPipeline`` used by the dispatcher.

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
    """Construct a fully-wired production ``JobWorker``.

    Pure factory — testable in isolation by patching the module-level
    constructors (``JobStore``, ``JobQueue``, ``ProgressEstimator``,
    ``StagingStorageService``, ``build_staging_pipeline``,
    ``JobWorker``, ``configure_dispatcher_dependencies``) via
    ``monkeypatch.setattr``. See ``tests/test_worker_factory.py``.
    """
    store = JobStore()
    queue = JobQueue()

    # ProgressEstimator reads historical p50 data from the JobStore
    # container. Sharing the container reference keeps a single Cosmos
    # read path; passing False here would silence synthetic progress
    # entirely, which we don't want in production OR development.
    estimator = ProgressEstimator(container=store.container)

    # Build the heavy services once at startup and capture them in
    # closures so the dispatcher's per-call factory invocation is
    # essentially free. ``configure_dispatcher_dependencies`` accepts
    # zero-arg callables that return services; ``lambda: pipeline``
    # is the cheapest possible factory.
    storage = StagingStorageService()
    pipeline = build_staging_pipeline()

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
