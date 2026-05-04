"""``staging_dispatcher`` — kind-switch entry point for the JobWorker.

Owns routing between job kinds. ``staging_dispatcher(job, is_cancelled)``
conforms to the ``backend.core.job_worker.Dispatcher`` contract so it
can be passed straight to ``JobWorker(dispatcher=staging_dispatcher)``
in the production worker bootstrap (issue 007).

Today the module knows about ``regenerate_variation``. Issue 005 fills
in the ``generate_project`` branch (which currently falls through to
the unknown-kind error so the placeholder is visible).

Dependencies (storage and pipeline) are wired in via
``configure_dispatcher_dependencies``. Production calls this once at
worker startup (issue 007); tests inject mocks via the same path. The
indirection keeps this module side-effect free at import time so it
stays cheap and safe to import from tests and other modules.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-hint imports only
    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.staging_storage import StagingStorageService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CancelCheck = Callable[[], bool]
JobDoc = dict[str, Any]
DispatchResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Module-level dependency injection
# ---------------------------------------------------------------------------

# Production: ``backend/worker_main.py`` (issue 007) calls
# ``configure_dispatcher_dependencies`` once at startup with real factories.
# Tests: call the same function with mocks.
#
# Factories (rather than instances) so each dispatch call can ask for a
# fresh service if the production wiring ever wants per-call objects.
_storage_factory: Optional[Callable[[], "StagingStorageService"]] = None
_pipeline_factory: Optional[Callable[[], "StagingPipeline"]] = None


def configure_dispatcher_dependencies(
    *,
    storage_factory: Optional[Callable[[], "StagingStorageService"]] = None,
    pipeline_factory: Optional[Callable[[], "StagingPipeline"]] = None,
) -> None:
    """Wire dependencies for the dispatchers.

    Production: called once at worker startup by ``worker_main`` (issue
    007). Tests: called per test (typically inside a fixture) to inject
    mocks. Either argument may be ``None`` to leave that factory
    unchanged — the production bootstrap may want to wire storage and
    pipeline separately.
    """
    global _storage_factory, _pipeline_factory
    if storage_factory is not None:
        _storage_factory = storage_factory
    if pipeline_factory is not None:
        _pipeline_factory = pipeline_factory


def reset_dispatcher_dependencies() -> None:
    """Clear all configured dependencies. For test isolation only —
    production code should never call this once the worker is wired."""
    global _storage_factory, _pipeline_factory
    _storage_factory = None
    _pipeline_factory = None


def _get_storage() -> "StagingStorageService":
    if _storage_factory is None:
        raise RuntimeError(
            "staging_dispatcher: storage_factory not configured. Call "
            "configure_dispatcher_dependencies(storage_factory=...) in "
            "the worker bootstrap or test setup."
        )
    return _storage_factory()


def _get_pipeline() -> "StagingPipeline":
    if _pipeline_factory is None:
        raise RuntimeError(
            "staging_dispatcher: pipeline_factory not configured. Call "
            "configure_dispatcher_dependencies(pipeline_factory=...) in "
            "the worker bootstrap or test setup."
        )
    return _pipeline_factory()


# ---------------------------------------------------------------------------
# regenerate_variation_dispatcher
# ---------------------------------------------------------------------------


async def regenerate_variation_dispatcher(
    job: JobDoc,
    is_cancelled: CancelCheck,
) -> DispatchResult:
    """Run a queued single-variation regen job to completion.

    Loads the project from storage, finds the room and variation, and
    streams ``StagingPipeline.process_single_variation`` to completion.
    Returns the final yielded event as the job result so the worker can
    persist it on the job doc.

    The adapted prompt is sourced in this order:
      1. ``job["payload"]["adapted_prompt"]`` if the POST handler stashed
         one (the path issue 005/006 will use for the queued initial
         generation flow's per-variation continuations).
      2. ``variation.generation_metadata.adapted_prompt`` if the prior
         attempt persisted one (the standard retry path, mirroring what
         the legacy POST endpoint reads at line 1102 of staging.py).

    No prompt anywhere → ``ValueError``. The dispatcher intentionally
    does NOT call ``pipeline.adapt_prompt`` here; that decision belongs
    in the POST handler so the worker side stays a pure consumer of
    pre-computed inputs (matches the brief-reuse contract for
    ``generate_project`` from issue 004).

    Cancellation: ``is_cancelled()`` is polled after the pipeline
    stream completes. If True, ``JobCancelled`` is raised so the
    worker routes the message to ``complete`` (drop) rather than
    ``abandon`` (retry).
    """
    # Imports inside the function so the test fixture's fresh-state
    # ``reset_dispatcher_dependencies`` doesn't pull in heavy deps it
    # doesn't need (StagingProject -> pydantic; JobCancelled -> JobWorker).
    from backend.core.job_worker import JobCancelled
    from backend.models.staging import StagingProject

    storage = _get_storage()
    pipeline = _get_pipeline()

    project_id = job["project_id"]
    payload = job.get("payload") or {}
    room_id = payload.get("room_id") or job.get("room_id")
    variation_id = payload.get("variation_id") or job.get("variation_id")
    if not room_id or not variation_id:
        raise ValueError(
            "regenerate_variation job is missing room_id or variation_id "
            f"(job_id={job.get('id')!r})"
        )

    project_data = storage.get_project(project_id)
    if not project_data:
        raise ValueError(f"Project not found: {project_id}")

    # Same Pydantic-friendly cleanup the legacy POST handler does (see
    # backend/api/endpoints/staging.py — every endpoint that materializes
    # a StagingProject from storage strips ``doc_type`` and Cosmos
    # internals like ``_etag`` before calling the constructor).
    clean = {
        k: v
        for k, v in project_data.items()
        if k != "doc_type" and not k.startswith("_")
    }
    project = StagingProject(**clean)

    room = next((r for r in project.rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(
            f"Room not found: {room_id} (project={project_id})"
        )

    variation = next(
        (v for v in room.variations if v.id == variation_id),
        None,
    )
    if variation is None:
        raise ValueError(
            f"Variation not found: {variation_id} "
            f"(project={project_id}, room={room_id})"
        )

    adapted_prompt = payload.get("adapted_prompt")
    if not adapted_prompt and variation.generation_metadata is not None:
        meta = variation.generation_metadata
        # ``generation_metadata`` is typed as the Pydantic ``GenerationMetadata``
        # model on the schema, but storage round-trips can serialize it
        # as a dict. Read both shapes to match the legacy endpoint's
        # tolerant read at lines 1101–1104 of staging.py.
        if isinstance(meta, dict):
            adapted_prompt = meta.get("adapted_prompt")
        else:
            adapted_prompt = getattr(meta, "adapted_prompt", None)

    if not adapted_prompt:
        raise ValueError(
            "regenerate_variation job has no adapted_prompt in payload "
            "and the variation has no prior adapted_prompt to retry. "
            "The POST handler must compute and stash an adapted_prompt "
            "before enqueueing."
        )

    last_event: Optional[dict] = None
    async for event in pipeline.process_single_variation(
        project=project,
        room=room,
        variation=variation,
        adapted_prompt=adapted_prompt,
    ):
        last_event = event

    if is_cancelled():
        raise JobCancelled(
            "cancel_requested observed after process_single_variation "
            f"(job_id={job.get('id')!r})"
        )

    if last_event is None:
        # Defensive: ``process_single_variation`` is documented to yield
        # exactly one event. If the contract ever changes, surface a
        # minimal success result rather than ``None`` so the worker's
        # ``record_completion`` call doesn't trip on an empty doc patch.
        return {
            "status": "completed",
            "room_id": room_id,
            "variation_id": variation_id,
        }
    return last_event


# ---------------------------------------------------------------------------
# Top-level kind-switch
# ---------------------------------------------------------------------------


async def staging_dispatcher(
    job: JobDoc,
    is_cancelled: CancelCheck,
) -> DispatchResult:
    """Top-level kind-switch dispatcher for the ``JobWorker``.

    Routes by ``job["kind"]``. Unknown kinds (including the
    placeholder ``generate_project`` until issue 005 lands) raise
    ``ValueError`` so the worker's normal failure path runs and the
    job is marked failed rather than silently no-op'd.
    """
    kind = job.get("kind")
    if kind == "regenerate_variation":
        return await regenerate_variation_dispatcher(job, is_cancelled)
    raise ValueError(f"Unknown kind: {kind}")


__all__ = [
    "configure_dispatcher_dependencies",
    "reset_dispatcher_dependencies",
    "regenerate_variation_dispatcher",
    "staging_dispatcher",
]
