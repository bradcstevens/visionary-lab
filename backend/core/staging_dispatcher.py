"""``staging_dispatcher`` — kind-switch entry point for the JobWorker.

Owns routing between job kinds. ``staging_dispatcher(job, is_cancelled)``
conforms to the ``backend.core.job_worker.Dispatcher`` contract so it
can be passed straight to ``JobWorker(dispatcher=staging_dispatcher)``
in the production worker bootstrap (issue 007).

The module knows about ``regenerate_variation`` (issue 003) and
``generate_project`` (issue 005).

Dependencies (storage, pipeline, store) are wired in via
``configure_dispatcher_dependencies``. Production calls this once at
worker startup (issue 007); tests inject mocks via the same path. The
indirection keeps this module side-effect free at import time so it
stays cheap and safe to import from tests and other modules.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-hint imports only
    from backend.core.job_store import JobStore
    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.staging_storage import StagingStorageService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants (monkeypatched by tests for fast loops)
# ---------------------------------------------------------------------------

# Project lease poll-acquire interval. Issue 005: when a different
# non-terminal job already holds ``current_project_job_id`` we wait for
# release rather than raising — the visibility-timeout heartbeat (issue
# 001) keeps the queue lease alive for the duration. 5s is small
# relative to the 30s heartbeat interval and 90s queue visibility, and
# large enough that the polling cost is negligible (~120 reads max).
LEASE_POLL_INTERVAL_SECONDS: float = 5.0

# Hard ceiling on how long a single dispatcher invocation will wait for
# the project lease. Crossing this threshold raises ``LeaseBusy`` which
# JobWorker's existing exception path abandons → Storage Queue
# redelivers. 10 minutes covers a typical project generation runtime
# without holding a worker replica indefinitely against a stuck lease.
LEASE_MAX_WAIT_SECONDS: float = 600.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LeaseBusy(Exception):
    """Raised by ``generate_project_dispatcher`` when the project lease
    cannot be acquired within ``LEASE_MAX_WAIT_SECONDS``.

    JobWorker's existing exception handler treats this as any other
    unhandled error: marks the job ``pending`` (or ``failed`` on the
    final attempt), abandons the queue message, Storage Queue
    redelivers after the visibility timeout. After ``MAX_DEQUEUE_COUNT``
    attempts the message poisons. In practice the dispatcher's
    in-process poll-acquire loop (with the visibility heartbeat keeping
    the message alive) absorbs lease contention without ever surfacing
    this exception under normal load.
    """


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
_store_factory: Optional[Callable[[], "JobStore"]] = None


def configure_dispatcher_dependencies(
    *,
    storage_factory: Optional[Callable[[], "StagingStorageService"]] = None,
    pipeline_factory: Optional[Callable[[], "StagingPipeline"]] = None,
    store_factory: Optional[Callable[[], "JobStore"]] = None,
) -> None:
    """Wire dependencies for the dispatchers.

    Production: called once at worker startup by ``worker_main`` (issue
    007). Tests: called per test (typically inside a fixture) to inject
    mocks. Any argument may be ``None`` to leave that factory
    unchanged — the production bootstrap may want to wire storage,
    pipeline, and store separately.
    """
    global _storage_factory, _pipeline_factory, _store_factory
    if storage_factory is not None:
        _storage_factory = storage_factory
    if pipeline_factory is not None:
        _pipeline_factory = pipeline_factory
    if store_factory is not None:
        _store_factory = store_factory


def reset_dispatcher_dependencies() -> None:
    """Clear all configured dependencies. For test isolation only —
    production code should never call this once the worker is wired."""
    global _storage_factory, _pipeline_factory, _store_factory
    _storage_factory = None
    _pipeline_factory = None
    _store_factory = None


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


def _get_store() -> "JobStore":
    if _store_factory is None:
        raise RuntimeError(
            "staging_dispatcher: store_factory not configured. Call "
            "configure_dispatcher_dependencies(store_factory=...) in "
            "the worker bootstrap or test setup."
        )
    return _store_factory()


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
# generate_project_dispatcher (issue 005)
# ---------------------------------------------------------------------------


async def generate_project_dispatcher(
    job: JobDoc,
    is_cancelled: CancelCheck,
) -> DispatchResult:
    """Run a queued project-level generation job to completion.

    End-to-end flow (mirrors PRD § Worker dispatcher (project-kind branch)):

    1. **Acquire the distributed per-project lease** (issue 002). If a
       different non-terminal job already holds it, poll-acquire with
       ``LEASE_POLL_INTERVAL_SECONDS`` cadence until the visibility
       heartbeat (issue 001) or ``LEASE_MAX_WAIT_SECONDS`` resolves.
       ``is_cancelled()`` is honored during the wait so a user-cancel
       drops out cleanly. Timeout → :class:`LeaseBusy` (JobWorker
       abandons → Storage Queue redelivers).
    2. **Hydrate the project AFTER** acquiring the lease — eliminates
       the rubber-duck-flagged stale-snapshot race where Job B could
       read pre-A-completion data, lose the acquire race, then later
       overwrite A's finished work with its stale in-memory rooms.
    3. **regenerate_all=true** clears ``image_url`` / ``thumb_url`` /
       ``md_url`` on every variation, schedules deletion of the
       previously-pointed blobs (mirrors single-variation regen
       cleanup, applied in bulk), resets variation + room status to
       PENDING, and persists. A pre-reset ``is_cancelled()`` poll
       guards against destroying completed work just before a user
       cancel.
    4. **Brief reuse on retry**: ``payload["brief_prompts"]`` is passed
       through verbatim to ``generate_project_for_job`` so a
       redelivery does NOT re-run the LLM compose pass. ``None`` is
       passed through as-is (the pipeline owns the
       None→legacy-compute fallback).
    5. **Progress callback** writes to ``JobStore.update_job(phase=)``
       per event so progress flows through the change feed and reaches
       SSE consumers via ``/jobs/stream``. (Synthetic-progress
       percentages are owned by the JobWorker estimator heartbeat.)
    6. **Cancel mid-flight**: when ``generate_project_for_job`` raises
       :class:`JobCancelled` (its ``is_cancelled()`` poll fired), revert
       PROCESSING variations → PENDING, preserve COMPLETED variations
       (cancel-during-image-edit edge), revert rooms whose
       ``_room_worker`` exception handler set ``status=FAILED`` +
       ``error="cancelled"`` back to PENDING, recompute project
       status, persist, then re-raise so the worker routes the
       message to ``complete`` (drop).
    7. **Lease release in finally** — only if we actually acquired it
       (the pre-acquire ``JobCancelled``/``LeaseBusy`` paths must NOT
       try to release a lease they don't own).

    Cascade-cancel of in-flight ``regenerate_variation`` jobs is the
    POST handler's responsibility (issue 006), NOT this dispatcher's.
    """
    # Late imports mirror the existing dispatcher pattern: the project-
    # lease helpers are pure functions but kept lazy so test fixtures
    # that ``reset_dispatcher_dependencies`` between tests don't pull
    # in pydantic / Cosmos at module import time.
    from backend.core.job_worker import JobCancelled
    from backend.core.project_lease import (
        acquire_project_lease,
        release_project_lease,
    )
    from backend.core.project_status import ProjectStatusCalculator
    from backend.models.staging import ItemStatus, StagingProject

    storage = _get_storage()
    pipeline = _get_pipeline()
    store = _get_store()

    project_id = job["project_id"]
    job_id = job["id"]
    payload = job.get("payload") or {}

    # 1) Poll-acquire the project lease. The visibility heartbeat
    # (issue 001) extends the queue lease in the background while we
    # wait, so a multi-minute hold is safe. Honor is_cancelled() so a
    # user-cancel drops out instead of waiting out the full timeout.
    started = time.monotonic()
    lease_acquired = False
    while True:
        if acquire_project_lease(
            storage=storage,
            store=store,
            project_id=project_id,
            job_id=job_id,
        ):
            lease_acquired = True
            break
        if time.monotonic() - started >= LEASE_MAX_WAIT_SECONDS:
            raise LeaseBusy(
                f"timed out after {LEASE_MAX_WAIT_SECONDS:.0f}s waiting for "
                f"project lease (project={project_id}, job={job_id})"
            )
        if is_cancelled():
            raise JobCancelled(
                f"cancel_requested observed while waiting for project "
                f"lease (project={project_id}, job={job_id})"
            )
        await asyncio.sleep(LEASE_POLL_INTERVAL_SECONDS)

    try:
        # 2) Hydrate AFTER acquire (eliminates stale-snapshot race).
        project_data = storage.get_project(project_id)
        if not project_data:
            raise ValueError(f"Project not found: {project_id}")
        # Strip Cosmos internals before pydantic constructs the model
        # — same cleanup the variation dispatcher does (line 163).
        clean = {
            k: v
            for k, v in project_data.items()
            if k != "doc_type" and not k.startswith("_")
        }
        project = StagingProject(**clean)

        # 3) regenerate_all=true: pre-reset cancel poll, then clear
        # blob URLs, schedule deletion, reset statuses, persist.
        if payload.get("regenerate_all"):
            if is_cancelled():
                raise JobCancelled(
                    f"cancel_requested observed before destructive "
                    f"regenerate_all reset (project={project_id})"
                )
            for room in project.rooms:
                for variation in room.variations:
                    for url in (
                        variation.image_url,
                        variation.thumb_url,
                        variation.md_url,
                    ):
                        if url:
                            pipeline._schedule_blob_cleanup(url)
                    variation.image_url = None
                    variation.thumb_url = None
                    variation.md_url = None
                    variation.status = ItemStatus.PENDING
                    variation.error = None
                room.status = ItemStatus.PENDING
                room.error = None
            await pipeline._persist_project_locked(project)

        # 4) Progress callback maps each event['type'] to a JobStore
        # phase write. Job worker's synthetic-progress estimator
        # owns the ``progress`` percentage independently.
        def progress_callback(event: dict) -> None:
            phase = event.get("type")
            if phase:
                store.update_job(job_id, project_id, phase=phase)

        # 5) Run the pipeline. brief_prompts is passed verbatim — None
        # is the legitimate "no pre-computed prompts; pipeline falls
        # back to compute or empty" signal.
        try:
            return await pipeline.generate_project_for_job(
                project,
                brief_prompts=payload.get("brief_prompts"),
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )
        except JobCancelled:
            # 6) Cleanup: revert PROCESSING variations to PENDING,
            # preserve COMPLETED, revert rooms cancel-failed by the
            # _room_worker exception handler. Do NOT mark anything
            # FAILED (PRD constraint).
            for room in project.rooms:
                for variation in room.variations:
                    if variation.status == ItemStatus.PROCESSING:
                        variation.status = ItemStatus.PENDING
                        variation.error = None
                if (
                    room.status == ItemStatus.FAILED
                    and room.error == "cancelled"
                ):
                    room.status = ItemStatus.PENDING
                    room.error = None
            project.status = ProjectStatusCalculator.compute_status(project.rooms)
            await pipeline._persist_project_locked(project)
            raise
    finally:
        # 7) Release the lease — only if we actually acquired it.
        # The pre-acquire JobCancelled/LeaseBusy paths skip release
        # because they never owned the lease (foreign holder still
        # has it).
        if lease_acquired:
            release_project_lease(
                storage=storage,
                project_id=project_id,
                job_id=job_id,
            )


# ---------------------------------------------------------------------------
# Top-level kind-switch
# ---------------------------------------------------------------------------


async def staging_dispatcher(
    job: JobDoc,
    is_cancelled: CancelCheck,
) -> DispatchResult:
    """Top-level kind-switch dispatcher for the ``JobWorker``.

    Routes by ``job["kind"]``. Unknown kinds raise ``ValueError`` so
    the worker's normal failure path runs and the job is marked
    failed rather than silently no-op'd.
    """
    kind = job.get("kind")
    if kind == "regenerate_variation":
        return await regenerate_variation_dispatcher(job, is_cancelled)
    if kind == "generate_project":
        return await generate_project_dispatcher(job, is_cancelled)
    raise ValueError(f"Unknown kind: {kind}")


__all__ = [
    "configure_dispatcher_dependencies",
    "reset_dispatcher_dependencies",
    "regenerate_variation_dispatcher",
    "generate_project_dispatcher",
    "staging_dispatcher",
    "LeaseBusy",
    "LEASE_POLL_INTERVAL_SECONDS",
    "LEASE_MAX_WAIT_SECONDS",
]
