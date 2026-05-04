"""``JobWorker`` — long-running consumer for the async image-job queue.

Runs in the worker Container App replicas (``ROLE=worker``, see
``infra/modules/containerAppWorker.bicep``). Pulls pointer messages
from ``JobQueue``, fetches the durable job state from ``JobStore``,
dispatches to a caller-supplied async ``dispatcher``, and persists
state transitions back to ``JobStore``. The ``dispatcher`` keeps the
worker decoupled from any specific pipeline — issue 004 wires it to
``ImagePipelineService``; tests inject a mock.

Public surface (per PRD § JobWorker + issue 003 AC):

  - ``process_one()`` — runs the full state machine for one queued
    message: dequeue → mark running (attempts++) → dispatch → mark
    terminal → complete or abandon. Returns True if a message was
    processed, False if the queue was empty.

  - ``run()`` — poll loop until ``stop()``.

  - ``stop()`` — signal the loop to exit at the next idle.

  - ``JobCancelled`` — sentinel exception the dispatcher raises when
    it observes ``is_cancelled()`` returning True. The worker treats
    that as a cancel (terminal, message completed), not a failure to
    retry.

State machine (per PRD § Schema):

  pending ──[pickup]──► running ──[ok]─────► succeeded   (complete)
                            │
                            ├──[exception, attempts < MAX]─► pending  (abandon → redeliver)
                            ├──[exception, attempts = MAX]─► failed   (abandon → poison)
                            └──[JobCancelled OR cancel_requested observed]─► cancelled (complete)

The ``IMAGE_GEN_MAX_CONCURRENT`` semaphore is acquired *inside*
``ImagePipelineService`` via ``call_with_retry`` (see
``backend/core/image_pipeline.py``), NOT here — there must be exactly
one cap, applied at the call site. The worker's per-replica cap on
in-flight messages is the queue's own ``max_messages=1`` plus its
visibility-timeout window; spawning concurrent ``process_one`` tasks
on a single replica is left as an issue-004+ concern.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from backend.core.config import settings
from backend.core.job_queue import (
    MAX_DEQUEUE_COUNT,
    VISIBILITY_TIMEOUT_SECONDS,
    JobMessage,
    JobQueue,
)
from backend.core.job_store import JobStore
from backend.core.progress_estimator import ProgressEstimator

logger = logging.getLogger(__name__)


# Heartbeat cadence for the visibility-timeout extension loop spawned
# by ``process_one`` while a dispatcher runs. Pinned by the PRD's
# "every 30 seconds" requirement; comfortably under the 90s queue
# visibility timeout so we get ~3 extensions per window.
HEARTBEAT_INTERVAL_SECONDS = 30.0


class JobCancelled(Exception):
    """Raised by the dispatcher when it observes ``is_cancelled()`` True.

    Distinguishes user-requested cancellation from a downstream failure
    so the worker can route the message to ``complete`` (drop it) rather
    than ``abandon`` (which would re-deliver and eventually poison).
    """


# Type alias for clarity. The dispatcher receives the live job doc and
# a synchronous ``is_cancelled`` callback it MUST poll between external
# calls. It returns the ``result`` dict to persist on success.
Dispatcher = Callable[[dict[str, Any], Callable[[], bool]], Awaitable[dict[str, Any]]]


class JobWorker:
    """Consumes pointer messages from ``JobQueue`` and runs them.

    Constructor injection of ``queue``, ``store``, and ``dispatcher``
    keeps this module testable in isolation per the PRD's "deep modules,
    testable in isolation" constraint.
    """

    def __init__(
        self,
        *,
        queue: JobQueue,
        store: JobStore,
        dispatcher: Dispatcher,
        idle_sleep: float = 1.0,
        estimator: Optional[ProgressEstimator] = None,
        progress_interval: float = 2.0,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    ):
        self._queue = queue
        self._store = store
        self._dispatcher = dispatcher
        self._idle_sleep = idle_sleep
        self._estimator = estimator
        self._progress_interval = progress_interval
        self._heartbeat_interval = heartbeat_interval
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal ``run()`` to exit at the next idle tick."""
        self._stop.set()

    async def run(self) -> None:
        """Poll the queue forever (or until ``stop()``).

        On an empty queue we sleep ``idle_sleep`` seconds before the
        next poll; on a successful pickup we loop again immediately to
        drain bursts. Replica termination handling lives in the
        Container App lifecycle (visibility-timeout re-delivery covers
        SIGTERM mid-flight — see DEPLOYMENT.md drain-window section).
        """
        while not self._stop.is_set():
            try:
                processed = await self.process_one()
            except Exception:  # noqa: BLE001 — never let the loop die
                logger.exception("job_worker.process_one crashed; continuing")
                processed = False
            if not processed:
                # Sleep but wake early if stop fires.
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._idle_sleep)
                except asyncio.TimeoutError:
                    pass

    # ------------------------------------------------------------------
    # Per-message state machine
    # ------------------------------------------------------------------

    async def process_one(self) -> bool:
        """Run one full state-machine pass. Returns True if a message
        was processed, False if the queue was empty.
        """
        message = self._queue.dequeue()
        if message is None:
            return False
        await self._handle(message)
        return True

    async def _handle(self, message: JobMessage) -> None:
        job_id = message.job_id
        project_id = message.project_id
        is_final_attempt = message.dequeue_count >= MAX_DEQUEUE_COUNT

        # 1) Resolve durable state. A missing doc means the project (or
        #    the job) was deleted out from under the queue; drop the
        #    pointer so we don't loop on it forever.
        job = self._store.get_job(job_id, project_id)
        if job is None:
            logger.warning(
                "job.missing job_id=%s project_id=%s — dropping stale pointer",
                job_id,
                project_id,
            )
            self._queue.complete(message)
            return

        # 2) Honor a pre-pickup cancel.
        if job.get("cancel_requested"):
            self._store.update_job(
                job_id, project_id, status="cancelled"
            )
            logger.info(
                "job.cancelled job_id=%s project_id=%s phase=pre-dispatch",
                job_id,
                project_id,
            )
            self._queue.complete(message)
            return

        # 3) Transition to running and bump attempts BEFORE dispatch so
        #    the doc reflects the in-flight state for SSE consumers.
        attempts = int(job.get("attempts", 0)) + 1
        self._store.update_job(
            job_id,
            project_id,
            status="running",
            attempts=attempts,
        )
        # In-memory mirror of the running state. We don't trust
        # update_job's return value to be a coherent dict (Cosmos
        # round-trip strips fields) — compose it locally so the
        # dispatcher sees a stable view.
        running = {**job, "status": "running", "attempts": attempts}
        logger.info(
            "job.started job_id=%s project_id=%s attempt=%d/%d",
            job_id,
            project_id,
            attempts,
            MAX_DEQUEUE_COUNT,
        )

        # 4) Dispatch with a live cancel probe. The dispatcher MUST call
        #    is_cancelled() between external calls and raise
        #    JobCancelled when it returns True.
        def is_cancelled() -> bool:
            current = self._store.get_job(job_id, project_id)
            return bool(current and current.get("cancel_requested"))

        # 4a) Synthetic-progress heartbeat (issue 008). When an
        # ``estimator`` is wired, spawn a background task that emits
        # ``phase`` + ``progress`` updates every ``progress_interval``
        # seconds while the dispatcher runs. Cancelled in finally so a
        # crashed dispatcher cannot leak the task. Without an estimator
        # the worker behaves exactly as before.
        started_monotonic = time.monotonic()
        model = (job.get("payload") or {}).get(
            "model"
        ) or getattr(settings, "DEFAULT_IMAGE_MODEL", "default")
        kind = job.get("kind") or "unknown"
        progress_heartbeat: Optional[asyncio.Task[None]] = None
        if self._estimator is not None:
            progress_heartbeat = asyncio.create_task(
                self._emit_progress(
                    job_id=job_id,
                    project_id=project_id,
                    model=model,
                    kind=kind,
                    started_monotonic=started_monotonic,
                )
            )

        # 4b) Visibility-timeout heartbeat (issue 001 of project-
        # generation-async-queue-cutover). UNCONDITIONAL — fast jobs
        # (~20s) cancel the task before its first wake; long-running
        # jobs (project generation, multi-minute) get their queue
        # visibility extended every ``heartbeat_interval`` seconds so
        # Storage Queue does NOT redeliver a successful run after the
        # 90s window. The shared ``message_lock`` serializes extend
        # against complete/abandon — without it, an in-flight extend
        # could race with the post-dispatch delete and either lose
        # the freshest pop_receipt or 404 on delete.
        message_lock = asyncio.Lock()
        visibility_heartbeat: asyncio.Task[None] = asyncio.create_task(
            self._extend_visibility_loop(message, message_lock)
        )

        try:
            try:
                result = await self._dispatcher(running, is_cancelled)
            except JobCancelled:
                self._store.update_job(
                    job_id, project_id, status="cancelled"
                )
                logger.info(
                    "job.cancelled job_id=%s project_id=%s phase=mid-dispatch",
                    job_id,
                    project_id,
                )
                async with message_lock:
                    self._queue.complete(message)
                return
            except Exception as exc:  # noqa: BLE001 — convert to durable error
                error_payload = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
                terminal = is_final_attempt
                self._store.update_job(
                    job_id,
                    project_id,
                    status=("failed" if terminal else "pending"),
                    error=error_payload,
                )
                logger.warning(
                    "job.failed job_id=%s project_id=%s attempt=%d terminal=%s error=%s",
                    job_id,
                    project_id,
                    attempts,
                    terminal,
                    error_payload,
                )
                async with message_lock:
                    self._queue.abandon(message)
                return
        finally:
            # Cancel both heartbeats. Awaiting cancellation guarantees
            # neither task is mid-flight when complete()/abandon() is
            # called below the finally — combined with the lock guard
            # this gives belt-and-suspenders safety against the
            # rubber-duck blocking #1 finding (stale pop_receipt → 404
            # on delete → message redelivered → pipeline runs twice).
            for task in (visibility_heartbeat, progress_heartbeat):
                if task is None:
                    continue
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        # 5) Success.
        self._store.update_job(
            job_id,
            project_id,
            status="succeeded",
            progress=100,
            phase="finalizing",
            result=result,
        )
        if self._estimator is not None:
            elapsed = max(0.0, time.monotonic() - started_monotonic)
            # Best-effort — record_completion swallows its own
            # exceptions, but guard the call too in case the estimator
            # reference is itself broken.
            try:
                self._estimator.record_completion(
                    model=model, kind=kind, elapsed_seconds=elapsed
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "job_worker.record_completion failed job_id=%s",
                    job_id,
                    exc_info=True,
                )
        logger.info(
            "job.succeeded job_id=%s project_id=%s attempt=%d",
            job_id,
            project_id,
            attempts,
        )
        async with message_lock:
            self._queue.complete(message)

    async def _extend_visibility_loop(
        self,
        message: JobMessage,
        lock: asyncio.Lock,
    ) -> None:
        """Periodically extend the queue message's visibility timeout.

        Spawned by ``_handle`` for every dispatched message. Each tick
        sleeps ``heartbeat_interval`` seconds, then calls
        ``JobQueue.extend_visibility`` under the shared ``lock`` so the
        extend cannot race with a concurrent ``complete()``/``abandon()``
        on the same message.

        Exit conditions:
          - ``CancelledError`` (the dispatcher returned/raised, finally
            cancels us). Re-raised so the awaiter's
            ``contextlib.suppress`` sees the cancel.
          - ``ResourceNotFoundError`` / ``HttpResponseError`` — the
            message has almost certainly been deleted out from under us
            (success path completed first, or a peer worker raced).
            Logged at debug; loop exits cleanly so the awaiting task
            terminates without bubbling.
          - Any other exception is logged and the loop continues — the
            heartbeat must NOT derail an in-flight long-running job
            because of a transient storage hiccup. The next tick will
            try again; if the storage is genuinely down, the queue's
            visibility-timeout safety net kicks in (redelivery), which
            is the same outcome we'd get without the heartbeat.
        """
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                raise
            try:
                async with lock:
                    self._queue.extend_visibility(
                        message, VISIBILITY_TIMEOUT_SECONDS
                    )
                logger.debug(
                    "heartbeat.extend job_id=%s",
                    message.job_id,
                )
            except asyncio.CancelledError:
                raise
            except (ResourceNotFoundError, HttpResponseError) as exc:
                logger.debug(
                    "heartbeat.skip job_id=%s reason=%s",
                    message.job_id,
                    exc.__class__.__name__,
                )
                return
            except Exception:  # noqa: BLE001 — never derail the worker
                logger.warning(
                    "heartbeat.extend_failed job_id=%s",
                    message.job_id,
                    exc_info=True,
                )

    async def _emit_progress(
        self,
        *,
        job_id: str,
        project_id: str,
        model: str,
        kind: str,
        started_monotonic: float,
    ) -> None:
        """Periodically write ``phase`` + ``progress`` while a job runs.

        Synthetic progress only — the estimator's curve approaches but
        never crosses the finalizing floor (90%). The terminal
        ``progress=100`` + ``phase=finalizing`` write happens in the
        success branch of ``_handle`` after the dispatcher returns.

        Best-effort: any exception from the estimator or the store is
        logged and the heartbeat continues. A flapping cosmos call
        must not derail an in-flight image-gen.
        """
        assert self._estimator is not None
        prior = 0
        while True:
            try:
                await asyncio.sleep(self._progress_interval)
                elapsed = time.monotonic() - started_monotonic
                phase, progress = self._estimator.estimate(
                    model=model,
                    kind=kind,
                    elapsed_seconds=elapsed,
                    prior_progress=prior,
                )
                if progress > prior:
                    prior = progress
                    self._store.update_job(
                        job_id,
                        project_id,
                        status="running",
                        phase=phase,
                        progress=progress,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — cosmetic, never fatal
                logger.warning(
                    "job_worker._emit_progress tick failed job_id=%s",
                    job_id,
                    exc_info=True,
                )
