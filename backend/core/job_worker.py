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
import logging
from typing import Any, Awaitable, Callable, Optional

from backend.core.job_queue import MAX_DEQUEUE_COUNT, JobMessage, JobQueue
from backend.core.job_store import JobStore

logger = logging.getLogger(__name__)


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
    ):
        self._queue = queue
        self._store = store
        self._dispatcher = dispatcher
        self._idle_sleep = idle_sleep
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
            self._queue.abandon(message)
            return

        # 5) Success.
        self._store.update_job(
            job_id,
            project_id,
            status="succeeded",
            progress=100,
            phase="finalizing",
            result=result,
        )
        logger.info(
            "job.succeeded job_id=%s project_id=%s attempt=%d",
            job_id,
            project_id,
            attempts,
        )
        self._queue.complete(message)
