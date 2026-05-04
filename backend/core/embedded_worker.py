"""``backend.core.embedded_worker`` — auto-start a JobWorker inside the API process.

Production runs the worker as a separate Container App
(``ROLE=worker`` + ``python -m backend.worker_main``). In **development**
(``uv run fastapi dev``) we don't want the user to also need to start a
second process — the queue would silently fill up and the user would
see "0 running" with no explanation.

This module owns the **policy decision** (auto-start when
``AUTO_START_WORKER`` is true and ``ROLE != "worker"``), the
**asyncio task handle**, the ``[embedded-worker]`` log prefix, and the
**clean shutdown sequence** (signal stop → await with timeout →
force-cancel + warn).

Public surface:

  - ``LOG_PREFIX`` — every log line emitted from this module starts
    with this exact string so an ops grep for ``[embedded-worker]``
    catches both start/skip decisions and shutdown warnings.

  - ``should_auto_start(role, env_flag) -> bool`` — pure policy. Pinned
    by a parametrised truth table in ``test_embedded_worker.py``.

  - ``EmbeddedWorker`` — lifecycle manager. ``start(builder)`` and
    ``stop()`` are async; both are idempotent.

The FastAPI lifespan handler in ``backend/main.py`` instantiates an
``EmbeddedWorker`` per app and delegates to ``start`` on startup and
``stop`` on shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Optional

from backend.core.config import settings
from backend.core.job_worker import JobWorker

logger = logging.getLogger(__name__)


LOG_PREFIX = "[embedded-worker]"

# Default ceiling on how long ``stop()`` will wait for the worker task
# to exit gracefully before force-cancelling it. The standalone
# JobWorker only checks ``self._stop`` between dispatch calls, so a
# slow dispatch can keep ``run()`` alive briefly past stop(). 10s is
# generous enough for the typical between-message idle wait without
# wedging FastAPI lifespan teardown.
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def should_auto_start(role: Optional[str], env_flag: bool) -> bool:
    """Pure policy: should this process auto-start an embedded worker?

    ``role`` is the value of ``os.environ.get("ROLE")`` — None when
    unset, "" when set-but-empty (Container Apps occasionally surfaces
    unset env vars as ""), or any string. Strict equality with
    ``"worker"`` is deliberate so a typo (``"workers"``, ``"WORKER"``)
    can't accidentally suppress auto-start in dev.

    ``env_flag`` is the boolean from ``settings.AUTO_START_WORKER``.
    Off by default in production (set explicitly to False in the API
    container's bicep env block); on by default in dev.
    """
    return env_flag and role != "worker"


class EmbeddedWorker:
    """Lifecycle manager for an in-process JobWorker.

    Construct one per FastAPI app (caller is responsible for stashing
    it on ``app.state``). ``start`` and ``stop`` are both async.

    State machine:
      created  --start(skip)-->  no-op (worker=None, task=None)
      created  --start(go)---->  running (worker=W, task=T)
      running  --stop()------->  stopped (task.done())
    """

    def __init__(
        self,
        *,
        role: Optional[str] = None,
        env_flag: Optional[bool] = None,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        # When the caller doesn't explicitly pass role/env_flag, read
        # them from the live process environment + settings. Tests
        # pin both arguments explicitly so policy is deterministic and
        # independent of os.environ contamination.
        self._role = role if role is not None else os.environ.get("ROLE")
        self._env_flag = env_flag if env_flag is not None else settings.AUTO_START_WORKER
        self._shutdown_timeout = shutdown_timeout

        self.worker: Optional[JobWorker] = None
        self.task: Optional[asyncio.Task] = None

    async def start(self, builder: Callable[[], JobWorker]) -> None:
        """Construct a worker via ``builder`` and spawn its run loop.

        Skip path (policy returns False): logs the decision and returns
        without calling ``builder`` — important because builder
        construction touches Cosmos / Storage Queues / Foundry, which
        is exactly what we don't want to do in the worker container.

        Happy path: ``builder()`` constructs the JobWorker, then we
        schedule ``worker.run()`` as a task. A done-callback logs any
        unexpected exit so the API doesn't silently serve traffic with
        a dead worker.
        """
        if not should_auto_start(self._role, self._env_flag):
            logger.info(
                "%s auto-start disabled (role=%r, auto_start=%s)",
                LOG_PREFIX,
                self._role,
                self._env_flag,
            )
            return

        logger.info(
            "%s starting embedded JobWorker (role=%r, auto_start=%s)",
            LOG_PREFIX,
            self._role,
            self._env_flag,
        )
        worker = builder()
        self.worker = worker
        # ``asyncio.create_task`` schedules the coroutine immediately;
        # the first poll happens on the next event-loop yield from the
        # caller. Naming the task aids ``asyncio.all_tasks()`` triage.
        self.task = asyncio.create_task(worker.run(), name="embedded-worker-run")
        self.task.add_done_callback(self._on_task_done)
        logger.info("%s embedded JobWorker started", LOG_PREFIX)

    async def stop(self) -> None:
        """Stop the worker and wait for the task to exit.

        Idempotent: returns immediately when ``start`` was a skip path
        (no worker, no task) or when ``stop`` has already cleaned up.

        Graceful path: ``worker.stop()`` flips the worker's internal
        stop event; the next idle tick of ``run()`` exits and the task
        completes.

        Stuck path: if the task doesn't exit within
        ``self._shutdown_timeout``, we force-cancel and warn. The
        queue's visibility-timeout re-delivers any in-flight message
        cancelled mid-dispatch, so cancellation is recoverable.
        """
        if self.worker is None or self.task is None:
            return

        if self.task.done():
            # Task already exited (e.g. via the unexpected-exit path
            # logged by ``_on_task_done``). Nothing left to do.
            return

        logger.info("%s stopping embedded JobWorker", LOG_PREFIX)
        try:
            self.worker.stop()
        except Exception:  # noqa: BLE001 — defensive
            logger.exception("%s worker.stop() raised", LOG_PREFIX)

        try:
            await asyncio.wait_for(self.task, timeout=self._shutdown_timeout)
            logger.info("%s embedded JobWorker stopped cleanly", LOG_PREFIX)
        except asyncio.TimeoutError:
            logger.warning(
                "%s embedded JobWorker did not stop within %.1fs; cancelling task",
                LOG_PREFIX,
                self._shutdown_timeout,
            )
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                # Swallow both cancellation and any post-cancel
                # exception — we're shutting down.
                pass

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Done-callback: log unexpected task exits.

        Cancellation is expected (graceful + forced shutdown both end
        in cancellation when the task is mid-await). Clean completion
        is also expected (worker.stop() → run() returns). Any other
        exception means the worker died on us and the API now has no
        consumer — make that visible.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        logger.error(
            "%s embedded JobWorker task exited unexpectedly: %r",
            LOG_PREFIX,
            exc,
        )
