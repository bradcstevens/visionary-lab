"""Integration tests for the ProgressEstimator → JobWorker wiring.

Pinned by issue 008 of the image-pipeline-and-project-ux-overhaul PRD:
the worker writes ``phase`` + ``progress`` updates while a job runs,
and calls ``record_completion`` after a successful dispatch so the p50
stats doc is seeded by live traffic only.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from backend.core.job_worker import JobWorker
from backend.core.progress_estimator import (
    GENERATING_FLOOR,
    ProgressEstimator,
)


def _make_message(job_id="p1:r1:v1:0", project_id="p1", dequeue_count=1):
    from backend.core.job_queue import JobMessage

    return JobMessage(
        job_id=job_id,
        project_id=project_id,
        dequeue_count=dequeue_count,
        raw=MagicMock(),
    )


def _make_job_doc(payload=None, kind="regenerate_variation"):
    return {
        "id": "p1:r1:v1:0",
        "project_id": "p1",
        "room_id": "r1",
        "variation_id": "v1",
        "revision": 0,
        "kind": kind,
        "status": "pending",
        "progress": 0,
        "phase": None,
        "attempts": 0,
        "payload": payload or {"prompt": "x"},
        "result": None,
        "error": None,
        "cancel_requested": False,
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# Heartbeat emits progress while dispatcher runs
# ---------------------------------------------------------------------------


def test_heartbeat_emits_running_progress_while_dispatcher_runs():
    """While the dispatcher is awaiting, the worker should write at
    least one ``status=running`` doc patch with ``phase=generating``
    and a non-zero ``progress`` value.
    """
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    estimator = ProgressEstimator(container=False, default_p50_seconds=2.0)

    async def slow_dispatcher(job, is_cancelled):
        # Long enough to see multiple heartbeat ticks.
        await asyncio.sleep(0.25)
        return {"image_url": "x"}

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=slow_dispatcher,
        estimator=estimator,
        progress_interval=0.05,
    )

    asyncio.run(worker.process_one())

    # Inspect every update_job call. We want at least one with
    # status=running + phase=generating + progress in the generating
    # window. The pre-dispatch attempts++ write also has status=running
    # but no phase/progress, so filter on phase.
    running_writes = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("phase") == "generating"
    ]
    assert running_writes, "expected at least one heartbeat write"
    for call in running_writes:
        assert call.kwargs["status"] == "running"
        assert call.kwargs["progress"] >= GENERATING_FLOOR
        assert call.kwargs["progress"] < 90  # never reach finalizing


def test_heartbeat_progress_is_monotonic_across_ticks():
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    estimator = ProgressEstimator(container=False, default_p50_seconds=1.0)

    async def slow_dispatcher(job, is_cancelled):
        await asyncio.sleep(0.4)
        return {"image_url": "x"}

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=slow_dispatcher,
        estimator=estimator,
        progress_interval=0.05,
    )

    asyncio.run(worker.process_one())

    progresses = [
        c.kwargs["progress"]
        for c in store.update_job.call_args_list
        if c.kwargs.get("phase") == "generating"
    ]
    assert progresses, "expected heartbeat writes"
    assert progresses == sorted(progresses), (
        f"progress sequence not monotonic: {progresses}"
    )


def test_heartbeat_cancelled_after_dispatcher_returns():
    """No heartbeat writes should be issued AFTER the success terminal
    write — the terminal must be the last update_job call.
    """
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    estimator = ProgressEstimator(container=False, default_p50_seconds=2.0)

    async def quick(job, is_cancelled):
        await asyncio.sleep(0.15)
        return {"image_url": "x"}

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=quick,
        estimator=estimator,
        progress_interval=0.05,
    )

    asyncio.run(worker.process_one())

    # Last update_job MUST be the terminal succeeded write.
    last = store.update_job.call_args_list[-1]
    assert last.kwargs["status"] == "succeeded"
    assert last.kwargs["progress"] == 100
    assert last.kwargs["phase"] == "finalizing"


def test_heartbeat_skipped_when_no_estimator_configured():
    """Back-compat: a worker built without an estimator must not
    introduce any new update_job calls — existing behavior preserved.
    """
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def quick(job, is_cancelled):
        await asyncio.sleep(0.05)
        return {"image_url": "x"}

    worker = JobWorker(queue=queue, store=store, dispatcher=quick)

    asyncio.run(worker.process_one())

    # Without an estimator, only the running-attempt-bump and the
    # terminal succeeded write happen — exactly two update_job calls.
    assert store.update_job.call_count == 2
    phases = [c.kwargs.get("phase") for c in store.update_job.call_args_list]
    assert phases == [None, "finalizing"], phases


# ---------------------------------------------------------------------------
# record_completion: called on success only
# ---------------------------------------------------------------------------


def test_record_completion_called_after_successful_dispatch():
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()
    store = MagicMock()
    store.get_job.return_value = _make_job_doc(
        payload={"model": "gpt-image-2", "prompt": "x"},
        kind="regenerate_variation",
    )

    estimator = MagicMock()
    estimator.estimate.return_value = ("generating", 25)

    async def quick(job, is_cancelled):
        return {"image_url": "x"}

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=quick,
        estimator=estimator,
        progress_interval=10.0,  # too slow to tick during the test
    )

    asyncio.run(worker.process_one())

    estimator.record_completion.assert_called_once()
    kwargs = estimator.record_completion.call_args.kwargs
    assert kwargs["model"] == "gpt-image-2"
    assert kwargs["kind"] == "regenerate_variation"
    assert kwargs["elapsed_seconds"] >= 0


def test_record_completion_not_called_on_failure():
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    estimator = MagicMock()

    async def boom(job, is_cancelled):
        raise RuntimeError("upstream borked")

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=boom,
        estimator=estimator,
        progress_interval=10.0,
    )

    asyncio.run(worker.process_one())

    estimator.record_completion.assert_not_called()


def test_record_completion_not_called_on_cancellation():
    from backend.core.job_worker import JobCancelled

    queue = MagicMock()
    queue.dequeue.return_value = _make_message()
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    estimator = MagicMock()

    async def cancel_mid(job, is_cancelled):
        raise JobCancelled()

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=cancel_mid,
        estimator=estimator,
        progress_interval=10.0,
    )

    asyncio.run(worker.process_one())

    estimator.record_completion.assert_not_called()


# ---------------------------------------------------------------------------
# Heartbeat resilience: store flap must not derail the dispatcher
# ---------------------------------------------------------------------------


def test_heartbeat_swallows_store_update_failures():
    """A flapping update_job during a heartbeat tick must NOT cause the
    job to fail. The dispatcher runs to completion, the success write
    still happens, and the queue message is still completed.
    """
    queue = MagicMock()
    queue.dequeue.return_value = _make_message()

    # The pre-dispatch attempts-bump (call #1) and the terminal
    # succeeded write (last) must succeed; only the heartbeat ticks
    # (calls in between with phase=generating) blow up.
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    def flaky_update(*args, **kwargs):
        if kwargs.get("phase") == "generating":
            raise RuntimeError("cosmos blip")
        return {}

    store.update_job.side_effect = flaky_update

    estimator = ProgressEstimator(container=False, default_p50_seconds=1.0)

    async def slow(job, is_cancelled):
        await asyncio.sleep(0.2)
        return {"image_url": "x"}

    worker = JobWorker(
        queue=queue,
        store=store,
        dispatcher=slow,
        estimator=estimator,
        progress_interval=0.05,
    )

    asyncio.run(worker.process_one())

    # Terminal write still happened.
    last = store.update_job.call_args_list[-1]
    assert last.kwargs["status"] == "succeeded"
    queue.complete.assert_called_once()
    queue.abandon.assert_not_called()
