"""Unit tests for ``backend.core.job_worker.JobWorker``.

Public-interface contract pinned by these tests (per PRD § JobWorker +
issue 003 AC):

  - ``process_one()`` runs the full state machine for one queued
    message: dequeue → mark running → dispatch → mark terminal →
    complete or abandon. Returns True if a message was processed,
    False if the queue was empty.

  - On the happy path the job transitions ``pending → running →
    succeeded`` with ``progress=100`` and ``result`` populated; the
    queue message is ``complete``-d.

  - On dispatcher exception, the job's ``error`` field is populated
    and the queue message is ``abandon``-ed. Status stays ``pending``
    (re-runnable) until the final allowed attempt; on the final
    allowed attempt (``dequeue_count >= MAX_DEQUEUE_COUNT``) status
    becomes ``failed`` (terminal) AND the message is abandoned (which
    routes to the poison queue per ``JobQueue`` policy).

  - If ``cancel_requested`` is True before dispatch, the worker
    transitions the job to ``cancelled`` and ``complete``-s the
    message (does NOT abandon — cancellation is a terminal user
    intent, not a failure to retry).

  - The dispatcher receives a synchronous ``is_cancelled`` callback
    so it can poll between external calls. If the callback returns
    True mid-dispatch and the dispatcher raises ``JobCancelled``, the
    worker treats it as a cancel (not a failure).

  - ``attempts`` is incremented on every dispatch (including
    cancellations and successes — it counts attempts regardless of
    outcome).

  - When the queue is empty, ``process_one`` returns False without
    touching the store.

Tests use ``unittest.mock`` (sync + async) only. No real Cosmos / Azurite.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    job_id: str = "p1:r1:v1:0",
    project_id: str = "p1",
    dequeue_count: int = 1,
):
    from backend.core.job_queue import JobMessage

    return JobMessage(
        job_id=job_id,
        project_id=project_id,
        dequeue_count=dequeue_count,
        raw=MagicMock(),
    )


def _make_job_doc(
    job_id: str = "p1:r1:v1:0",
    project_id: str = "p1",
    status: str = "pending",
    attempts: int = 0,
    cancel_requested: bool = False,
    payload: dict | None = None,
):
    return {
        "id": job_id,
        "project_id": project_id,
        "room_id": "r1",
        "variation_id": "v1",
        "revision": 0,
        "kind": "regenerate_variation",
        "status": status,
        "progress": 0,
        "phase": None,
        "attempts": attempts,
        "payload": payload or {"prompt": "x"},
        "result": None,
        "error": None,
        "cancel_requested": cancel_requested,
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:00:00Z",
    }


def _make_worker(
    *,
    queue=None,
    store=None,
    dispatcher=None,
):
    """Build a JobWorker with mocks. ``dispatcher`` may be a coroutine fn."""
    from backend.core.job_worker import JobWorker

    queue = queue or MagicMock()
    store = store or MagicMock()
    if dispatcher is None:

        async def dispatcher(job, is_cancelled):  # pragma: no cover — overridden
            return {"ok": True}

    return JobWorker(queue=queue, store=store, dispatcher=dispatcher)


# ---------------------------------------------------------------------------
# Empty queue
# ---------------------------------------------------------------------------


def test_process_one_returns_false_when_queue_empty():
    queue = MagicMock()
    queue.dequeue.return_value = None
    store = MagicMock()
    worker = _make_worker(queue=queue, store=store)

    result = asyncio.run(worker.process_one())

    assert result is False
    store.get_job.assert_not_called()
    store.update_job.assert_not_called()
    queue.complete.assert_not_called()
    queue.abandon.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_process_one_happy_path_transitions_running_then_succeeded_and_completes():
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    seen_status_when_dispatched = []

    async def dispatcher(job, is_cancelled):
        # The dispatcher should see the running-state job, with attempts
        # already bumped — so it can log "job started".
        seen_status_when_dispatched.append(job["status"])
        return {"image_url": "https://example/blob.png"}

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)

    result = asyncio.run(worker.process_one())

    assert result is True
    # First update: pending → running, attempts incremented.
    # Final update: status=succeeded, progress=100, result populated.
    update_calls = store.update_job.call_args_list
    assert len(update_calls) >= 2, f"expected ≥2 updates (running, succeeded), got {update_calls}"

    first = update_calls[0].kwargs
    assert first.get("status") == "running"
    assert first.get("attempts") == 1, "attempts must be incremented before dispatch"

    last = update_calls[-1].kwargs
    assert last.get("status") == "succeeded"
    assert last.get("progress") == 100
    assert last.get("result") == {"image_url": "https://example/blob.png"}

    # Dispatcher saw the running state.
    assert seen_status_when_dispatched == ["running"]

    # Queue message completed (not abandoned) on success.
    queue.complete.assert_called_once_with(msg)
    queue.abandon.assert_not_called()


# ---------------------------------------------------------------------------
# Failure path — non-final attempt
# ---------------------------------------------------------------------------


def test_process_one_failure_under_max_dequeue_keeps_status_pending_and_abandons():
    msg = _make_message(dequeue_count=1)  # first attempt
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        raise RuntimeError("transient downstream blip")

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    update_calls = store.update_job.call_args_list
    final = update_calls[-1].kwargs
    # NOT terminal — message will be re-delivered, job should stay
    # eligible for re-running.
    assert final.get("status") == "pending", (
        "non-final failure should not mark job terminal — message will be redelivered"
    )
    # Structured error captured.
    error = final.get("error")
    assert error is not None
    assert "transient downstream blip" in str(error)

    # Abandoned, not completed.
    queue.abandon.assert_called_once_with(msg)
    queue.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Failure path — final attempt (poison)
# ---------------------------------------------------------------------------


def test_process_one_failure_at_max_dequeue_marks_failed_and_abandons():
    """On the 3rd (final) attempt that fails, the worker marks the job
    'failed' (terminal) AND calls abandon — JobQueue's abandon() routes
    to poison when dequeue_count >= MAX_DEQUEUE_COUNT, so the message
    is removed from the main queue AND the job's terminal state is
    persisted for SSE consumers.
    """
    from backend.core.job_queue import MAX_DEQUEUE_COUNT

    msg = _make_message(dequeue_count=MAX_DEQUEUE_COUNT)  # final attempt
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc(attempts=MAX_DEQUEUE_COUNT - 1)

    async def dispatcher(job, is_cancelled):
        raise RuntimeError("downstream really down")

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    update_calls = store.update_job.call_args_list
    final = update_calls[-1].kwargs
    assert final.get("status") == "failed", (
        "final-attempt failure must mark job terminal so SSE clients stop waiting"
    )
    error = final.get("error")
    assert error is not None
    assert "downstream really down" in str(error)
    # Issue 002 (active-and-queued PRD): the worker's terminal-failure
    # path classifies the exception via ``backend.core.job_errors``
    # and persists the resulting kind on the job doc. A bare
    # RuntimeError is classified UNKNOWN.
    assert final.get("error_kind") == "UNKNOWN", (
        "issue 002: terminal failures must persist error_kind so the "
        "front-end can surface kind-specific recovery messages"
    )

    # Abandon (not complete) — JobQueue.abandon will route to poison.
    queue.abandon.assert_called_once_with(msg)
    queue.complete.assert_not_called()


def test_process_one_failure_persists_error_kind_for_classified_exception():
    """Issue 002: when a worker dispatch raises a CLASSIFIED exception
    (e.g. ``CosmosHttpResponseError``), the job_errors classifier maps
    it to ``STORE_FAILED`` and the worker persists that kind on the
    job doc so the front-end can render a 'database hiccup' message
    instead of generic 'something went wrong'.
    """
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from backend.core.job_queue import MAX_DEQUEUE_COUNT

    msg = _make_message(dequeue_count=MAX_DEQUEUE_COUNT)
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc(attempts=MAX_DEQUEUE_COUNT - 1)

    async def dispatcher(job, is_cancelled):
        raise CosmosHttpResponseError(message="cosmos timeout")

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    final = store.update_job.call_args_list[-1].kwargs
    assert final.get("status") == "failed"
    assert final.get("error_kind") == "STORE_FAILED", (
        "CosmosHttpResponseError must classify as STORE_FAILED — see "
        "backend.core.job_errors.classify"
    )


# ---------------------------------------------------------------------------
# Cancellation — observed before dispatch
# ---------------------------------------------------------------------------


def test_process_one_cancel_requested_before_dispatch_skips_dispatch_and_completes():
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc(cancel_requested=True)

    dispatch_called = []

    async def dispatcher(job, is_cancelled):
        dispatch_called.append(True)
        return {}

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    assert dispatch_called == [], "must not dispatch when cancel_requested before pickup"

    update_calls = store.update_job.call_args_list
    assert len(update_calls) >= 1
    final = update_calls[-1].kwargs
    assert final.get("status") == "cancelled"

    # Cancellation is terminal user intent → drop the queue message.
    queue.complete.assert_called_once_with(msg)
    queue.abandon.assert_not_called()


# ---------------------------------------------------------------------------
# Cancellation — observed mid-dispatch (cooperative)
# ---------------------------------------------------------------------------


def test_process_one_cancel_raised_during_dispatch_completes_as_cancelled():
    """Dispatcher polls is_cancelled() between external calls. When the
    callback returns True it raises JobCancelled — the worker treats
    that as a cancel (terminal, message completed), not a failure.
    """
    from backend.core.job_worker import JobCancelled

    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    # First read returns the doc with cancel_requested=False (so dispatch
    # starts); the dispatcher itself will re-check via the live store.
    initial_doc = _make_job_doc(cancel_requested=False)
    cancelled_doc = _make_job_doc(cancel_requested=True)
    # get_job called: (1) at pickup, (2) by is_cancelled() probe inside dispatcher.
    store = MagicMock()
    store.get_job.side_effect = [initial_doc, cancelled_doc, cancelled_doc]

    async def dispatcher(job, is_cancelled):
        if is_cancelled():
            raise JobCancelled()
        return {}

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    final = store.update_job.call_args_list[-1].kwargs
    assert final.get("status") == "cancelled"
    queue.complete.assert_called_once_with(msg)
    queue.abandon.assert_not_called()


# ---------------------------------------------------------------------------
# Missing job (race: queue has a stale pointer to a deleted job doc)
# ---------------------------------------------------------------------------


def test_process_one_missing_job_doc_completes_message_without_dispatch():
    """If the queue carries a pointer to a job that no longer exists in
    Cosmos (e.g. project was deleted and its docs purged), the worker
    must NOT loop forever retrying — it drops the message.
    """
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = None  # not found

    dispatched = []

    async def dispatcher(job, is_cancelled):
        dispatched.append(True)
        return {}

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    assert dispatched == []
    store.update_job.assert_not_called()
    queue.complete.assert_called_once_with(msg)
    queue.abandon.assert_not_called()


# ---------------------------------------------------------------------------
# run() loop honors stop()
# ---------------------------------------------------------------------------


def test_run_loop_stops_on_stop_call():
    """``run()`` polls until ``stop()`` is called. With an empty queue
    and a stop set immediately, run() must exit promptly.
    """
    queue = MagicMock()
    queue.dequeue.return_value = None
    store = MagicMock()

    worker = _make_worker(queue=queue, store=store)

    async def go():
        worker.stop()
        await asyncio.wait_for(worker.run(), timeout=2.0)

    asyncio.run(go())  # must not raise / hang


def test_run_loop_processes_messages_until_stop():
    """run() drains the queue, then idles. We simulate two messages then
    empty + stop, and assert both were processed.
    """
    msg1 = _make_message(job_id="p1:r1:v1:0")
    msg2 = _make_message(job_id="p1:r1:v2:0")
    # Sequence: msg1, msg2, then None forever.
    _drain = [msg1, msg2]
    queue = MagicMock()
    queue.dequeue.side_effect = lambda *a, **kw: _drain.pop(0) if _drain else None
    store = MagicMock()
    # Echo back a doc keyed off the requested job_id so the dispatcher
    # can distinguish the two messages.
    store.get_job.side_effect = lambda jid, pid: _make_job_doc(job_id=jid, project_id=pid)

    processed = []

    async def dispatcher(job, is_cancelled):
        processed.append(job["id"])
        return {}

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    # Override idle sleep to keep test fast.
    worker._idle_sleep = 0.01

    async def go():
        # Stop after a short delay — gives the loop time to drain.
        async def stopper():
            await asyncio.sleep(0.05)
            worker.stop()

        await asyncio.gather(worker.run(), stopper())

    asyncio.run(asyncio.wait_for(go(), timeout=2.0))

    assert "p1:r1:v1:0" in processed
    assert "p1:r1:v2:0" in processed


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def test_happy_path_emits_started_and_succeeded_log_events(caplog):
    import logging as _logging

    caplog.set_level(_logging.INFO, logger="backend.core.job_worker")

    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        return {"ok": True}

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "job.started" in msgs
    assert "job.succeeded" in msgs


def test_failure_emits_failed_log_event(caplog):
    import logging as _logging

    caplog.set_level(_logging.INFO, logger="backend.core.job_worker")

    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg
    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        raise RuntimeError("boom")

    worker = _make_worker(queue=queue, store=store, dispatcher=dispatcher)
    asyncio.run(worker.process_one())

    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "job.failed" in msgs
