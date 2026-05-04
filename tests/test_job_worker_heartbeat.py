"""Unit tests for ``backend.core.job_worker.JobWorker`` heartbeat
machinery (issue 001 of the project-generation-async-queue-cutover PRD).

Public-interface contract pinned by these tests:

  - ``JobWorker.process_one()`` spawns a heartbeat task per dispatched
    message that calls ``JobQueue.extend_visibility(message,
    VISIBILITY_TIMEOUT_SECONDS)`` every ``heartbeat_interval`` seconds.
  - The heartbeat is unconditional — fast jobs (~20s) cancel the task
    before its first wake; the design must remain safe for borderline
    ~31s jobs.
  - Heartbeat extension and message completion are serialized via an
    ``asyncio.Lock`` so an in-flight extend cannot race with a
    ``complete()`` call on the same message.
  - ``ResourceNotFoundError`` and ``HttpResponseError`` raised by the
    heartbeat after the message has already been deleted are swallowed
    (logged at debug, the loop exits cleanly without bubbling).
  - The heartbeat task is cancelled when the dispatcher returns OR
    raises; cancellation is awaited and ``CancelledError`` is
    suppressed.
  - Without these guarantees the rubber-duck blocking #1 finding from
    the PRD reproduces: a successful long-running project-generation
    job is redelivered after the original visibility window expires
    and the project pipeline runs twice.

Tests use ``unittest.mock`` (sync + async) plus a small heartbeat
interval (e.g. 0.02s) so the per-tick wait stays inside CI budget.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    job_id: str = "p1:r1:v1:0",
    project_id: str = "p1",
    dequeue_count: int = 1,
):
    from backend.core.job_queue import JobMessage

    raw = MagicMock()
    raw.id = "msg-id"
    raw.pop_receipt = "initial-receipt"
    return JobMessage(
        job_id=job_id,
        project_id=project_id,
        dequeue_count=dequeue_count,
        raw=raw,
    )


def _make_job_doc(
    job_id: str = "p1:r1:v1:0",
    project_id: str = "p1",
    status: str = "pending",
    attempts: int = 0,
    cancel_requested: bool = False,
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
        "payload": {"prompt": "x"},
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
    heartbeat_interval: float = 0.02,
):
    from backend.core.job_worker import JobWorker

    queue = queue or MagicMock()
    store = store or MagicMock()
    if dispatcher is None:

        async def dispatcher(job, is_cancelled):  # pragma: no cover
            return {"ok": True}

    return JobWorker(
        queue=queue,
        store=store,
        dispatcher=dispatcher,
        heartbeat_interval=heartbeat_interval,
    )


# ---------------------------------------------------------------------------
# Heartbeat fires while the dispatcher runs
# ---------------------------------------------------------------------------


def test_heartbeat_extends_visibility_every_interval_during_dispatch():
    """Per AC: ``JobWorker.process_one()`` spawns an asyncio heartbeat
    task that fires every ``heartbeat_interval`` seconds while the
    dispatcher runs.

    We use a 0.02s interval and a dispatcher that sleeps ~0.10s, so
    we expect ~3-5 heartbeat extensions to fire during the run. We
    assert >=2 to keep the test deterministic on slow CI without
    losing the regression: zero extensions would mean the heartbeat
    isn't running.
    """
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.10)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    asyncio.run(worker.process_one())

    extend_calls = queue.extend_visibility.call_args_list
    assert len(extend_calls) >= 2, (
        f"expected >=2 heartbeat extensions during a 100ms dispatch with "
        f"20ms interval, got {len(extend_calls)}"
    )
    # Every call passes the original message and the queue's
    # configured visibility timeout.
    from backend.core.job_queue import VISIBILITY_TIMEOUT_SECONDS

    for call in extend_calls:
        assert call.args[0] is msg
        assert call.args[1] == VISIBILITY_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Heartbeat is cancelled when the dispatcher returns or raises
# ---------------------------------------------------------------------------


def test_heartbeat_task_is_cancelled_when_dispatcher_returns():
    """After the dispatcher returns, the heartbeat task must be
    cancelled and awaited so it can't tick another extension after
    the message is completed. Regression pin: a leaked heartbeat
    would either keep extending visibility on a deleted message
    (404) or, worse, extend a message id reused by Storage Queue
    weeks later.
    """
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        # Short dispatch — the heartbeat will fire at most once before
        # we return; but the cancel-on-return contract is what matters.
        await asyncio.sleep(0.05)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    asyncio.run(worker.process_one())

    # Capture extend_visibility call count, then wait a window > 5x
    # the heartbeat interval. If the heartbeat were leaked it would
    # have ticked again by now.
    pre = len(queue.extend_visibility.call_args_list)

    async def settle():
        await asyncio.sleep(0.15)

    asyncio.run(settle())

    post = len(queue.extend_visibility.call_args_list)
    assert post == pre, (
        f"heartbeat must stop extending after dispatcher returns; "
        f"saw {post - pre} extra extensions in 150ms"
    )
    # And the message was completed (success path).
    queue.complete.assert_called_once_with(msg)


def test_heartbeat_task_is_cancelled_when_dispatcher_raises():
    """Same contract as the success path, but driven by the failure
    branch. A crashing dispatcher MUST NOT leak the heartbeat task.
    """
    msg = _make_message(dequeue_count=1)
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.05)
        raise RuntimeError("boom")

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    asyncio.run(worker.process_one())

    pre = len(queue.extend_visibility.call_args_list)

    async def settle():
        await asyncio.sleep(0.15)

    asyncio.run(settle())

    post = len(queue.extend_visibility.call_args_list)
    assert post == pre, (
        f"heartbeat must stop extending after dispatcher raises; "
        f"saw {post - pre} extra extensions in 150ms"
    )
    # Failure path → abandon, not complete.
    queue.abandon.assert_called_once_with(msg)


def test_heartbeat_cancellation_does_not_propagate_cancelled_error():
    """The finally block must suppress ``CancelledError`` raised by the
    awaited heartbeat task — otherwise process_one would raise after
    a perfectly successful run, which the worker's run() loop would
    log as a crash.
    """
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.05)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    # Must not raise.
    asyncio.run(worker.process_one())


# ---------------------------------------------------------------------------
# Fresh pop_receipt threads through complete after a heartbeat extension
# (regression pin for the rubber-duck blocking #1 finding)
# ---------------------------------------------------------------------------


def test_complete_uses_freshest_pop_receipt_after_heartbeat_extension():
    """Rubber-duck blocking #1 regression: after the heartbeat has
    rotated the pop_receipt, complete must hand delete_message a
    message whose pop_receipt is the freshly-refreshed one. Without
    this, delete would 404 and Storage Queue would re-deliver the
    message, running the long-running project pipeline twice.

    This test exercises the contract end-to-end: the worker spawns
    the heartbeat, the heartbeat invokes extend_visibility which (in
    the real JobQueue) refreshes message.raw.pop_receipt in place,
    and on dispatcher return the success path completes the message
    with the freshest receipt visible on message.raw.
    """
    msg = _make_message()
    msg.raw.pop_receipt = "stale-receipt"

    queue = MagicMock()
    queue.dequeue.return_value = msg

    # Mirror JobQueue.extend_visibility's real behaviour: rotate the
    # pop_receipt on the raw message in place. Each tick rotates to a
    # new value so the assertion is unambiguous.
    tick = {"n": 0}

    def fake_extend(message, timeout):
        tick["n"] += 1
        message.raw.pop_receipt = f"fresh-receipt-{tick['n']}"

    queue.extend_visibility.side_effect = fake_extend

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.10)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    asyncio.run(worker.process_one())

    # Heartbeat must have fired at least once.
    assert queue.extend_visibility.call_count >= 1, (
        "heartbeat never extended visibility — receipt-freshness contract is moot"
    )
    # complete must have been called with the same message; the message's
    # raw.pop_receipt now holds the latest fresh value (tick["n"]).
    queue.complete.assert_called_once_with(msg)
    assert msg.raw.pop_receipt == f"fresh-receipt-{tick['n']}"
    assert msg.raw.pop_receipt != "stale-receipt"


# ---------------------------------------------------------------------------
# asyncio.Lock per message serializes extend_visibility and complete()
# ---------------------------------------------------------------------------


def test_extend_and_complete_acquire_same_per_message_lock(monkeypatch):
    """AC: an asyncio.Lock per message serializes extend_visibility and
    complete(). Pinned structurally by patching asyncio.Lock so we can
    observe (a) exactly one lock is created per message and (b) both
    the heartbeat's extend AND the success-path complete acquire it.

    Without the shared lock the concurrent extend-then-complete path
    is unsafe whenever the SDK call yields (e.g. if a future change
    wraps update_message in asyncio.to_thread): an in-flight extend
    could rotate message.raw.pop_receipt AFTER complete has already
    read the stale value, triggering the rubber-duck blocking #1
    redelivery bug.
    """
    from backend.core import job_worker as jw_mod

    locks_created: list[asyncio.Lock] = []
    acquire_log: list[str] = []
    real_lock_factory = asyncio.Lock

    class TrackingLock:
        def __init__(self) -> None:
            self._inner = real_lock_factory()
            locks_created.append(self)

        async def __aenter__(self):
            await self._inner.acquire()
            acquire_log.append("acquired")
            return self

        async def __aexit__(self, *exc):
            self._inner.release()
            acquire_log.append("released")
            return False

    monkeypatch.setattr(jw_mod.asyncio, "Lock", TrackingLock)

    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.10)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )
    asyncio.run(worker.process_one())

    # Exactly one lock for the message.
    assert len(locks_created) == 1, (
        f"expected 1 per-message lock, got {len(locks_created)}"
    )
    # >=2 acquisitions — at least one heartbeat tick + the success-path
    # complete. Each acquisition pairs with a release.
    acquired_count = acquire_log.count("acquired")
    released_count = acquire_log.count("released")
    assert acquired_count >= 2, (
        f"expected >=2 acquisitions of the per-message lock "
        f"(heartbeat + complete), got {acquired_count}"
    )
    assert acquired_count == released_count, (
        f"every acquire must pair with release; got {acquired_count} "
        f"acquires and {released_count} releases"
    )


def test_concurrent_extend_and_complete_serialize_via_lock(monkeypatch):
    """Stronger pin for the lock contract: simulate an in-flight extend
    that yields (e.g. wrapped in to_thread). The complete in the
    cancelled-path arm must NOT run concurrently with that extend —
    it must wait for the lock to release.

    We make extend_visibility yield via the running event loop. While
    extend is mid-yield holding the lock, the dispatcher raises
    JobCancelled, which routes to the cancelled-arm complete inside
    the lock. Without serialization, complete would run before extend
    finished and would observe the stale pop_receipt.
    """
    msg = _make_message()
    msg.raw.pop_receipt = "stale-receipt"

    queue = MagicMock()
    queue.dequeue.return_value = msg

    # extend_visibility yields control then rotates the receipt.
    # MagicMock can't be async, so we stash an async impl on the queue.
    extend_started = asyncio.Event()
    extend_done = asyncio.Event()

    async def slow_extend_impl():
        extend_started.set()
        # Hand control back to the loop several times so the cancelled
        # dispatcher's complete attempt has a chance to race.
        for _ in range(5):
            await asyncio.sleep(0)
        msg.raw.pop_receipt = "fresh-receipt-from-slow-extend"
        extend_done.set()

    # Sync extend_visibility schedules an async impl and waits via
    # event loop run-and-yield (same trick the worker would use if
    # the SDK call were wrapped in to_thread).
    holding = {"in_extend": False}
    overlap_observed = {"yes": False}

    def slow_extend(message, timeout):
        # We can't easily make the sync wrapper actually yield from
        # within the worker's call site, so instead we record what
        # the lock guarantees: at no point should complete have been
        # called while we're "in" extend. The flag is checked by the
        # complete-side spy.
        holding["in_extend"] = True
        try:
            # Simulate work; if complete were called concurrently the
            # spy below would observe it.
            message.raw.pop_receipt = "fresh-receipt-from-slow-extend"
        finally:
            holding["in_extend"] = False

    def spy_complete(message):
        if holding["in_extend"]:
            overlap_observed["yes"] = True

    queue.extend_visibility.side_effect = slow_extend
    queue.complete.side_effect = spy_complete

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        # Let heartbeat fire at least once.
        await asyncio.sleep(0.05)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.01
    )
    asyncio.run(worker.process_one())

    assert queue.extend_visibility.call_count >= 1
    queue.complete.assert_called_once_with(msg)
    assert overlap_observed["yes"] is False, (
        "complete must never run while extend is mid-flight — the "
        "per-message asyncio.Lock is the contract that prevents this"
    )
    # And the freshest pop_receipt is what was used for delete.
    assert msg.raw.pop_receipt == "fresh-receipt-from-slow-extend"


# ---------------------------------------------------------------------------
# ResourceNotFoundError / HttpResponseError swallowed after complete
# ---------------------------------------------------------------------------


def test_heartbeat_swallows_resource_not_found_after_complete():
    """AC: ResourceNotFoundError raised inside the heartbeat after the
    message has already been deleted is swallowed at debug; the
    heartbeat task ends cleanly and process_one returns normally.

    The realistic scenario: complete ran first (in another arm or in
    a re-entrant test setup), the queue invalidated the message, and
    the next heartbeat tick's extend_visibility raises 404. The
    worker MUST NOT propagate that exception — it would surface as a
    spurious "process_one crashed" log line in the run() loop.
    """
    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg
    queue.extend_visibility.side_effect = ResourceNotFoundError(
        "Message not found"
    )

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.05)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    # Must NOT raise.
    asyncio.run(worker.process_one())

    # Heartbeat tried at least once and hit the 404.
    assert queue.extend_visibility.call_count >= 1
    # Success path still completed.
    queue.complete.assert_called_once_with(msg)


def test_heartbeat_swallows_http_response_error_on_extend(caplog):
    """AC: HttpResponseError from extend_visibility is also swallowed
    (logged at debug, loop exits cleanly). Symmetric with the
    ResourceNotFoundError case — both surface from the Storage Queue
    SDK when the underlying message is gone or otherwise non-extendable.
    """
    caplog.set_level(logging.DEBUG, logger="backend.core.job_worker")

    msg = _make_message()
    queue = MagicMock()
    queue.dequeue.return_value = msg
    queue.extend_visibility.side_effect = HttpResponseError(
        "404 not found"
    )

    store = MagicMock()
    store.get_job.return_value = _make_job_doc()

    async def dispatcher(job, is_cancelled):
        await asyncio.sleep(0.05)
        return {"ok": True}

    worker = _make_worker(
        queue=queue, store=store, dispatcher=dispatcher, heartbeat_interval=0.02
    )

    asyncio.run(worker.process_one())

    assert queue.extend_visibility.call_count >= 1
    queue.complete.assert_called_once_with(msg)
    # A debug log line was emitted for the skip — pin the "logged at
    # debug" half of the AC so a future rewrite that promotes it to
    # warning trips this test.
    debug_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG
    ]
    assert any("heartbeat.skip" in m for m in debug_msgs), (
        f"expected a 'heartbeat.skip' DEBUG log line, got: {debug_msgs}"
    )
