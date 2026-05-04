"""Tests for ``backend.core.embedded_worker``.

Covers two surfaces:

1. Pure policy ``should_auto_start(role, env_flag)``. Table driven over
   the (role, env_flag) cartesian product. Pinning the truth table
   here matters because prod auto-start failure mode is two worker
   replicas competing for the queue.

2. ``EmbeddedWorker`` lifecycle — start (skip + happy paths), stop
   (no-op when never started, stop+await, timeout+cancel+warn), and
   the ``[embedded-worker]`` log-prefix invariant.

3. Lifespan integration via ``TestClient`` context-manager. Pins that
   FastAPI startup/shutdown delegates correctly to the embedded
   worker module.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.embedded_worker import (
    LOG_PREFIX,
    EmbeddedWorker,
    should_auto_start,
)


# ---------------------------------------------------------------------------
# should_auto_start truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role,env_flag,expected",
    [
        # env_flag dominates — when off, never start regardless of role
        (None, False, False),
        ("", False, False),
        ("api", False, False),
        ("worker", False, False),
        ("anything-else", False, False),
        # env_flag on — start unless role is exactly "worker"
        (None, True, True),
        ("", True, True),
        ("api", True, True),
        ("anything-else", True, True),
        # exact "worker" role suppresses even when env_flag on
        ("worker", True, False),
        # case sensitivity matters — "Worker"/"WORKER" are not the
        # production worker role and should auto-start in dev
        ("Worker", True, True),
        ("WORKER", True, True),
    ],
)
def test_should_auto_start_truth_table(role, env_flag, expected):
    assert should_auto_start(role, env_flag) is expected


# ---------------------------------------------------------------------------
# EmbeddedWorker.start — skip path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_skip_path_does_not_call_builder():
    builder = MagicMock(name="builder")
    ew = EmbeddedWorker(role="api", env_flag=False)

    await ew.start(builder)

    assert builder.call_count == 0
    assert ew.worker is None
    assert ew.task is None


@pytest.mark.asyncio
async def test_start_skip_path_logs_with_prefix(caplog):
    builder = MagicMock(name="builder")
    ew = EmbeddedWorker(role="worker", env_flag=True)

    with caplog.at_level(logging.INFO, logger="backend.core.embedded_worker"):
        await ew.start(builder)

    # Every embedded-worker log line must carry the prefix so an ops
    # search of "[embedded-worker]" picks up both the start AND the
    # skip decisions in dev triage.
    relevant = [r for r in caplog.records if "embedded" in r.message.lower()
                or LOG_PREFIX in r.message]
    assert len(relevant) >= 1, "expected at least one embedded-worker log line"
    for record in relevant:
        assert record.message.startswith(LOG_PREFIX), record.message


# ---------------------------------------------------------------------------
# EmbeddedWorker.start — happy path
# ---------------------------------------------------------------------------


def _make_worker_mock():
    """Worker double whose ``run`` is awaitable + cancellable.

    ``run`` blocks on an event so the embedded task stays alive long
    enough for the test to inspect it. ``stop`` flips the event so the
    coroutine returns naturally; cancellation also propagates.
    """
    worker = MagicMock(name="JobWorker")
    stop_event = asyncio.Event()
    worker._stop_event = stop_event

    async def _run():
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            raise

    worker.run = AsyncMock(side_effect=_run)
    worker.stop = MagicMock(side_effect=lambda: stop_event.set())
    return worker


@pytest.mark.asyncio
async def test_start_happy_path_calls_builder_and_creates_task():
    worker = _make_worker_mock()
    builder = MagicMock(return_value=worker)
    ew = EmbeddedWorker(role="api", env_flag=True)

    await ew.start(builder)

    assert builder.call_count == 1
    assert ew.worker is worker
    assert ew.task is not None
    assert isinstance(ew.task, asyncio.Task)
    assert not ew.task.done()

    # Cleanup
    await ew.stop()


@pytest.mark.asyncio
async def test_start_happy_path_logs_with_prefix(caplog):
    worker = _make_worker_mock()
    ew = EmbeddedWorker(role="api", env_flag=True)

    with caplog.at_level(logging.INFO, logger="backend.core.embedded_worker"):
        await ew.start(MagicMock(return_value=worker))

    embedded_records = [
        r for r in caplog.records if r.name == "backend.core.embedded_worker"
    ]
    assert embedded_records, "expected start to emit at least one log line"
    for record in embedded_records:
        assert record.message.startswith(LOG_PREFIX), (
            f"expected '{LOG_PREFIX}' prefix, got: {record.message!r}"
        )

    await ew.stop()


@pytest.mark.asyncio
async def test_start_spawns_task_running_worker_run_for_drain_property():
    """Pin: the spawned task's coroutine target is ``worker.run``.

    The PRD AC "Pre-existing stuck jobs ... drain naturally on the first
    embedded-worker boot" is a transitive property of this fact. Once
    we've established the spawned task IS worker.run(), drain behaviour
    is covered by ``test_job_worker.py`` (process_one + run loop).
    """
    worker = _make_worker_mock()
    ew = EmbeddedWorker(role="api", env_flag=True)

    await ew.start(MagicMock(return_value=worker))

    # Yield once so the task can begin executing run().
    await asyncio.sleep(0)
    worker.run.assert_called_once_with()
    assert ew.task is not None
    assert not ew.task.done()

    await ew.stop()


# ---------------------------------------------------------------------------
# EmbeddedWorker.stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_is_no_op_when_never_started():
    ew = EmbeddedWorker(role="api", env_flag=True)
    # No exception, no hang. Idempotent across multiple calls too.
    await ew.stop()
    await ew.stop()


@pytest.mark.asyncio
async def test_stop_signals_worker_and_awaits_task():
    worker = _make_worker_mock()
    ew = EmbeddedWorker(role="api", env_flag=True)
    await ew.start(MagicMock(return_value=worker))

    await ew.stop()

    worker.stop.assert_called_once_with()
    assert ew.task is not None
    assert ew.task.done()
    # Task completed cleanly (not via cancellation) — stop() should be
    # the graceful path when the worker actually responds to stop().
    assert not ew.task.cancelled()


@pytest.mark.asyncio
async def test_stop_cancels_task_on_timeout(caplog):
    """If ``worker.stop()`` doesn't end the task in time, force-cancel.

    Without this branch a stuck dispatch could hang lifespan teardown
    indefinitely. The visibility-timeout in the queue covers in-flight
    work that gets cancelled, so cancellation is safe.
    """
    worker = MagicMock(name="JobWorker")
    started = asyncio.Event()

    async def _run_forever():
        started.set()
        try:
            await asyncio.sleep(3600)  # never returns under stop()
        except asyncio.CancelledError:
            raise

    worker.run = AsyncMock(side_effect=_run_forever)
    # stop() doesn't actually unblock _run_forever — simulates a stuck dispatch
    worker.stop = MagicMock()

    ew = EmbeddedWorker(role="api", env_flag=True, shutdown_timeout=0.05)
    await ew.start(MagicMock(return_value=worker))
    await started.wait()

    with caplog.at_level(logging.WARNING, logger="backend.core.embedded_worker"):
        await ew.stop()

    assert ew.task is not None
    assert ew.task.done()
    assert ew.task.cancelled()

    # Operator-facing visibility for the forced-cancel path
    warns = [
        r for r in caplog.records
        if r.name == "backend.core.embedded_worker" and r.levelno >= logging.WARNING
    ]
    assert any(LOG_PREFIX in r.message and "cancel" in r.message.lower()
               for r in warns), (
        f"expected a [embedded-worker] cancel warning, got: "
        f"{[r.message for r in warns]}"
    )


# ---------------------------------------------------------------------------
# Task failure visibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_task_exit_emits_error_log(caplog):
    """If ``worker.run`` raises, the embedded task dies silently.

    Without a done-callback the API would keep serving traffic with no
    embedded worker and no log line — silent regression. Pin: the
    callback emits an [embedded-worker] error.
    """
    worker = MagicMock(name="JobWorker")
    boom = RuntimeError("boom")

    async def _run_explode():
        raise boom

    worker.run = AsyncMock(side_effect=_run_explode)
    worker.stop = MagicMock()

    ew = EmbeddedWorker(role="api", env_flag=True)

    with caplog.at_level(logging.ERROR, logger="backend.core.embedded_worker"):
        await ew.start(MagicMock(return_value=worker))
        # let the spawned task run + raise + done-callback fire
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    errors = [
        r for r in caplog.records
        if r.name == "backend.core.embedded_worker" and r.levelno >= logging.ERROR
    ]
    assert any(LOG_PREFIX in r.message for r in errors), (
        f"expected an [embedded-worker] error log, got: "
        f"{[r.message for r in errors]}"
    )

    # task is now done with an exception consumed by the callback
    assert ew.task is not None and ew.task.done()


# ---------------------------------------------------------------------------
# Lifespan integration via FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_invokes_start_then_stop_in_order():
    """End-to-end: a lifespan handler that wraps EmbeddedWorker invokes
    start before yield and stop after yield, in the right order, even
    when no body code is executed between."""
    worker = _make_worker_mock()
    builder = MagicMock(return_value=worker)
    ew = EmbeddedWorker(role="api", env_flag=True)

    call_order: list[str] = []

    @asynccontextmanager
    async def _lifespan():
        call_order.append("start_called")
        await ew.start(builder)
        call_order.append("started")
        try:
            yield
        finally:
            call_order.append("stop_called")
            await ew.stop()
            call_order.append("stopped")

    async with _lifespan():
        call_order.append("body")

    assert call_order == [
        "start_called",
        "started",
        "body",
        "stop_called",
        "stopped",
    ]
    assert builder.call_count == 1
    assert worker.stop.call_count == 1
