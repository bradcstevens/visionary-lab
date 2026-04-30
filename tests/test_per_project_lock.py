"""Tests for the per-project asyncio.Lock in StagingPipeline.

This slice (parallel-processing PRD, issue 005) ensures concurrent updates
to the same staging-project document are serialized via a per-project
asyncio.Lock, so read-modify-write races against Cosmos cannot lose
updates when slice 006 (hybrid parallelism) lands variation-level fan-out.

Tests verify externally-observable behavior at the public seams of the
pipeline. ``StagingStorageService.update_project`` is replaced with a
sync stub that performs a real read-modify-write on a dict, with a
deliberate ``time.sleep`` between read and write to widen the race
window. Production code now runs the storage call via ``asyncio.to_thread``,
so without the lock two concurrent tasks would run the stub on parallel
threads and the later writer would clobber the earlier writer.

Per the parent PRD's *Testing decisions*, no test asserts on internal
helper signatures, log strings, or counter values that aren't part of
the contract — only externally observable behavior at the public seam.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.staging_pipeline import StagingPipeline, _PROJECT_LOCKS
from backend.models.staging import (
    ItemStatus,
    Room,
    StagingProject,
    StagingSettings,
    Variation,
)


@pytest.fixture(autouse=True)
def _clear_project_locks():
    """Clear the module-level lock registry between tests.

    asyncio.Lock instances are bound to the event loop on which they were
    first awaited; pytest-asyncio creates a fresh loop per test, so locks
    leaked from a previous test would raise RuntimeError on next use.
    """
    _PROJECT_LOCKS.clear()
    yield
    _PROJECT_LOCKS.clear()


class _RmwStorageStub:
    """In-memory storage stub that does a real read-modify-write on a dict.

    Each ``update_project`` call:
      1. Increments the ``in_flight`` counter (under a thread lock).
      2. Sleeps for ``hold_seconds`` — the race window.
      3. Reads the current persisted doc.
      4. Merges in the supplied updates and writes back.
      5. Decrements ``in_flight``.

    Because the production code runs this stub via ``asyncio.to_thread``,
    two concurrent invocations would otherwise execute on parallel threads
    and both observe ``in_flight == 2``. With the per-project lock in
    place, ``peak_in_flight_for(project_id)`` stays at 1.
    """

    def __init__(self, hold_seconds: float = 0.02) -> None:
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.hold_seconds = hold_seconds
        # Tracking concurrency
        self._cnt_lock = threading.Lock()
        self._in_flight: Dict[str, int] = {}
        self._peak: Dict[str, int] = {}
        # Optional injected exception (one-shot)
        self.raise_once: Optional[Exception] = None
        self.raised_count = 0
        # Total invocation counter
        self.call_count = 0

    def peak_in_flight_for(self, project_id: str) -> int:
        return self._peak.get(project_id, 0)

    def total_in_flight(self) -> int:
        return sum(self._in_flight.values())

    def update_project(
        self, project_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        with self._cnt_lock:
            self.call_count += 1
            self._in_flight[project_id] = self._in_flight.get(project_id, 0) + 1
            self._peak[project_id] = max(
                self._peak.get(project_id, 0), self._in_flight[project_id]
            )
        try:
            if self.raise_once is not None and self.raised_count == 0:
                self.raised_count += 1
                raise self.raise_once
            existing = dict(self.docs.get(project_id, {}))
            time.sleep(self.hold_seconds)
            existing.update(updates)
            self.docs[project_id] = existing
            return existing
        finally:
            with self._cnt_lock:
                self._in_flight[project_id] -= 1


def _make_project(
    project_id: str = "proj-1",
    n_rooms: int = 1,
    n_variations: int = 1,
) -> StagingProject:
    rooms = []
    for i in range(n_rooms):
        variations = [Variation(id=f"var-{i}-{j}") for j in range(n_variations)]
        rooms.append(
            Room(
                id=f"room-{i}",
                label=f"Room {i+1}",
                original_image_url=(
                    "https://acct.blob.core.windows.net/images/staging/proj/"
                    "originals/photo.png"
                ),
                variations=variations,
            )
        )
    return StagingProject(
        id=project_id,
        name="Locked",
        prompt="prompt",
        settings=StagingSettings(variations_per_room=n_variations),
        rooms=rooms,
    )


def _build_staging_pipeline(storage: Any) -> StagingPipeline:
    """Construct a StagingPipeline with all deps mocked except storage."""
    return StagingPipeline(
        async_llm_client=AsyncMock(),
        llm_deployment="gpt-4o",
        image_analyzer=AsyncMock(),
        image_pipeline=AsyncMock(),
        storage_service=storage,
        blob_service=MagicMock(),
    )


class TestProjectLockHelper:
    """Lock-dictionary management."""

    @pytest.mark.asyncio
    async def test_lock_dict_is_initially_empty(self):
        # Pre-test fixture clears the module-level registry; verify that.
        assert _PROJECT_LOCKS == {}

    @pytest.mark.asyncio
    async def test_same_id_returns_same_lock(self):
        staging = _build_staging_pipeline(MagicMock())
        a = staging._get_project_lock("proj-X")
        b = staging._get_project_lock("proj-X")
        assert a is b
        assert isinstance(a, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_different_ids_get_different_locks(self):
        staging = _build_staging_pipeline(MagicMock())
        a = staging._get_project_lock("proj-A")
        b = staging._get_project_lock("proj-B")
        assert a is not b


class TestPerProjectLockSerializesSameProject:
    """The lock must serialize concurrent persists for the same project."""

    @pytest.mark.asyncio
    async def test_same_project_persists_are_serialized(self):
        """25 concurrent ``_update_room_in_project`` calls for the same
        project must NOT overlap inside ``update_project`` — peak
        in-flight for that project is exactly 1.

        Without the lock, ``asyncio.to_thread`` would let multiple threads
        enter ``update_project`` simultaneously (both inside ``time.sleep``),
        producing a peak of 2 or more.
        """
        storage = _RmwStorageStub(hold_seconds=0.005)
        staging = _build_staging_pipeline(storage)
        project = _make_project(project_id="proj-1", n_rooms=25, n_variations=1)

        async def update(idx: int) -> None:
            project.rooms[idx].status = ItemStatus.COMPLETED
            await staging._update_room_in_project(project, project.rooms[idx])

        await asyncio.gather(*(update(i) for i in range(25)))

        assert storage.peak_in_flight_for("proj-1") == 1, (
            "Per-project lock failed: peak in-flight for proj-1 was "
            f"{storage.peak_in_flight_for('proj-1')} (expected 1)."
        )
        # All 25 calls completed.
        assert storage.call_count == 25

    @pytest.mark.asyncio
    async def test_no_lost_updates_under_concurrent_persists(self):
        """All concurrent updates land in the final persisted state."""
        storage = _RmwStorageStub(hold_seconds=0.005)
        staging = _build_staging_pipeline(storage)
        project = _make_project(project_id="proj-keep", n_rooms=10, n_variations=1)

        async def update(idx: int) -> None:
            project.rooms[idx].status = ItemStatus.COMPLETED
            await staging._update_room_in_project(project, project.rooms[idx])

        await asyncio.gather(*(update(i) for i in range(10)))

        persisted = storage.docs["proj-keep"]
        completed_room_ids = {
            r["id"] for r in persisted.get("rooms", []) if r.get("status") == "completed"
        }
        assert completed_room_ids == {f"room-{i}" for i in range(10)}, (
            "Lost updates: not every room shows COMPLETED in the final "
            f"persisted state. Got {completed_room_ids}."
        )


class TestPerProjectLockDoesNotBlockDifferentProjects:
    """The lock must be per-project — different projects run in parallel."""

    @pytest.mark.asyncio
    async def test_different_projects_persist_in_parallel(self):
        """An update on project A must not block an update on project B."""
        hold = 0.10
        storage = _RmwStorageStub(hold_seconds=hold)
        staging = _build_staging_pipeline(storage)
        project_a = _make_project(project_id="proj-A")
        project_b = _make_project(project_id="proj-B")

        async def update(p: StagingProject) -> None:
            p.rooms[0].status = ItemStatus.COMPLETED
            await staging._update_room_in_project(p, p.rooms[0])

        start = time.monotonic()
        await asyncio.gather(update(project_a), update(project_b))
        elapsed = time.monotonic() - start

        # Sequential would be ~2*hold = 0.20s. Parallel should be ~hold + small
        # asyncio overhead. Allow a generous CI margin: < 1.5*hold proves
        # clear parallelism.
        assert elapsed < hold * 1.5, (
            f"Different projects appear serialized: took {elapsed:.3f}s "
            f"(expected < {hold * 1.5:.3f}s)."
        )

    @pytest.mark.asyncio
    async def test_different_projects_can_be_in_flight_simultaneously(self):
        """Observe that, at peak, both projects had a thread inside
        ``update_project`` at once. This is the smoking-gun proof that the
        locks are scoped per-project, not globally."""
        # Use an asyncio.Event to gate release so we can observe in-flight
        # state mid-call.
        release = threading.Event()

        class _GatedStub(_RmwStorageStub):
            def update_project(self, project_id, updates):  # type: ignore[override]
                with self._cnt_lock:
                    self.call_count += 1
                    self._in_flight[project_id] = self._in_flight.get(project_id, 0) + 1
                    self._peak[project_id] = max(
                        self._peak.get(project_id, 0), self._in_flight[project_id]
                    )
                try:
                    # Hold here until the test releases.
                    release.wait(timeout=2.0)
                    existing = dict(self.docs.get(project_id, {}))
                    existing.update(updates)
                    self.docs[project_id] = existing
                    return existing
                finally:
                    with self._cnt_lock:
                        self._in_flight[project_id] -= 1

        storage = _GatedStub()
        staging = _build_staging_pipeline(storage)
        project_a = _make_project(project_id="proj-A")
        project_b = _make_project(project_id="proj-B")

        async def update(p: StagingProject) -> None:
            p.rooms[0].status = ItemStatus.COMPLETED
            await staging._update_room_in_project(p, p.rooms[0])

        task_a = asyncio.create_task(update(project_a))
        task_b = asyncio.create_task(update(project_b))

        # Wait until both threads are inside update_project.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if storage.total_in_flight() >= 2:
                break
            await asyncio.sleep(0.005)

        try:
            assert storage.total_in_flight() == 2, (
                "Different projects did not enter update_project concurrently "
                f"(total_in_flight={storage.total_in_flight()}); the per-project "
                "lock appears to be blocking unrelated projects."
            )
        finally:
            release.set()
            await asyncio.gather(task_a, task_b)


class TestLockReleasedOnException:
    """The lock must be released on exception in the critical section."""

    @pytest.mark.asyncio
    async def test_follow_up_call_does_not_deadlock_after_exception(self):
        """An injected failure in the first persist must not leak the lock."""
        storage = _RmwStorageStub(hold_seconds=0.0)
        storage.raise_once = RuntimeError("injected boom")
        staging = _build_staging_pipeline(storage)
        project = _make_project(project_id="proj-x")

        # _update_room_in_project catches Exception internally and logs it
        # (so the caller does not observe the failure).
        await staging._update_room_in_project(project, project.rooms[0])
        assert storage.raised_count == 1

        # The follow-up call must complete promptly — no stuck lock.
        async def follow_up():
            await staging._update_room_in_project(project, project.rooms[0])

        await asyncio.wait_for(follow_up(), timeout=1.0)

        # And the lock for this project is no longer held.
        assert not staging._get_project_lock("proj-x").locked()

    @pytest.mark.asyncio
    async def test_concurrent_call_after_exception_serializes_correctly(self):
        """After an exception releases the lock, a subsequent concurrent
        burst on the same project still serializes correctly."""
        storage = _RmwStorageStub(hold_seconds=0.005)
        storage.raise_once = RuntimeError("first call boom")
        staging = _build_staging_pipeline(storage)
        project = _make_project(project_id="proj-y", n_rooms=5)

        # First call raises and is swallowed.
        await staging._update_room_in_project(project, project.rooms[0])

        # Now drive 5 concurrent updates; lock must serialize them.
        async def update(idx: int) -> None:
            project.rooms[idx].status = ItemStatus.COMPLETED
            await staging._update_room_in_project(project, project.rooms[idx])

        await asyncio.gather(*(update(i) for i in range(5)))

        assert storage.peak_in_flight_for("proj-y") == 1


class TestProjectLockIsProcessWide:
    """The lock registry must be shared across StagingPipeline *instances*.

    `get_staging_pipeline()` (the FastAPI Depends factory) constructs a new
    StagingPipeline per request. The lock dict therefore cannot live on the
    instance — two concurrent requests for the same project would otherwise
    grab disjoint locks and race the read-modify-write. This test
    constructs two independent StagingPipeline instances pointing at the
    same storage stub and asserts they still serialize.
    """

    @pytest.mark.asyncio
    async def test_two_pipeline_instances_share_lock_registry(self):
        storage = _RmwStorageStub(hold_seconds=0.005)
        pipeline_a = _build_staging_pipeline(storage)
        pipeline_b = _build_staging_pipeline(storage)
        assert pipeline_a is not pipeline_b
        # Same project_id from two independently-constructed pipelines must
        # resolve to the *same* asyncio.Lock instance.
        lock_a = pipeline_a._get_project_lock("proj-shared")
        lock_b = pipeline_b._get_project_lock("proj-shared")
        assert lock_a is lock_b

        project_a = _make_project(project_id="proj-shared", n_rooms=1)
        project_b = _make_project(project_id="proj-shared", n_rooms=1)

        async def via_a() -> None:
            await pipeline_a._update_room_in_project(project_a, project_a.rooms[0])

        async def via_b() -> None:
            await pipeline_b._update_room_in_project(project_b, project_b.rooms[0])

        # Drive a burst of 10 concurrent persists across both instances.
        await asyncio.gather(
            *(via_a() if i % 2 == 0 else via_b() for i in range(10))
        )

        # Despite being driven from two different pipeline instances, peak
        # in-flight stays at 1 — the shared registry serializes them.
        assert storage.peak_in_flight_for("proj-shared") == 1


class _GatedStorageStub:
    """Storage stub whose `update_project` blocks on a threading.Event.

    Used to simulate a slow Cosmos write so we can deterministically cancel
    the awaiter task while the worker thread is mid-call.
    """

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.completed_calls: list[str] = []
        self._cnt_lock = threading.Lock()

    def update_project(
        self, project_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        with self._cnt_lock:
            self.entered.set()
        self.gate.wait(timeout=5.0)  # Block until released.
        with self._cnt_lock:
            self.completed_calls.append(project_id)
        return dict(updates)


class TestPersistLockedSurvivesCancellation:
    """If the awaiter is cancelled while the Cosmos thread is in flight,
    the lock must remain held until that thread completes — otherwise a
    second writer enters the lock and races the still-running write.
    """

    @pytest.mark.asyncio
    async def test_cancellation_holds_lock_until_thread_finishes(self):
        storage = _GatedStorageStub()
        staging = _build_staging_pipeline(storage)
        project = _make_project(project_id="proj-cxl", n_rooms=1)

        # Task A: enter the lock, get suspended inside `to_thread` waiting
        # on the gate, then we cancel its awaiter.
        task_a = asyncio.create_task(staging._persist_project_locked(project))

        # Wait for the worker thread to actually enter `update_project`.
        await asyncio.to_thread(storage.entered.wait, 5.0)
        assert storage.entered.is_set()

        # Cancel A's awaiter while its thread is still blocked on the gate.
        task_a.cancel()

        # Give the cancellation a moment to propagate without releasing the
        # gate — the lock must NOT be released yet because the thread is
        # still running.
        await asyncio.sleep(0.05)
        assert staging._get_project_lock("proj-cxl").locked(), (
            "Lock must remain held while the thread finishes — releasing "
            "early would let the next writer race the in-flight Cosmos call."
        )

        # Now release the gate; the thread completes; A's task finally
        # raises CancelledError; the lock is released.
        storage.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task_a

        # Lock is now released.
        assert not staging._get_project_lock("proj-cxl").locked()
        # And the thread's write did complete.
        assert storage.completed_calls == ["proj-cxl"]

    @pytest.mark.asyncio
    async def test_cancelled_then_next_writer_does_not_overlap(self):
        """Stronger version: after cancellation, a second writer must not
        observe the first writer's thread still in flight."""
        storage = _GatedStorageStub()
        staging = _build_staging_pipeline(storage)
        project = _make_project(project_id="proj-cxl-2", n_rooms=1)

        task_a = asyncio.create_task(staging._persist_project_locked(project))

        # A enters the gate.
        await asyncio.to_thread(storage.entered.wait, 5.0)

        # Cancel A; the lock is held until A's thread finishes.
        task_a.cancel()

        # Schedule B — it should block on the lock.
        b_started_evt = threading.Event()

        # Reset entered before launching B so we observe its entry.
        storage.entered.clear()

        async def writer_b() -> None:
            await staging._persist_project_locked(project)
            b_started_evt.set()

        task_b = asyncio.create_task(writer_b())

        # Give B time to attempt to enter; it must NOT have entered the
        # storage yet because A's lock is still held.
        await asyncio.sleep(0.05)
        assert not storage.entered.is_set(), (
            "Writer B must not enter `update_project` while A's thread "
            "is still in flight."
        )

        # Release A's gate; A's thread completes; A raises CancelledError;
        # lock is released; B acquires; B blocks on the (still-set) gate.
        storage.gate.set()

        # A finishes (cancelled).
        with pytest.raises(asyncio.CancelledError):
            await task_a

        # B now enters and completes (gate is already set).
        await asyncio.wait_for(task_b, timeout=2.0)

        # Both writes completed in order: A's thread first, then B's.
        assert storage.completed_calls == ["proj-cxl-2", "proj-cxl-2"]
