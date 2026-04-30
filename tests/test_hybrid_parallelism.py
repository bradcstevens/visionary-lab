"""Tests for hybrid parallelism in the staging pipeline.

Issue 006 of the parallel-processing PRD: rooms run concurrently AND
variations within each room run concurrently. The global image-call
cap (slice 4, ``IMAGE_GEN_SEMAPHORE``) is the rate-limit bound; the
room-level cap (``STAGING_CONCURRENT_ROOMS``) is purely a memory bound
and ``process_single_variation`` no longer acquires it.

Tests verify externally-observable behavior at the public seams of the
pipeline:

* ``image_pipeline.process_pipeline`` is mocked at its boundary; we
  observe how many invocations are concurrently in flight to assert
  variation fan-out.
* ``StagingStorageService.update_project`` is mocked at its boundary;
  we observe write-counts to assert no zombie writes for cancelled
  variations.

Per the PRD's testing decisions, no test asserts on
``Semaphore._value``, internal helper signatures, or log strings.
Concurrency assertions use ``asyncio.Event``-gated mocks so ordering
is observable rather than timing-dependent.

The IMAGE_GEN_SEMAPHORE-cap-enforcement contract is verified
exhaustively in ``tests/test_global_image_semaphore.py``; this file
exercises the new fan-out behavior in ``StagingPipeline``.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.staging_pipeline import StagingPipeline, _PROJECT_LOCKS
from backend.models.images import (
    ImageGenerationResponse,
    ImagePipelineResponse,
    ImageSaveResponse,
    PipelineStepResult,
)
from backend.models.staging import (
    ItemStatus,
    ProjectStatus,
    Room,
    StagingProject,
    StagingSettings,
    Variation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_project_locks():
    """Clear the module-level lock registry between tests.

    asyncio.Lock instances are bound to the event loop on which they
    were first awaited; pytest-asyncio creates a fresh loop per test,
    so locks leaked from a previous test would raise RuntimeError on
    next use.
    """
    _PROJECT_LOCKS.clear()
    yield
    _PROJECT_LOCKS.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(
    project_id: str = "proj-hybrid",
    n_rooms: int = 1,
    n_variations: int = 5,
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
        name="Hybrid",
        prompt="Modern minimalist",
        settings=StagingSettings(variations_per_room=n_variations),
        rooms=rooms,
    )


def _make_pipeline_response(
    image_url: str = "https://acct.blob.core.windows.net/images/staging/proj/variations/room-0/img.png",
) -> ImagePipelineResponse:
    gen = ImageGenerationResponse(
        success=True,
        message="ok",
        imgen_model_response={
            "data": [{"b64_json": "AAAA"}],
            "usage": {"total_tokens": 100, "input_tokens": 50, "output_tokens": 50},
        },
    )
    save = ImageSaveResponse(
        success=True,
        message="Saved 1 image(s)",
        saved_images=[{"url": image_url, "blob_name": "x"}],
        total_saved=1,
    )
    return ImagePipelineResponse(
        success=True,
        message="Pipeline completed",
        steps=[
            PipelineStepResult(step="edit", success=True),
            PipelineStepResult(step="save", success=True),
        ],
        generation=gen,
        save=save,
    )


def _build_staging_pipeline(
    image_pipeline: Any = None,
    storage: Any = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> StagingPipeline:
    """Construct a StagingPipeline with all deps mocked.

    ``semaphore`` overrides the room-level semaphore for tests that
    pin it independently of the runtime default.
    """
    if image_pipeline is None:
        image_pipeline = AsyncMock()
        image_pipeline.process_pipeline = AsyncMock(
            return_value=_make_pipeline_response()
        )

    if storage is None:
        storage = MagicMock()
        storage.update_project = MagicMock()

    mock_blob = MagicMock()
    mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

    mock_analyzer = AsyncMock()
    mock_analyzer.async_image_chat.return_value = {
        "description": "A room",
        "features": [],
    }

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]'
                )
            )
        ]
    )

    pipeline = StagingPipeline(
        async_llm_client=mock_llm,
        llm_deployment="gpt-4o",
        image_analyzer=mock_analyzer,
        image_pipeline=image_pipeline,
        storage_service=storage,
        blob_service=mock_blob,
    )
    if semaphore is not None:
        pipeline.semaphore = semaphore
    return pipeline


class _GatedProcessPipeline:
    """Async-callable that records concurrent in-flight calls.

    Each call:
      1. Increments in_flight + records peak.
      2. Records arrival in arrived_event for tests that need to know
         when N calls have entered.
      3. Awaits the release event (caller releases at will).
      4. Returns a successful pipeline response.

    Use this to observe fan-out concurrency at the
    ``image_pipeline.process_pipeline`` seam.
    """

    def __init__(self, target_arrivals: Optional[int] = None) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.total_calls = 0
        self.completed_calls = 0
        self.release = asyncio.Event()
        # Set by _check_target after `target_arrivals` calls have entered.
        self.all_arrived: Optional[asyncio.Event] = (
            asyncio.Event() if target_arrivals is not None else None
        )
        self._target = target_arrivals
        self._lock = asyncio.Lock()
        # Optional: cancellation observed
        self.cancelled_count = 0

    async def __call__(self, **kwargs: Any) -> ImagePipelineResponse:
        async with self._lock:
            self.total_calls += 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            if self.all_arrived is not None and self._target is not None:
                if self.in_flight >= self._target:
                    self.all_arrived.set()
        try:
            await self.release.wait()
            async with self._lock:
                self.completed_calls += 1
            return _make_pipeline_response()
        except asyncio.CancelledError:
            async with self._lock:
                self.cancelled_count += 1
            raise
        finally:
            async with self._lock:
                self.in_flight -= 1


# ---------------------------------------------------------------------------
# Within-room variation fan-out
# ---------------------------------------------------------------------------


class TestVariationsRunConcurrentlyWithinRoom:
    """A single room with N variations should fan all N image-gen calls
    out concurrently, not iterate them sequentially.
    """

    @pytest.mark.asyncio
    async def test_one_room_five_variations_all_in_flight_concurrently(self):
        """1×5 project: at peak, 5 ``process_pipeline`` calls are in flight."""
        gated = _GatedProcessPipeline(target_arrivals=5)
        mock_pipeline = MagicMock()
        mock_pipeline.process_pipeline = gated

        staging = _build_staging_pipeline(image_pipeline=mock_pipeline)
        project = _make_project(n_rooms=1, n_variations=5)

        events: List[Dict[str, Any]] = []

        async def consume() -> None:
            async for event in staging.generate_project(project):
                events.append(event)

        task = asyncio.create_task(consume())

        # Wait until all 5 variation workers have entered process_pipeline.
        # Without fan-out, only 1 would ever be in flight at once.
        await asyncio.wait_for(gated.all_arrived.wait(), timeout=2.0)

        assert gated.in_flight == 5, (
            f"Expected 5 in-flight image calls (variation fan-out), "
            f"got {gated.in_flight}. Variations are running sequentially."
        )
        assert gated.peak_in_flight == 5

        # Release them all and let the project finish.
        gated.release.set()
        await asyncio.wait_for(task, timeout=5.0)

        # Externally observable: 5 variation_completed events emitted.
        completed = [e for e in events if e["type"] == "variation_completed"]
        assert len(completed) == 5
        # All variation_index values 0..4 are present.
        indexes = sorted(e["variation_index"] for e in completed)
        assert indexes == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_room_completed_emitted_after_all_variation_events(self):
        """``room_completed`` must arrive only after every variation event
        for that room has been yielded (no interleaving where the
        room-completion races ahead of late variation completions).
        """
        gated = _GatedProcessPipeline(target_arrivals=3)
        mock_pipeline = MagicMock()
        mock_pipeline.process_pipeline = gated

        staging = _build_staging_pipeline(image_pipeline=mock_pipeline)
        project = _make_project(n_rooms=1, n_variations=3)

        events: List[Dict[str, Any]] = []

        async def consume() -> None:
            async for event in staging.generate_project(project):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(gated.all_arrived.wait(), timeout=2.0)
        gated.release.set()
        await asyncio.wait_for(task, timeout=5.0)

        # Find the room_completed event index, count variation events
        # before it: must be 3.
        room_completed_idx = next(
            i for i, e in enumerate(events) if e["type"] == "room_completed"
        )
        variation_events_before_room_completed = [
            e
            for e in events[:room_completed_idx]
            if e["type"] in ("variation_completed", "variation_failed")
        ]
        assert len(variation_events_before_room_completed) == 3, (
            f"room_completed arrived after only "
            f"{len(variation_events_before_room_completed)} variation events; "
            f"expected 3."
        )


# ---------------------------------------------------------------------------
# Room-level cap (memory bound)
# ---------------------------------------------------------------------------


class TestRoomLevelSemaphoreBoundsRoomWorkers:
    """The room-level ``STAGING_CONCURRENT_ROOMS`` semaphore continues
    to bound the number of concurrent ROOM workers (each holds a base64
    original + SSE generator in memory). Variation fan-out within a
    single room is gated only by the global image-call cap.
    """

    @pytest.mark.asyncio
    async def test_25_room_project_bounded_by_room_semaphore(self):
        """25 rooms × 1 variation, with room-level cap=10: at most 10
        rooms concurrently *inside* ``process_room`` (i.e., past the
        ``self.semaphore`` acquisition).

        We replace ``process_room`` with a stub that acquires the same
        semaphore the production code does, then sleeps to widen the
        observation window. This mirrors the pattern used in
        ``tests/test_parallel_rooms.py::test_semaphore_limits_concurrency``.
        """
        room_semaphore = asyncio.Semaphore(10)
        staging = _build_staging_pipeline(semaphore=room_semaphore)
        project = _make_project(n_rooms=25, n_variations=1)

        peak = 0
        current = 0
        peak_lock = asyncio.Lock()

        async def counting_process_room(proj, room, brief_prompts=None):
            nonlocal peak, current
            async with staging.semaphore:
                async with peak_lock:
                    current += 1
                    peak = max(peak, current)
                try:
                    await asyncio.sleep(0.02)
                    yield {"type": "room_completed", "room_id": room.id}
                finally:
                    async with peak_lock:
                        current -= 1

        staging.process_room = counting_process_room

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        room_completed = [e for e in events if e["type"] == "room_completed"]
        assert len(room_completed) == 25
        assert peak <= 10, (
            f"Peak room concurrency was {peak}; expected <= 10 "
            f"(STAGING_CONCURRENT_ROOMS bound)."
        )


# ---------------------------------------------------------------------------
# process_single_variation no longer acquires the room-level semaphore
# ---------------------------------------------------------------------------


class TestProcessSingleVariationDropsRoomLevelCap:
    """Issue 006: ``process_single_variation`` is a single image call
    bound only by the global image-call cap. It must NOT acquire the
    room-level ``self.semaphore`` — otherwise an in-flight room batch
    could starve a regen request that the user is actively waiting on.
    """

    @pytest.mark.asyncio
    async def test_regen_does_not_block_on_room_semaphore_at_zero(self):
        """If we drain the room-level semaphore to 0 capacity (so no
        room worker could acquire it), a single-variation regen still
        completes — proving it doesn't try to acquire that semaphore.
        """
        # Cap=1 then exhaust it: nothing else can hold this semaphore.
        room_semaphore = asyncio.Semaphore(1)
        await room_semaphore.acquire()  # exhaust

        staging = _build_staging_pipeline(semaphore=room_semaphore)
        project = _make_project(n_rooms=1, n_variations=2)
        room = project.rooms[0]
        variation = room.variations[0]

        # If process_single_variation tried to acquire self.semaphore,
        # it would block forever and wait_for would time out. Instead:
        events = []
        try:
            async with asyncio.timeout(2.0):
                async for event in staging.process_single_variation(
                    project, room, variation, "a fresh prompt"
                ):
                    events.append(event)
        except asyncio.TimeoutError:
            pytest.fail(
                "process_single_variation blocked on the room-level "
                "semaphore — it should not acquire that semaphore."
            )

        # And it really did the work:
        assert any(
            e["type"] in ("variation_completed", "variation_failed")
            for e in events
        )


# ---------------------------------------------------------------------------
# Cancellation & no-zombie-write
# ---------------------------------------------------------------------------


class TestClientDisconnectCancelsVariations:
    """When the SSE consumer disconnects mid-job, in-flight variation
    tasks are cancelled cleanly. After the cancellation point:

    * No further ``update_project`` writes are issued for the cancelled
      variations (no "zombie write" carrying their completion outcome).
    * No ``variation_completed`` / ``variation_failed`` events are
      emitted for the cancelled variations.
    """

    @pytest.mark.asyncio
    async def test_disconnect_no_completion_events_for_cancelled_variations(
        self,
    ):
        gated = _GatedProcessPipeline(target_arrivals=3)
        mock_pipeline = MagicMock()
        mock_pipeline.process_pipeline = gated

        # Storage stub: track total update_project calls.
        storage = MagicMock()
        update_calls: List[Dict[str, Any]] = []

        def _record_update(project_id: str, updates: Dict[str, Any]) -> None:
            # Snapshot the variation statuses at write time.
            rooms_state = []
            for r in updates.get("rooms", []):
                rooms_state.append(
                    {
                        "room_id": r.get("id"),
                        "variations": [
                            (v.get("id"), v.get("status"))
                            for v in r.get("variations", [])
                        ],
                    }
                )
            update_calls.append(
                {"project_id": project_id, "rooms_state": rooms_state}
            )

        storage.update_project = MagicMock(side_effect=_record_update)

        staging = _build_staging_pipeline(
            image_pipeline=mock_pipeline,
            storage=storage,
        )
        project = _make_project(n_rooms=1, n_variations=3)

        events: List[Dict[str, Any]] = []

        async def consume() -> None:
            async for event in staging.generate_project(project):
                events.append(event)

        task = asyncio.create_task(consume())

        # Wait until all 3 variation workers have entered process_pipeline.
        await asyncio.wait_for(gated.all_arrived.wait(), timeout=2.0)

        # Snapshot writes-so-far before disconnect.
        writes_before_cancel = len(update_calls)
        events_before_cancel = list(events)

        # Disconnect: cancel the consumer.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Let any cleanup tasks settle.
        await asyncio.sleep(0.05)

        # No NEW variation_completed / variation_failed events were
        # emitted between cancellation and cleanup.
        new_events = events[len(events_before_cancel) :]
        new_var_events = [
            e
            for e in new_events
            if e["type"] in ("variation_completed", "variation_failed")
        ]
        assert new_var_events == [], (
            f"Cancelled variations emitted completion events after "
            f"disconnect: {new_var_events}"
        )

        # No write was ever issued that records a variation in the
        # COMPLETED state (cancelled variations must not "complete"
        # in Cosmos after the cancel).
        for write in update_calls:
            for room_state in write["rooms_state"]:
                for var_id, status in room_state["variations"]:
                    assert status != ItemStatus.COMPLETED.value, (
                        f"Cancelled variation {var_id} was written as "
                        f"COMPLETED to Cosmos — zombie write."
                    )

    @pytest.mark.asyncio
    async def test_disconnect_does_not_leak_tasks(self):
        """All tasks spawned by the staging pipeline (room workers and
        their child variation workers) must terminate cleanly after a
        consumer disconnect. No pending tasks should remain.
        """
        gated = _GatedProcessPipeline(target_arrivals=3)
        mock_pipeline = MagicMock()
        mock_pipeline.process_pipeline = gated

        staging = _build_staging_pipeline(image_pipeline=mock_pipeline)
        project = _make_project(n_rooms=1, n_variations=3)

        async def consume() -> None:
            async for _event in staging.generate_project(project):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(gated.all_arrived.wait(), timeout=2.0)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Allow cleanup to drain.
        await asyncio.sleep(0.1)

        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t is not asyncio.current_task()
        ]
        assert pending == [], (
            f"After disconnect, pending tasks remain: {pending}"
        )


# ---------------------------------------------------------------------------
# Sanity: existing room-fanout behavior is preserved
# ---------------------------------------------------------------------------


class TestExistingRoomFanoutPreserved:
    """Slice 005's room-level concurrency must continue to work
    end-to-end after the variation fan-out refactor: a 3×2 project
    yields 6 ``variation_completed`` events and a project_completed
    final event, project status COMPLETED.
    """

    @pytest.mark.asyncio
    async def test_three_rooms_two_variations_completes(self):
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=2)

        events: List[Dict[str, Any]] = []
        async for event in staging.generate_project(project):
            events.append(event)

        completed = [e for e in events if e["type"] == "variation_completed"]
        assert len(completed) == 6
        room_completed = [e for e in events if e["type"] == "room_completed"]
        assert len(room_completed) == 3
        assert events[-1]["type"] == "project_completed"
        assert project.status == ProjectStatus.COMPLETED
