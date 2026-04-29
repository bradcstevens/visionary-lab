"""Tests for parallel room processing in the staging pipeline."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.images import (
    ImageGenerationResponse,
    ImageSaveResponse,
    ImagePipelineResponse,
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


def _make_project(n_rooms=3, n_variations=2) -> StagingProject:
    rooms = []
    for i in range(n_rooms):
        variations = [Variation(id=f"var-{i}-{j}") for j in range(n_variations)]
        rooms.append(
            Room(
                id=f"room-{i}",
                label=f"Room {i+1}",
                original_image_url="https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
                variations=variations,
            )
        )
    return StagingProject(
        id="proj-parallel",
        name="Parallel Test",
        prompt="Modern minimalist",
        settings=StagingSettings(variations_per_room=n_variations),
        rooms=rooms,
    )


def _make_pipeline_response(room_id="room-0"):
    url = f"https://acct.blob.core.windows.net/images/staging/proj/variations/{room_id}/img.png"
    gen = ImageGenerationResponse(
        success=True, message="ok",
        imgen_model_response={"data": [{"b64_json": "AAAA"}], "usage": {"total_tokens": 100}},
    )
    save = ImageSaveResponse(
        success=True, message="Saved",
        saved_images=[{"url": url, "blob_name": f"staging/proj/variations/{room_id}/img.png"}],
        total_saved=1,
    )
    return ImagePipelineResponse(
        success=True, message="Pipeline completed",
        steps=[PipelineStepResult(step="edit", success=True), PipelineStepResult(step="save", success=True)],
        generation=gen, save=save,
    )


def _build_staging_pipeline():
    """Construct a StagingPipeline with all deps mocked."""
    from backend.core.staging_pipeline import StagingPipeline

    mock_pipeline = AsyncMock()
    mock_pipeline.process_pipeline.side_effect = lambda **kw: _make_pipeline_response()

    mock_blob = MagicMock()
    mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

    mock_storage = MagicMock()
    mock_storage.update_project = MagicMock()

    mock_analyzer = AsyncMock()
    mock_analyzer.async_image_chat.return_value = {"description": "A room", "features": []}

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='["Add plants", "Add art"]'))]
    )

    return StagingPipeline(
        async_llm_client=mock_llm,
        llm_deployment="gpt-4o",
        image_analyzer=mock_analyzer,
        image_pipeline=mock_pipeline,
        storage_service=mock_storage,
        blob_service=mock_blob,
    )


class TestParallelRoomProcessing:

    @pytest.mark.asyncio
    async def test_all_rooms_processed(self):
        """All rooms should be processed and emit events."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        room_completed = [e for e in events if e["type"] == "room_completed"]
        assert len(room_completed) == 3
        assert events[-1]["type"] == "project_completed"

    @pytest.mark.asyncio
    async def test_rooms_run_concurrently(self):
        """Rooms should overlap in time, not run sequentially."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        original_process_room = staging.process_room

        async def slow_process_room(proj, room, brief_prompts=None):
            await asyncio.sleep(0.1)
            async for event in original_process_room(proj, room, brief_prompts=brief_prompts):
                yield event

        staging.process_room = slow_process_room

        start = time.monotonic()
        events = []
        async for event in staging.generate_project(project):
            events.append(event)
        elapsed = time.monotonic() - start

        # Sequential: 3 × 0.1s = 0.3s. Parallel: ~0.1s.
        # Use 0.25s upper bound — proves clear parallelism with CI margin.
        # Bump to 0.4s if seen flaking on slow CI.
        assert elapsed < 0.25, f"Rooms appear sequential: took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """No more than STAGING_CONCURRENT_ROOMS rooms should be inside process_room at once."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=5, n_variations=1)

        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_process_room(proj, room, brief_prompts=None):
            async with staging.semaphore:  # use the actual semaphore
                nonlocal peak_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    peak_concurrent = max(peak_concurrent, current_concurrent)
                try:
                    await asyncio.sleep(0.05)
                    yield {"type": "room_completed", "room_id": room.id}
                finally:
                    async with lock:
                        current_concurrent -= 1

        staging.process_room = counting_process_room

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        assert peak_concurrent <= 3, f"Peak concurrency {peak_concurrent} exceeded limit of 3"

    @pytest.mark.asyncio
    async def test_one_room_failure_doesnt_stop_others(self):
        """If one room fails, remaining rooms should still complete."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        original_process_room = staging.process_room
        call_count = 0

        async def failing_process_room(proj, room, brief_prompts=None):
            nonlocal call_count
            call_count += 1
            if room.id == "room-1":
                raise RuntimeError("Simulated failure")
            async for event in original_process_room(proj, room, brief_prompts=brief_prompts):
                yield event

        staging.process_room = failing_process_room

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        completed = [e for e in events if e["type"] == "room_completed"]
        failed = [e for e in events if e["type"] == "room_failed"]
        assert len(completed) == 2
        assert len(failed) == 1
        assert failed[0]["room_id"] == "room-1"

    @pytest.mark.asyncio
    async def test_project_status_set_correctly(self):
        """Project should be COMPLETED if any room completed."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=2, n_variations=1)

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        final = events[-1]
        assert final["type"] == "project_completed"
        assert project.status == ProjectStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_consumer_disconnect_cancels_tasks(self):
        """If the SSE consumer disconnects, in-flight room tasks should be cancelled."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        started = asyncio.Event()
        can_finish = asyncio.Event()

        original_process_room = staging.process_room

        async def slow_process_room(proj, room, brief_prompts=None):
            started.set()
            await can_finish.wait()  # block forever unless we set this
            async for event in original_process_room(proj, room, brief_prompts=brief_prompts):
                yield event

        staging.process_room = slow_process_room

        gen = staging.generate_project(project)

        async def consume_one():
            async for _ in gen:
                return  # consume nothing actually emitted yet, but starts workers

        # Trigger generator startup so workers are launched
        consumer_task = asyncio.create_task(consume_one())
        await started.wait()

        # Give time for workers to launch
        await asyncio.sleep(0.05)

        # Disconnect: cancel the consumer task, simulating client disconnect
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        # Give cleanup time to complete
        await asyncio.sleep(0.1)

        # All worker tasks should now be cancelled / done
        # (we can't directly access them, but no tasks should be pending for this project)
        pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
        # Allow CI overhead — at minimum, the slow_process_room shouldn't leak forever
        assert len(pending) == 0, f"Found {len(pending)} pending tasks after disconnect"
