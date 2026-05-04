"""Tests for ``StagingPipeline.generate_project_for_job``.

The queue-friendly sibling to ``generate_project``: same room +
variation orchestration but with a synchronous ``progress_callback``
instead of ``yield`` and an ``is_cancelled`` poll between events.

Tests mock ``process_room`` (the inner unit) so we focus on the new
orchestration logic — event dispatch, brief-prompt reuse, smart-skip,
cancellation — without re-testing what the existing
``test_staging_pipeline.py`` suite already covers.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.job_worker import JobCancelled
from backend.core.staging_pipeline import StagingPipeline
from backend.models.staging import (
    ItemStatus,
    Room,
    StagingProject,
    StagingSettings,
    Variation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_project(
    *,
    n_rooms: int = 1,
    n_variations: int = 2,
    room_statuses: list[ItemStatus] | None = None,
    design_brief: dict | None = None,
) -> StagingProject:
    rooms = []
    for i in range(n_rooms):
        variations = [Variation(id=f"var-{i}-{j}") for j in range(n_variations)]
        rooms.append(
            Room(
                id=f"room-{i}",
                label=f"Room {i+1}",
                original_image_url=f"https://example/room-{i}.png",
                variations=variations,
                status=room_statuses[i] if room_statuses else ItemStatus.PENDING,
            )
        )
    return StagingProject(
        id="proj-test",
        name="Test",
        prompt="Modern minimalist",
        settings=StagingSettings(variations_per_room=n_variations),
        rooms=rooms,
        design_brief=design_brief,
    )


def _make_pipeline() -> StagingPipeline:
    """Build a StagingPipeline with mocked dependencies, just enough
    to exercise generate_project_for_job."""
    pipeline = StagingPipeline(
        async_llm_client=MagicMock(),
        llm_deployment="test-deployment",
        image_analyzer=MagicMock(),
        image_pipeline=MagicMock(),
        storage_service=MagicMock(),
        blob_service=MagicMock(),
    )
    # Bypass the real Cosmos persist — we want fast, deterministic tests.
    pipeline._persist_project_locked = AsyncMock()
    return pipeline


def _async_gen(events: list[dict]):
    """Build an async generator that yields the given events. Used to
    stub process_room without spinning up real image / LLM calls."""

    async def _gen(*args, **kwargs):
        for event in events:
            yield event

    return _gen


def _never_cancelled() -> bool:
    return False


# ---------------------------------------------------------------------------
# Public surface — return shape, callback cadence, smart-skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_dict_with_project_id_and_status():
    pipeline = _make_pipeline()
    project = _make_project(room_statuses=[ItemStatus.COMPLETED])
    callback = MagicMock()

    result = await pipeline.generate_project_for_job(
        project,
        brief_prompts=None,
        progress_callback=callback,
        is_cancelled=_never_cancelled,
    )

    assert isinstance(result, dict)
    assert result["project_id"] == "proj-test"
    assert "status" in result


@pytest.mark.asyncio
async def test_progress_callback_invoked_for_every_room_event_plus_completion():
    pipeline = _make_pipeline()
    project = _make_project(n_rooms=1)
    events = [
        {"type": "room_started", "room_id": "room-0"},
        {"type": "variation_completed", "room_id": "room-0", "variation_index": 0},
        {"type": "variation_completed", "room_id": "room-0", "variation_index": 1},
        {"type": "room_completed", "room_id": "room-0"},
    ]
    pipeline.process_room = MagicMock(side_effect=_async_gen(events))
    callback = MagicMock()

    await pipeline.generate_project_for_job(
        project,
        brief_prompts={},
        progress_callback=callback,
        is_cancelled=_never_cancelled,
    )

    delivered = [c.args[0] for c in callback.call_args_list]
    assert events[0] in delivered
    assert events[1] in delivered
    assert events[2] in delivered
    assert events[3] in delivered
    assert any(e.get("type") == "project_completed" for e in delivered)


@pytest.mark.asyncio
async def test_smart_skip_only_processes_pending_or_failed_rooms():
    pipeline = _make_pipeline()
    project = _make_project(
        n_rooms=3,
        room_statuses=[ItemStatus.PENDING, ItemStatus.COMPLETED, ItemStatus.FAILED],
    )
    pipeline.process_room = MagicMock(side_effect=_async_gen([]))

    await pipeline.generate_project_for_job(
        project,
        brief_prompts={},
        progress_callback=MagicMock(),
        is_cancelled=_never_cancelled,
    )

    processed_room_ids = {
        call.args[1].id for call in pipeline.process_room.call_args_list
    }
    assert processed_room_ids == {"room-0", "room-2"}


@pytest.mark.asyncio
async def test_no_pending_rooms_emits_project_completed_only():
    pipeline = _make_pipeline()
    project = _make_project(
        n_rooms=2, room_statuses=[ItemStatus.COMPLETED, ItemStatus.COMPLETED]
    )
    pipeline.process_room = MagicMock()
    callback = MagicMock()

    await pipeline.generate_project_for_job(
        project,
        brief_prompts={},
        progress_callback=callback,
        is_cancelled=_never_cancelled,
    )

    pipeline.process_room.assert_not_called()
    delivered_types = [c.args[0].get("type") for c in callback.call_args_list]
    assert delivered_types == ["project_completed"]


# ---------------------------------------------------------------------------
# Brief-prompt reuse — the regression pin from rubber-duck non-blocking #2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brief_prompts_supplied_skips_brief_to_prompts_call():
    pipeline = _make_pipeline()
    project = _make_project(design_brief={"some": "brief"})
    pipeline.process_room = MagicMock(side_effect=_async_gen([]))

    with patch(
        "backend.core.brief_generator.BriefGeneratorService"
    ) as brief_cls:
        await pipeline.generate_project_for_job(
            project,
            brief_prompts={"room-0": ["precomputed-prompt-1", "precomputed-prompt-2"]},
            progress_callback=MagicMock(),
            is_cancelled=_never_cancelled,
        )
        brief_cls.assert_not_called()


@pytest.mark.asyncio
async def test_brief_prompts_none_with_design_brief_calls_brief_to_prompts():
    pipeline = _make_pipeline()
    project = _make_project(
        design_brief={
            "global_instructions": "Modern minimalist palette",
        }
    )
    pipeline.process_room = MagicMock(side_effect=_async_gen([]))

    with patch(
        "backend.core.brief_generator.BriefGeneratorService"
    ) as brief_cls:
        instance = brief_cls.return_value
        instance.brief_to_prompts = AsyncMock(return_value={"room-0": ["p1", "p2"]})

        await pipeline.generate_project_for_job(
            project,
            brief_prompts=None,
            progress_callback=MagicMock(),
            is_cancelled=_never_cancelled,
        )

        instance.brief_to_prompts.assert_awaited_once()


@pytest.mark.asyncio
async def test_brief_prompts_none_without_design_brief_skips_brief_to_prompts():
    pipeline = _make_pipeline()
    project = _make_project(design_brief=None)
    pipeline.process_room = MagicMock(side_effect=_async_gen([]))

    with patch(
        "backend.core.brief_generator.BriefGeneratorService"
    ) as brief_cls:
        await pipeline.generate_project_for_job(
            project,
            brief_prompts=None,
            progress_callback=MagicMock(),
            is_cancelled=_never_cancelled,
        )
        brief_cls.assert_not_called()


@pytest.mark.asyncio
async def test_supplied_brief_prompts_passed_to_process_room_verbatim():
    pipeline = _make_pipeline()
    project = _make_project()
    supplied = {"room-0": ["exact-prompt-1", "exact-prompt-2"]}
    pipeline.process_room = MagicMock(side_effect=_async_gen([]))

    await pipeline.generate_project_for_job(
        project,
        brief_prompts=supplied,
        progress_callback=MagicMock(),
        is_cancelled=_never_cancelled,
    )

    pipeline.process_room.assert_called_once()
    kwargs = pipeline.process_room.call_args.kwargs
    assert kwargs["brief_prompts"] is supplied


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_cancelled_true_before_drain_raises_job_cancelled():
    pipeline = _make_pipeline()
    project = _make_project()
    pipeline.process_room = MagicMock(side_effect=_async_gen([]))

    with pytest.raises(JobCancelled):
        await pipeline.generate_project_for_job(
            project,
            brief_prompts={},
            progress_callback=MagicMock(),
            is_cancelled=lambda: True,
        )


@pytest.mark.asyncio
async def test_is_cancelled_polled_mid_drain_raises_job_cancelled():
    pipeline = _make_pipeline()
    project = _make_project(n_rooms=1)
    events = [
        {"type": "room_started", "room_id": "room-0"},
        {"type": "variation_completed", "room_id": "room-0", "variation_index": 0},
        {"type": "variation_completed", "room_id": "room-0", "variation_index": 1},
        {"type": "room_completed", "room_id": "room-0"},
    ]
    pipeline.process_room = MagicMock(side_effect=_async_gen(events))

    cancel_state = {"polls": 0}

    def is_cancelled() -> bool:
        cancel_state["polls"] += 1
        return cancel_state["polls"] >= 3

    callback = MagicMock()

    with pytest.raises(JobCancelled):
        await pipeline.generate_project_for_job(
            project,
            brief_prompts={},
            progress_callback=callback,
            is_cancelled=is_cancelled,
        )

    delivered_types = [c.args[0].get("type") for c in callback.call_args_list]
    assert "project_completed" not in delivered_types


@pytest.mark.asyncio
async def test_is_cancelled_cancels_inflight_room_tasks():
    """When the cancel signal fires mid-drain, in-flight room workers
    must be cancelled — not left as zombies that keep emitting
    progress writes after the job has been marked cancelled."""
    pipeline = _make_pipeline()
    project = _make_project(n_rooms=2)

    cleanup_was_cancelled = {"flag": False}

    async def _slow_room(*args, **kwargs):
        try:
            yield {"type": "room_started", "room_id": kwargs.get("room", args[1]).id}
            await asyncio.sleep(10)
            yield {"type": "room_completed"}
        except asyncio.CancelledError:
            cleanup_was_cancelled["flag"] = True
            raise

    pipeline.process_room = MagicMock(side_effect=_slow_room)

    # Cancellation poll fires only AFTER the first event has been
    # drained — guarantees at least one room worker has started running
    # (past the first ``yield`` and into ``await asyncio.sleep``) so
    # the subsequent ``task.cancel()`` actually propagates a
    # CancelledError into the worker's await point.
    cancel_state = {"polls": 0}

    def is_cancelled() -> bool:
        cancel_state["polls"] += 1
        return cancel_state["polls"] >= 3

    with pytest.raises(JobCancelled):
        await pipeline.generate_project_for_job(
            project,
            brief_prompts={},
            progress_callback=MagicMock(),
            is_cancelled=is_cancelled,
        )

    assert cleanup_was_cancelled["flag"] is True


# ---------------------------------------------------------------------------
# Final status — via calculator (mirrors legacy generate_project)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_status_computed_via_project_status_calculator():
    """Mirror the legacy ``generate_project`` invariant: the final
    project status is delegated to ``ProjectStatusCalculator``, not
    a hard-coded value."""
    pipeline = _make_pipeline()
    project = _make_project(
        n_rooms=2,
        room_statuses=[ItemStatus.COMPLETED, ItemStatus.FAILED],
    )

    with patch(
        "backend.core.staging_pipeline.ProjectStatusCalculator"
    ) as calc:
        calc.compute_status = MagicMock(return_value="completed")

        await pipeline.generate_project_for_job(
            project,
            brief_prompts={},
            progress_callback=MagicMock(),
            is_cancelled=_never_cancelled,
        )

        calc.compute_status.assert_called_with(project.rooms)
