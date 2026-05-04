"""Unit tests for ``backend.core.staging_dispatcher``.

Public-interface contract pinned by these tests (per PRD § Worker
dispatcher + issue 003 AC):

  - ``staging_dispatcher(job, is_cancelled)`` is a kind-switch entry
    point conforming to the ``JobWorker.Dispatcher`` contract.
  - ``kind="regenerate_variation"`` → routes to
    ``regenerate_variation_dispatcher(job, is_cancelled)``.
  - ``kind="generate_project"`` → falls through to the unknown-kind
    branch and raises ``ValueError("Unknown kind: generate_project")``.
    Issue 005 visibly replaces this placeholder.
  - Any other kind (including missing) → ``ValueError("Unknown
    kind: <kind>")``.
  - ``regenerate_variation_dispatcher`` loads the project from a
    storage factory configured via
    ``configure_dispatcher_dependencies``, finds the room and
    variation, calls ``pipeline.process_single_variation``, and
    returns the final yielded event as the dispatch result.
  - Missing project / room / variation raise ``ValueError`` with a
    helpful message (the worker treats raised exceptions as job
    failures via the existing JobWorker state machine).
  - ``payload["adapted_prompt"]`` takes precedence over
    ``variation.generation_metadata.adapted_prompt`` (retry path).
  - With neither prompt available the dispatcher raises
    ``ValueError`` rather than silently calling the LLM.
  - ``is_cancelled()`` polled after stream completion raises
    ``JobCancelled`` so the worker routes the message to ``complete``
    (drop) rather than ``abandon`` (retry).
  - When dependencies are not configured, the first dispatcher call
    raises ``RuntimeError`` with a message pointing at
    ``configure_dispatcher_dependencies`` rather than blowing up
    deep inside the dispatcher.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_job_doc(
    *,
    kind: str = "regenerate_variation",
    project_id: str = "p1",
    room_id: str = "r1",
    variation_id: str = "v1",
    payload: dict | None = None,
) -> dict:
    """Construct a minimal job doc that the dispatcher reads from."""
    return {
        "id": f"{project_id}:{room_id}:{variation_id}:0",
        "project_id": project_id,
        "room_id": room_id,
        "variation_id": variation_id,
        "revision": 0,
        "kind": kind,
        "status": "running",
        "progress": 0,
        "phase": None,
        "attempts": 1,
        "payload": payload if payload is not None else {
            "room_id": room_id,
            "variation_id": variation_id,
            "revision": 0,
        },
        "result": None,
        "error": None,
        "cancel_requested": False,
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:00:00Z",
    }


def _make_project_data(
    *,
    project_id: str = "p1",
    room_id: str = "r1",
    variation_id: str = "v1",
    adapted_prompt: str | None = "prior prompt",
) -> dict:
    """Construct a minimal storage-shaped project dict that
    ``StagingProject(**clean)`` accepts.
    """
    variation: dict[str, Any] = {
        "id": variation_id,
        "status": "completed",
        "image_url": "https://blob/old.png",
    }
    if adapted_prompt is not None:
        variation["generation_metadata"] = {
            "model": "gpt-image-1",
            "adapted_prompt": adapted_prompt,
        }

    return {
        "id": project_id,
        "name": "Test Project",
        "prompt": "A modern living room",
        "doc_type": "project",
        "_etag": "abc",
        "rooms": [
            {
                "id": room_id,
                "label": "Living Room",
                "original_image_url": "https://blob/orig.png",
                "status": "completed",
                "variations": [variation],
            }
        ],
        "settings": {
            "model": "gpt-image-1",
            "size": "1024x1024",
            "quality": "high",
            "variations_per_room": 1,
        },
    }


@pytest.fixture(autouse=True)
def _reset_dispatcher_dependencies():
    """Reset module-level dependency factories between tests so cross-
    test leakage cannot mask a missing-config bug.
    """
    from backend.core.staging_dispatcher import reset_dispatcher_dependencies

    reset_dispatcher_dependencies()
    yield
    reset_dispatcher_dependencies()


def _configure_with_mocks(
    *,
    project_data: dict | None = None,
    pipeline_events: list[dict] | None = None,
    pipeline_raises: BaseException | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Wire mock storage + pipeline factories. Returns (storage, pipeline)."""
    from backend.core.staging_dispatcher import configure_dispatcher_dependencies

    storage = MagicMock()
    storage.get_project.return_value = project_data

    pipeline = MagicMock()

    async def _fake_stream(*args, **kwargs):
        if pipeline_raises is not None:
            raise pipeline_raises
        for event in pipeline_events or [{"type": "variation_completed"}]:
            yield event

    pipeline.process_single_variation.side_effect = _fake_stream

    configure_dispatcher_dependencies(
        storage_factory=lambda: storage,
        pipeline_factory=lambda: pipeline,
    )
    return storage, pipeline


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_kind_switch_and_variation_dispatcher():
    """The two public callables exist with the documented async shape."""
    from backend.core.staging_dispatcher import (
        regenerate_variation_dispatcher,
        staging_dispatcher,
    )

    assert asyncio.iscoroutinefunction(staging_dispatcher)
    assert asyncio.iscoroutinefunction(regenerate_variation_dispatcher)


# ---------------------------------------------------------------------------
# Kind-switch routing
# ---------------------------------------------------------------------------


def test_unknown_kind_raises_value_error():
    """Any kind not in the allow-list raises ValueError with the kind in the message."""
    from backend.core.staging_dispatcher import staging_dispatcher

    job = _make_job_doc(kind="not_a_real_kind")

    with pytest.raises(ValueError, match="Unknown kind: not_a_real_kind"):
        asyncio.run(staging_dispatcher(job, is_cancelled=lambda: False))


def test_missing_kind_raises_value_error():
    """A job without ``kind`` is also unknown."""
    from backend.core.staging_dispatcher import staging_dispatcher

    job = _make_job_doc()
    job.pop("kind")

    with pytest.raises(ValueError, match="Unknown kind: None"):
        asyncio.run(staging_dispatcher(job, is_cancelled=lambda: False))


def test_generate_project_kind_raises_unknown_placeholder():
    """Issue 003 leaves ``generate_project`` in the unknown branch.

    Issue 005 fills in this branch with ``generate_project_dispatcher``.
    Pinning the placeholder error here means issue 005 visibly replaces
    it (this test will fail when 005 routes the kind, signaling that the
    placeholder is gone).
    """
    from backend.core.staging_dispatcher import staging_dispatcher

    job = _make_job_doc(kind="generate_project")

    with pytest.raises(ValueError, match="Unknown kind: generate_project"):
        asyncio.run(staging_dispatcher(job, is_cancelled=lambda: False))


def test_regenerate_variation_kind_routes_to_variation_dispatcher(monkeypatch):
    """``kind=regenerate_variation`` is routed to the variation dispatcher.

    Mocks the variation dispatcher inside the module so this test pins
    the kind-switch wiring in isolation from the variation dispatcher's
    own behaviour.
    """
    from backend.core import staging_dispatcher as mod

    captured: dict = {}

    async def _fake_variation(job, is_cancelled):
        captured["job_id"] = job["id"]
        captured["called"] = True
        return {"image_url": "ok"}

    monkeypatch.setattr(mod, "regenerate_variation_dispatcher", _fake_variation)

    job = _make_job_doc(kind="regenerate_variation")
    result = asyncio.run(mod.staging_dispatcher(job, is_cancelled=lambda: False))

    assert captured.get("called") is True
    assert captured.get("job_id") == job["id"]
    assert result == {"image_url": "ok"}


# ---------------------------------------------------------------------------
# Dependency configuration
# ---------------------------------------------------------------------------


def test_dispatcher_without_dependencies_raises_runtime_error_pointing_at_configure():
    """If a caller forgets to call ``configure_dispatcher_dependencies``,
    the first dispatcher invocation raises a clear RuntimeError instead
    of crashing deep inside the dispatcher.
    """
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    # No configure call made — fixture just reset everything.
    job = _make_job_doc(kind="regenerate_variation")

    with pytest.raises(RuntimeError, match="configure_dispatcher_dependencies"):
        asyncio.run(regenerate_variation_dispatcher(job, is_cancelled=lambda: False))


# ---------------------------------------------------------------------------
# regenerate_variation_dispatcher — happy path + lookups
# ---------------------------------------------------------------------------


def test_regenerate_variation_dispatcher_happy_path_returns_final_event():
    """The dispatcher loads the project, finds the room/variation, calls
    ``process_single_variation`` with the variation's prior adapted
    prompt, and returns the final yielded event.
    """
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(adapted_prompt="prior prompt")
    final_event = {
        "type": "variation_completed",
        "room_id": "r1",
        "variation_index": 0,
        "image_url": "https://blob/new.png",
    }
    storage, pipeline = _configure_with_mocks(
        project_data=project_data,
        pipeline_events=[
            {"type": "variation_started"},
            final_event,
        ],
    )

    job = _make_job_doc(kind="regenerate_variation")
    result = asyncio.run(
        regenerate_variation_dispatcher(job, is_cancelled=lambda: False)
    )

    storage.get_project.assert_called_once_with("p1")
    pipeline.process_single_variation.assert_called_once()
    kwargs = pipeline.process_single_variation.call_args.kwargs
    assert kwargs["room"].id == "r1"
    assert kwargs["variation"].id == "v1"
    assert kwargs["adapted_prompt"] == "prior prompt"
    assert result == final_event


def test_regenerate_variation_dispatcher_missing_project_raises():
    """A non-existent project surfaces as ValueError."""
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    _configure_with_mocks(project_data=None)

    job = _make_job_doc(kind="regenerate_variation", project_id="ghost")
    with pytest.raises(ValueError, match="Project not found: ghost"):
        asyncio.run(
            regenerate_variation_dispatcher(job, is_cancelled=lambda: False)
        )


def test_regenerate_variation_dispatcher_missing_room_raises():
    """A room id not present on the project surfaces as ValueError."""
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(room_id="r1")
    _configure_with_mocks(project_data=project_data)

    job = _make_job_doc(kind="regenerate_variation", room_id="r_missing")
    with pytest.raises(ValueError, match="Room not found: r_missing"):
        asyncio.run(
            regenerate_variation_dispatcher(job, is_cancelled=lambda: False)
        )


def test_regenerate_variation_dispatcher_missing_variation_raises():
    """A variation id not present on the room surfaces as ValueError."""
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(variation_id="v1")
    _configure_with_mocks(project_data=project_data)

    job = _make_job_doc(kind="regenerate_variation", variation_id="v_missing")
    with pytest.raises(ValueError, match="Variation not found: v_missing"):
        asyncio.run(
            regenerate_variation_dispatcher(job, is_cancelled=lambda: False)
        )


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------


def test_payload_adapted_prompt_takes_precedence_over_metadata():
    """If the POST handler stashes a prompt in ``payload``, the dispatcher
    uses it verbatim instead of reading ``variation.generation_metadata``.
    """
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(adapted_prompt="from metadata")
    _, pipeline = _configure_with_mocks(
        project_data=project_data,
        pipeline_events=[{"type": "variation_completed"}],
    )

    job = _make_job_doc(
        kind="regenerate_variation",
        payload={
            "room_id": "r1",
            "variation_id": "v1",
            "revision": 0,
            "adapted_prompt": "from payload",
        },
    )
    asyncio.run(regenerate_variation_dispatcher(job, is_cancelled=lambda: False))

    kwargs = pipeline.process_single_variation.call_args.kwargs
    assert kwargs["adapted_prompt"] == "from payload"


def test_metadata_adapted_prompt_used_for_retry_when_payload_omits_it():
    """Retry path: when payload has no ``adapted_prompt``, fall back to
    the variation's prior ``generation_metadata.adapted_prompt``.
    """
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(adapted_prompt="prior retry prompt")
    _, pipeline = _configure_with_mocks(
        project_data=project_data,
        pipeline_events=[{"type": "variation_completed"}],
    )

    job = _make_job_doc(
        kind="regenerate_variation",
        payload={"room_id": "r1", "variation_id": "v1", "revision": 0},
    )
    asyncio.run(regenerate_variation_dispatcher(job, is_cancelled=lambda: False))

    kwargs = pipeline.process_single_variation.call_args.kwargs
    assert kwargs["adapted_prompt"] == "prior retry prompt"


def test_no_prompt_available_raises_value_error():
    """No payload prompt and no prior metadata → fail loudly rather than
    quietly call adapt_prompt (which would change behavior vs the
    legacy POST endpoint and cost an LLM round-trip).
    """
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(adapted_prompt=None)
    _configure_with_mocks(project_data=project_data)

    job = _make_job_doc(
        kind="regenerate_variation",
        payload={"room_id": "r1", "variation_id": "v1", "revision": 0},
    )
    with pytest.raises(ValueError, match="adapted_prompt"):
        asyncio.run(
            regenerate_variation_dispatcher(job, is_cancelled=lambda: False)
        )


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_is_cancelled_after_stream_raises_job_cancelled():
    """If ``is_cancelled()`` is True after the pipeline stream completes,
    the dispatcher raises ``JobCancelled`` so the worker routes the
    message to ``complete`` (drop) rather than ``abandon`` (retry).
    """
    from backend.core.job_worker import JobCancelled
    from backend.core.staging_dispatcher import regenerate_variation_dispatcher

    project_data = _make_project_data(adapted_prompt="p")
    _configure_with_mocks(
        project_data=project_data,
        pipeline_events=[{"type": "variation_completed"}],
    )

    job = _make_job_doc(kind="regenerate_variation")
    with pytest.raises(JobCancelled):
        asyncio.run(
            regenerate_variation_dispatcher(job, is_cancelled=lambda: True)
        )
