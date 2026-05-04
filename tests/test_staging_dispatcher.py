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


def test_generate_project_kind_routes_to_project_dispatcher(monkeypatch):
    """Issue 005 wires the ``generate_project`` branch.

    The kind-switch must call ``generate_project_dispatcher`` for
    ``kind="generate_project"`` (replacing the issue-003 placeholder
    that raised ``Unknown kind: generate_project``).
    """
    from backend.core import staging_dispatcher as mod

    captured: dict = {}

    async def _fake_project(job, is_cancelled):
        captured["job_id"] = job["id"]
        captured["called"] = True
        return {"project_id": job["project_id"], "status": "completed"}

    monkeypatch.setattr(mod, "generate_project_dispatcher", _fake_project)

    job = _make_job_doc(kind="generate_project")
    result = asyncio.run(mod.staging_dispatcher(job, is_cancelled=lambda: False))

    assert captured.get("called") is True
    assert captured.get("job_id") == job["id"]
    assert result == {"project_id": "p1", "status": "completed"}


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


# ---------------------------------------------------------------------------
# generate_project_dispatcher (issue 005)
# ---------------------------------------------------------------------------


def _make_project_for_generation(
    *,
    project_id: str = "p1",
    rooms: list[dict] | None = None,
) -> dict:
    """Construct a multi-room project shape for generate_project tests.

    Variations carry image_url / thumb_url / md_url so the
    regenerate_all blob-cleanup path has something to walk.
    """
    if rooms is None:
        rooms = [
            {
                "id": "r1",
                "label": "Living Room",
                "original_image_url": "https://blob/orig1.png",
                "status": "pending",
                "variations": [
                    {
                        "id": "v1",
                        "status": "pending",
                        "image_url": "https://blob/r1v1.png",
                        "thumb_url": "https://blob/r1v1-thumb.webp",
                        "md_url": "https://blob/r1v1-md.webp",
                    },
                    {
                        "id": "v2",
                        "status": "pending",
                        "image_url": "https://blob/r1v2.png",
                        "thumb_url": "https://blob/r1v2-thumb.webp",
                        "md_url": "https://blob/r1v2-md.webp",
                    },
                ],
            },
        ]
    return {
        "id": project_id,
        "name": "Test Project",
        "prompt": "A modern interior",
        "doc_type": "project",
        "_etag": "abc",
        "rooms": rooms,
        "settings": {
            "model": "gpt-image-1",
            "size": "1024x1024",
            "quality": "high",
            "variations_per_room": 2,
        },
    }


def _make_project_job_doc(
    *,
    project_id: str = "p1",
    payload: dict | None = None,
) -> dict:
    """Construct a project-level job doc."""
    return {
        "id": f"{project_id}:proj:0",
        "project_id": project_id,
        "kind": "generate_project",
        "status": "running",
        "progress": 0,
        "phase": None,
        "attempts": 1,
        "payload": payload if payload is not None else {},
        "result": None,
        "error": None,
        "cancel_requested": False,
        "created_at": "2026-05-03T00:00:00Z",
        "updated_at": "2026-05-03T00:00:00Z",
    }


def _configure_project_mocks(
    *,
    project_data: dict | None = None,
    pipeline_result: dict | None = None,
    pipeline_raises: BaseException | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Wire mock storage + pipeline + store factories. Returns (storage, pipeline, store).

    pipeline.generate_project_for_job is an AsyncMock that returns
    ``pipeline_result`` (default ``{"project_id": ..., "status": "completed"}``)
    or raises ``pipeline_raises``.
    pipeline._persist_project_locked is an AsyncMock returning {}.
    pipeline._schedule_blob_cleanup is a regular Mock.
    """
    from unittest.mock import AsyncMock

    from backend.core.staging_dispatcher import configure_dispatcher_dependencies

    storage = MagicMock()
    storage.get_project.return_value = project_data

    pipeline = MagicMock()
    pipeline._persist_project_locked = AsyncMock(return_value={})
    pipeline._schedule_blob_cleanup = MagicMock()

    if pipeline_raises is not None:
        pipeline.generate_project_for_job = AsyncMock(side_effect=pipeline_raises)
    else:
        pipeline.generate_project_for_job = AsyncMock(
            return_value=pipeline_result
            or {"project_id": "p1", "status": "completed"}
        )

    store = MagicMock()
    # Default: update_job is a no-op success
    store.update_job.return_value = None

    configure_dispatcher_dependencies(
        storage_factory=lambda: storage,
        pipeline_factory=lambda: pipeline,
        store_factory=lambda: store,
    )
    return storage, pipeline, store


def test_generate_project_dispatcher_missing_store_factory_raises_runtime_error():
    """A missing store_factory surfaces at first dispatch with a
    RuntimeError pointing at configure_dispatcher_dependencies.
    """
    from backend.core.staging_dispatcher import (
        configure_dispatcher_dependencies,
        generate_project_dispatcher,
    )

    # Wire only storage + pipeline; deliberately omit store_factory.
    storage = MagicMock()
    storage.get_project.return_value = _make_project_for_generation()
    pipeline = MagicMock()
    configure_dispatcher_dependencies(
        storage_factory=lambda: storage,
        pipeline_factory=lambda: pipeline,
    )

    job = _make_project_job_doc()
    with pytest.raises(RuntimeError, match="configure_dispatcher_dependencies"):
        asyncio.run(generate_project_dispatcher(job, is_cancelled=lambda: False))


def test_generate_project_dispatcher_acquires_lease_and_calls_pipeline(monkeypatch):
    """Happy path: acquire returns True → calls generate_project_for_job
    → returns its result."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    storage, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
        pipeline_result={"project_id": "p1", "status": "completed"},
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc()
    result = asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    acquire.assert_called_once()
    pipeline.generate_project_for_job.assert_called_once()
    assert result == {"project_id": "p1", "status": "completed"}


def test_generate_project_dispatcher_lease_busy_then_acquired_polls_then_succeeds(
    monkeypatch,
):
    """acquire returns False then True → dispatcher polls until acquired."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    storage, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    # First two attempts fail; third succeeds.
    acquire = MagicMock(side_effect=[False, False, True])
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)
    # Tiny poll interval so the test runs fast.
    monkeypatch.setattr(mod, "LEASE_POLL_INTERVAL_SECONDS", 0.001)

    job = _make_project_job_doc()
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    assert acquire.call_count == 3
    pipeline.generate_project_for_job.assert_called_once()


def test_generate_project_dispatcher_lease_timeout_raises_lease_busy(monkeypatch):
    """acquire returns False forever → eventually raises LeaseBusy."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    storage, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=False)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)
    monkeypatch.setattr(mod, "LEASE_POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(mod, "LEASE_MAX_WAIT_SECONDS", 0.005)

    job = _make_project_job_doc()
    with pytest.raises(mod.LeaseBusy):
        asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    pipeline.generate_project_for_job.assert_not_called()
    # Lease was never acquired → never released.
    release.assert_not_called()


def test_generate_project_dispatcher_cancel_during_lease_wait_raises_job_cancelled(
    monkeypatch,
):
    """is_cancelled() during the poll-acquire wait → raise JobCancelled
    immediately. No pipeline call. No lease release (never acquired)."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod
    from backend.core.job_worker import JobCancelled

    storage, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=False)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)
    monkeypatch.setattr(mod, "LEASE_POLL_INTERVAL_SECONDS", 0.001)

    cancel_calls = {"n": 0}

    def is_cancelled():
        cancel_calls["n"] += 1
        # First call (in the loop body, after 1st acquire fail): not cancelled
        # Second call (next iteration): cancelled
        return cancel_calls["n"] >= 2

    job = _make_project_job_doc()
    with pytest.raises(JobCancelled):
        asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=is_cancelled))

    pipeline.generate_project_for_job.assert_not_called()
    release.assert_not_called()


def test_generate_project_dispatcher_loads_project_after_acquiring_lease(monkeypatch):
    """Pin call ordering: storage.get_project must run AFTER the lease
    is successfully acquired. This eliminates the stale-snapshot race
    flagged by rubber-duck (load before acquire would return data
    while another job is mid-write)."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    storage, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    # Track call order via a parent MagicMock's mock_calls.
    parent = MagicMock()
    parent.attach_mock(acquire, "acquire")
    parent.attach_mock(storage.get_project, "get_project")

    job = _make_project_job_doc()
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    # The first method recorded must be acquire; get_project must come AFTER.
    names = [c[0] for c in parent.mock_calls]
    assert names[0] == "acquire", (
        f"acquire_project_lease must be called first; observed order: {names}"
    )
    assert "get_project" in names
    assert names.index("acquire") < names.index("get_project")


def test_generate_project_dispatcher_releases_lease_on_success(monkeypatch):
    """Lease released in finally on the happy path."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc()
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    release.assert_called_once()
    kwargs = release.call_args.kwargs
    assert kwargs["project_id"] == "p1"
    assert kwargs["job_id"] == job["id"]


def test_generate_project_dispatcher_releases_lease_on_pipeline_failure(monkeypatch):
    """Lease released in finally even if generate_project_for_job raises."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
        pipeline_raises=RuntimeError("boom"),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc()
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    release.assert_called_once()


def test_generate_project_dispatcher_releases_lease_on_job_cancelled(monkeypatch):
    """Lease released in finally on JobCancelled (which is re-raised)."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod
    from backend.core.job_worker import JobCancelled

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
        pipeline_raises=JobCancelled("user cancelled"),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc()
    with pytest.raises(JobCancelled):
        asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    release.assert_called_once()


def test_generate_project_dispatcher_regenerate_all_clears_blobs_and_resets_state(
    monkeypatch,
):
    """regenerate_all=true → schedule_blob_cleanup for every non-None
    image_url/thumb_url/md_url; clear those 3 fields; reset variations
    + rooms to PENDING; persist."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod
    from backend.models.staging import ItemStatus

    captured = {}

    async def capture_gen(project, *, brief_prompts, progress_callback, is_cancelled):
        # Snapshot the project state at the moment the pipeline is invoked,
        # so we can assert post-reset state without the pipeline mutating it.
        captured["project"] = project
        return {"project_id": project.id, "status": "completed"}

    storage, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    pipeline.generate_project_for_job.side_effect = capture_gen

    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc(payload={"regenerate_all": True})
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    # 2 variations × 3 URLs each = 6 blob-cleanup schedules.
    assert pipeline._schedule_blob_cleanup.call_count == 6
    cleanup_urls = {c.args[0] for c in pipeline._schedule_blob_cleanup.call_args_list}
    assert "https://blob/r1v1.png" in cleanup_urls
    assert "https://blob/r1v1-thumb.webp" in cleanup_urls
    assert "https://blob/r1v1-md.webp" in cleanup_urls
    assert "https://blob/r1v2.png" in cleanup_urls

    # _persist_project_locked is called once after the destructive reset
    # (and possibly more if generate_project_for_job persists too — but the
    # mocked pipeline doesn't, so exactly 1).
    assert pipeline._persist_project_locked.call_count >= 1

    # The project handed to generate_project_for_job has the URLs cleared
    # and statuses reset.
    project = captured["project"]
    for room in project.rooms:
        assert room.status == ItemStatus.PENDING
        for v in room.variations:
            assert v.image_url is None
            assert v.thumb_url is None
            assert v.md_url is None
            assert v.status == ItemStatus.PENDING


def test_generate_project_dispatcher_default_does_not_clear_blobs(monkeypatch):
    """With regenerate_all absent (default false): no _schedule_blob_cleanup,
    no destructive reset, no pre-pipeline _persist_project_locked."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc(payload={})
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    pipeline._schedule_blob_cleanup.assert_not_called()


def test_generate_project_dispatcher_cancel_before_destructive_reset_raises_clean(
    monkeypatch,
):
    """If is_cancelled() fires BEFORE the destructive regenerate_all
    reset writes, raise JobCancelled WITHOUT scheduling any blob
    cleanups. The pre-reset cancel poll guards against destroying
    completed work just before a user cancel."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod
    from backend.core.job_worker import JobCancelled

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc(payload={"regenerate_all": True})
    with pytest.raises(JobCancelled):
        asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: True))

    pipeline._schedule_blob_cleanup.assert_not_called()
    pipeline.generate_project_for_job.assert_not_called()
    release.assert_called_once()


def test_generate_project_dispatcher_brief_prompts_dict_passed_verbatim(monkeypatch):
    """payload.brief_prompts (a dict) is passed verbatim into
    generate_project_for_job. This pins the brief-reuse-on-retry
    contract: the POST handler computes brief_prompts ONCE; redeliveries
    do NOT recompose."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    bp = {"r1": ["prompt-1", "prompt-2"]}
    job = _make_project_job_doc(payload={"brief_prompts": bp})
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    kwargs = pipeline.generate_project_for_job.call_args.kwargs
    assert kwargs["brief_prompts"] is bp


def test_generate_project_dispatcher_brief_prompts_none_passed_verbatim(monkeypatch):
    """When payload has no brief_prompts: None passed verbatim. The
    dispatcher MUST NOT coerce to {} — generate_project_for_job owns
    the None→legacy-compute fallback."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc(payload={})
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    kwargs = pipeline.generate_project_for_job.call_args.kwargs
    assert kwargs["brief_prompts"] is None


def test_generate_project_dispatcher_progress_callback_writes_phase_to_jobstore(
    monkeypatch,
):
    """progress_callback maps each event['type'] to
    JobStore.update_job(job_id, project_id, phase=...)."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod

    captured_calls: list[dict] = []

    async def fire_events(project, *, brief_prompts, progress_callback, is_cancelled):
        progress_callback({"type": "room_started", "room_id": "r1"})
        progress_callback({"type": "variation_completed", "room_id": "r1"})
        progress_callback({"type": "project_completed", "status": "completed"})
        return {"project_id": project.id, "status": "completed"}

    _, pipeline, store = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    pipeline.generate_project_for_job.side_effect = fire_events
    store.update_job.side_effect = lambda *args, **kwargs: captured_calls.append(
        {"args": args, "kwargs": kwargs}
    )

    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc()
    asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    phases = [
        c["kwargs"].get("phase")
        for c in captured_calls
        if "phase" in c["kwargs"]
    ]
    assert phases == ["room_started", "variation_completed", "project_completed"]
    # Each call uses the correct (job_id, project_id) tuple.
    for call in captured_calls:
        assert call["args"][0] == job["id"]
        assert call["args"][1] == "p1"


def test_generate_project_dispatcher_cancel_mid_flight_reverts_processing_preserves_completed(
    monkeypatch,
):
    """JobCancelled mid-flight: revert PROCESSING → PENDING; preserve
    COMPLETED variations (cancel-during-image-edit edge); revert rooms
    whose pipeline-set FAILED+error='cancelled' to PENDING; recompute
    project status; persist; re-raise."""
    import backend.core.staging_dispatcher as mod
    import backend.core.project_lease as lease_mod
    from backend.core.job_worker import JobCancelled
    from backend.models.staging import ItemStatus

    async def mutate_then_cancel(
        project, *, brief_prompts, progress_callback, is_cancelled
    ):
        # Simulate the pipeline mid-run: variation 0 was in flight,
        # variation 1 finished before cancel arrived. Room 1 was
        # cancelled-failed by the room_worker exception handler.
        project.rooms[0].variations[0].status = ItemStatus.PROCESSING
        project.rooms[0].variations[0].error = "in flight when cancelled"
        project.rooms[0].variations[1].status = ItemStatus.COMPLETED
        project.rooms[0].status = ItemStatus.FAILED
        project.rooms[0].error = "cancelled"
        raise JobCancelled()

    _, pipeline, _ = _configure_project_mocks(
        project_data=_make_project_for_generation(),
    )
    pipeline.generate_project_for_job.side_effect = mutate_then_cancel
    # Capture what _persist_project_locked sees on cleanup.
    persisted: list = []

    async def capture_persist(project):
        # Snapshot mutable refs at call time.
        persisted.append(
            [
                {
                    "room_status": r.status,
                    "room_error": r.error,
                    "variations": [
                        {"status": v.status, "error": v.error}
                        for v in r.variations
                    ],
                }
                for r in project.rooms
            ]
        )
        return {}

    pipeline._persist_project_locked.side_effect = capture_persist

    acquire = MagicMock(return_value=True)
    release = MagicMock(return_value=True)
    monkeypatch.setattr(lease_mod, "acquire_project_lease", acquire)
    monkeypatch.setattr(lease_mod, "release_project_lease", release)

    job = _make_project_job_doc()
    with pytest.raises(JobCancelled):
        asyncio.run(mod.generate_project_dispatcher(job, is_cancelled=lambda: False))

    # _persist_project_locked called at least once after the cleanup.
    assert pipeline._persist_project_locked.call_count >= 1
    final_state = persisted[-1]

    # Variation 0: PROCESSING → PENDING; error cleared.
    assert final_state[0]["variations"][0]["status"] == ItemStatus.PENDING
    assert final_state[0]["variations"][0]["error"] is None
    # Variation 1: COMPLETED preserved.
    assert final_state[0]["variations"][1]["status"] == ItemStatus.COMPLETED
    # Room: FAILED+error="cancelled" → PENDING; error cleared.
    assert final_state[0]["room_status"] == ItemStatus.PENDING
    assert final_state[0]["room_error"] is None

    release.assert_called_once()
