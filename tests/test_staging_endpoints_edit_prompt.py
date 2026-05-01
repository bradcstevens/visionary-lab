"""Tests for the per-variation Edit Prompt endpoint — issue 004 of the
``projects-page-improvements`` PRD.

Endpoint contract (asserted by tests below):

    POST /api/v1/staging/projects/{pid}/rooms/{rid}/variations/{vid}/edit-prompt
    Body shape: ``{adapted_prompt: str}``  (required, non-empty after strip)

    Streams SSE events through the existing event vocabulary (variation_*,
    project_completed, error) and emits a NEW structured-log family
    ``staging.variation_edit_prompt.{started,completed,failed}`` so log
    analytics can count Edit Prompt usage separately from regen usage.

Behavior contract:

  - Validates the source variation's existence (404 if missing). The
    source itself is NEVER mutated — Edit Prompt is an APPEND operation.
  - Composes the final prompt via ``PromptComposer.compose`` with
    ``variation_override=body.adapted_prompt`` and
    ``design_brief=None`` — bypassing ``BriefGeneratorService`` entirely
    per PRD § Solution → 4. The room's ``prompt_addendum`` is still
    applied (per PRD Edit Prompt semantic).
  - Appends a brand-new Variation to ``room.variations`` with a fresh
    UUID; the source variation is left byte-identical (the whole point
    of Edit Prompt — preserve the original for A/B comparison).
  - Hands the new variation + composed prompt to
    ``pipeline.process_single_variation``. That path's existing failure
    rollback restores prior state (which for an appended variation is
    PROCESSING) — the endpoint's finally block forces the new
    variation to FAILED on terminal failure so it doesn't strand.
  - Recomputes room.status and project.status via
    ``ProjectStatusCalculator`` in the finalizer (mirrors regen
    finalizer pattern).
  - The preflight read-append-write is wrapped in the per-project
    asyncio.Lock so concurrent worker writes can't clobber the new
    variation. The finalizer write is also wrapped (same pattern).
  - 422 on empty / whitespace-only / missing ``adapted_prompt``.
  - 409 if source variation is currently PROCESSING (mirrors the
    regen 409 mutex).
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------

PROJECT_ID = "proj-edit-prompt"
ROOM_ID = "room-1"
SOURCE_VARIATION_ID = "var-source"
SOURCE_PRIOR_PROMPT = "ORIGINAL prompt that produced var-source"


def _project_one_completed_variation(*, with_addendum: bool = False) -> dict:
    """One-room project with a single COMPLETED variation that has
    ``generation_metadata.adapted_prompt``. The Edit Prompt flow
    operates on this completed variation as the "source" the user
    clicked Edit on, and appends a NEW variation alongside it.
    """
    project = {
        "id": PROJECT_ID,
        "name": "Edit Prompt Test",
        "prompt": "modern minimalist",
        "status": "completed",
        "rooms": [
            {
                "id": ROOM_ID,
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/lr.png",
                "status": "completed",
                "variations": [
                    {
                        "id": SOURCE_VARIATION_ID,
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/source.png",
                        "generation_metadata": {
                            "model": "gpt-image-2",
                            "adapted_prompt": SOURCE_PRIOR_PROMPT,
                            "generation_time_ms": 5000,
                        },
                    },
                ],
            },
        ],
        "settings": {
            "variations_per_room": 1,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
        "analyses": [],
    }
    if with_addendum:
        project["rooms"][0]["prompt_addendum"] = "always in front of fence"
    return project


def _empty_async_gen(*_args, **_kwargs):
    """Async generator that yields nothing (pipeline mock for tests
    that don't care about pipeline outputs)."""
    if False:
        yield  # pragma: no cover


def _setup_replace_item_capture(mock_container) -> list:
    """Configure ``mock_container.replace_item`` so each call's body is
    deep-copied immediately. Returns a mutable list that grows by one
    deep-copied dict per call.

    Without this capture, ``replace_item.call_args_list`` aliases the
    shared dict that storage's read-modify-write mutates in place, so
    tests asserting on intermediate states (e.g. the preflight write
    before the pipeline rolls back) see the post-loop final state
    instead of the per-call snapshot.
    """
    import copy as _copy
    captured = []

    def _side_effect(item, body):
        captured.append(_copy.deepcopy(body))
        return body

    mock_container.replace_item.side_effect = _side_effect
    return captured


def _captured_replace_bodies(mock_container) -> list:
    """Deep-copies of replace_item bodies extracted from
    ``call_args_list``. Use only when ``_setup_replace_item_capture``
    is NOT in effect — for the latter, read the returned list
    directly. Kept for tests that don't need intermediate-state
    isolation (only inspect a single call_count or last body).
    """
    import copy as _copy
    bodies = []
    for call in mock_container.replace_item.call_args_list:
        body = call.kwargs.get("body") or call.args[1]
        bodies.append(_copy.deepcopy(body))
    return bodies


# ---------------------------------------------------------------------------
# Validation: empty / missing prompt → 422.
# ---------------------------------------------------------------------------


def test_edit_prompt_empty_string_returns_422(client, mock_staging_deps):
    """Empty ``adapted_prompt`` is a 422 — clients shouldn't send no-op
    edits. Pydantic validation runs BEFORE the endpoint touches storage,
    so no replace_item should fire."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_one_completed_variation()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": ""},
    )
    assert response.status_code == 422, response.text
    assert mock_container.replace_item.call_count == 0


def test_edit_prompt_whitespace_only_returns_422(client, mock_staging_deps):
    """Whitespace-only is also 422 — same rationale as empty."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_one_completed_variation()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": "   \n  \t  "},
    )
    assert response.status_code == 422, response.text
    assert mock_container.replace_item.call_count == 0


def test_edit_prompt_missing_field_returns_422(client, mock_staging_deps):
    """The body is required. Sending ``{}`` is 422."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_one_completed_variation()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={},
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# 404 paths: project / room / variation missing.
# ---------------------------------------------------------------------------


def test_edit_prompt_project_not_found_returns_404(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(
        status_code=404, message="Not found"
    )

    response = client.post(
        f"/api/v1/staging/projects/nope/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": "anything"},
    )
    assert response.status_code == 404


def test_edit_prompt_room_not_found_returns_404(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_one_completed_variation()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/no-such-room/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": "anything"},
    )
    assert response.status_code == 404


def test_edit_prompt_variation_not_found_returns_404(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_one_completed_variation()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/no-such-var/edit-prompt",
        json={"adapted_prompt": "anything"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 409: source variation currently PROCESSING.
# (Backend safety net — frontend menu only shows on completed variations,
# but a direct caller could still send a request mid-regen.)
# ---------------------------------------------------------------------------


def test_edit_prompt_returns_409_when_source_variation_is_processing(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    project_data["rooms"][0]["variations"][0]["status"] = "processing"
    mock_container.read_item.return_value = project_data

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": "anything"},
    )
    assert response.status_code == 409
    # The 409 must fire BEFORE any storage write — no preflight should
    # have appended a new variation.
    assert mock_container.replace_item.call_count == 0


# ---------------------------------------------------------------------------
# Tracer bullet: append-not-mutate semantics + composer is called with
# variation_override + design_brief=None.
# ---------------------------------------------------------------------------


def test_edit_prompt_appends_new_variation_preserving_source(client, mock_staging_deps):
    """The CORE contract: an Edit Prompt request appends a fresh
    Variation to ``room.variations``; the source variation and any
    sibling variations are byte-identical post-call.

    Verified by snapshotting the room.variations list pre-call
    (canonicalized via Pydantic so the Pydantic-added defaults like
    ``thumbnail_url=None`` aren't false-positive diffs) and asserting
    the post-call prefix [:initial_len] matches byte-for-byte
    (rubber-duck-flagged: catches "append + mutate source" and
    "append + reorder existing items" regressions).
    """
    from backend.models.staging import StagingProject as _SP

    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    captured_bodies = _setup_replace_item_capture(mock_container)

    # Canonicalize the pre-call variations through Pydantic so the
    # post-call comparison sees the same canonical shape (Pydantic
    # round-trips fill defaults like ``thumbnail_url=None``,
    # ``error=None``, ``GenerationMetadata.tokens_used=None``).
    canonical_pre = json.loads(_SP(**project_data).json())["rooms"][0]["variations"]
    initial_len = len(canonical_pre)

    mock_pipeline = mock_staging_deps["pipeline"]
    mock_pipeline.process_single_variation = _empty_async_gen

    with client.stream(
        "POST",
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": "user-typed direction"},
    ) as response:
        assert response.status_code == 200, response.text
        for _ in response.iter_bytes():
            pass

    # The first replace_item call is the preflight write that appended
    # the new variation. Use the deep-copied capture so post-call
    # mutations by the (real, since the mock pipeline isn't routed for
    # all paths) pipeline rollback don't corrupt this snapshot.
    assert len(captured_bodies) >= 1
    preflight_body = captured_bodies[0]
    persisted_variations = preflight_body["rooms"][0]["variations"]

    # 1. Length grew by exactly one.
    assert len(persisted_variations) == initial_len + 1, (
        f"Expected variation count to grow from {initial_len} to {initial_len + 1}; "
        f"got {len(persisted_variations)}. Edit Prompt must APPEND, not REPLACE."
    )

    # 2. Source variation (and any other pre-existing variations) is
    #    byte-identical to its canonical pre-call state — Edit Prompt
    #    must NOT mutate or reorder the existing list. Compare against
    #    the Pydantic-canonicalized snapshot so default-filled fields
    #    don't false-positive.
    assert persisted_variations[:initial_len] == canonical_pre, (
        "Pre-existing variations must be byte-identical after Edit Prompt. "
        "Detected a mutation or reorder of the source/sibling variations."
    )

    # 3. The appended variation is fresh: distinct ID, status=processing,
    #    no image_url yet (the pipeline hasn't actually run since we
    #    stubbed it to an empty generator).
    appended = persisted_variations[-1]
    assert appended["id"] != SOURCE_VARIATION_ID, (
        "Appended variation must have a fresh UUID, not the source's ID."
    )
    assert appended["status"] == "processing", (
        "Appended variation should be PROCESSING in the preflight write so "
        "concurrent regen requests see a non-pending state."
    )
    assert not appended.get("image_url"), (
        "Appended variation should not have an image_url at preflight time "
        "(the pipeline hasn't run yet)."
    )


def test_edit_prompt_calls_composer_with_override_and_no_brief(client, mock_staging_deps):
    """Composer must be called with ``variation_override=body.adapted_prompt``
    and ``design_brief=None`` — the PRD requires bypassing
    ``BriefGeneratorService.brief_to_prompts`` entirely on the Edit
    Prompt path so the user's text is the authoritative base.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    mock_pipeline = mock_staging_deps["pipeline"]
    mock_pipeline.process_single_variation = _empty_async_gen

    with patch(
        "backend.api.endpoints.staging.PromptComposer.compose",
        return_value="COMPOSED-FROM-OVERRIDE",
    ) as compose_mock:
        with client.stream(
            "POST",
            f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
            json={"adapted_prompt": "USER-TYPED OVERRIDE"},
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert compose_mock.called, "PromptComposer.compose must be called on the Edit Prompt path"
    kwargs = compose_mock.call_args.kwargs
    assert kwargs.get("variation_override") == "USER-TYPED OVERRIDE", (
        f"compose() must receive variation_override=user-typed-text. Got: {kwargs}"
    )
    assert kwargs.get("design_brief") is None, (
        f"compose() must receive design_brief=None on Edit Prompt — the brief "
        f"is bypassed entirely per PRD § Solution → 4. Got: {kwargs}"
    )
    assert kwargs.get("project_prompt") == "modern minimalist", (
        f"compose() must receive the project's prompt for fallback. Got: {kwargs}"
    )


def test_edit_prompt_composes_room_addendum_into_final_prompt(client, mock_staging_deps):
    """When ``room.prompt_addendum`` is set, the composer's output (which
    appends the addendum to the override) must be the prompt handed to
    ``pipeline.process_single_variation``. Pins the joint behavior of
    Edit Prompt + per-room addendum — the addendum still applies even
    when the user types a custom prompt, per PRD § Further Notes.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation(with_addendum=True)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_prompts = []

    async def _capturing_psv(self, project, room, variation, adapted_prompt):
        captured_prompts.append(adapted_prompt)
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    with patch.object(StagingPipeline, "process_single_variation", _capturing_psv):
        with client.stream(
            "POST",
            f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
            json={"adapted_prompt": "USER-TYPED OVERRIDE"},
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    # The dependency-override fixture replaces get_staging_pipeline with a
    # MagicMock; patching StagingPipeline.process_single_variation at the
    # class level intercepts the call regardless of which instance is used.
    # If captured_prompts is empty, the mock pipeline override won the
    # routing — fall back to inspecting MagicMock's call_args.
    if not captured_prompts:
        mock_pipeline = mock_staging_deps["pipeline"]
        if mock_pipeline.process_single_variation.called:
            args = mock_pipeline.process_single_variation.call_args
            # Last positional arg or 'adapted_prompt' kwarg
            adapted = (
                args.kwargs.get("adapted_prompt")
                or (args.args[3] if len(args.args) > 3 else None)
            )
            captured_prompts.append(adapted)

    assert captured_prompts, "process_single_variation must be called once"
    final_prompt = captured_prompts[0]
    # The composer appends the addendum with a paragraph break to the
    # override base. Both substrings must appear.
    assert "USER-TYPED OVERRIDE" in final_prompt
    assert "always in front of fence" in final_prompt


def test_edit_prompt_does_not_call_brief_generator_service(client, mock_staging_deps):
    """``BriefGeneratorService.brief_to_prompts`` must NOT be invoked on
    the Edit Prompt path — bypassed entirely per PRD AC bullet 2.

    Patches the service's brief_to_prompts at import time so any
    accidental call surfaces as an assertion failure.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    # Add a design_brief so the worst-case scenario is covered: even
    # WITH a brief on the project, the Edit Prompt path skips it.
    project_data["design_brief"] = {
        "global_instructions": "would-be-used-on-fresh-regen-but-not-here",
        "object_palette": [],
    }
    project_data["analyses"] = [
        {"room_id": ROOM_ID, "description": "analysis", "features": []}
    ]
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    mock_pipeline = mock_staging_deps["pipeline"]
    mock_pipeline.process_single_variation = _empty_async_gen

    brief_mock = AsyncMock(return_value={ROOM_ID: ["should-not-be-called"]})
    with patch(
        "backend.core.brief_generator.BriefGeneratorService.brief_to_prompts",
        brief_mock,
    ):
        with client.stream(
            "POST",
            f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
            json={"adapted_prompt": "USER-TYPED OVERRIDE"},
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert not brief_mock.called, (
        "BriefGeneratorService.brief_to_prompts must NOT be called on the "
        "Edit Prompt path. The user's text bypasses the brief entirely."
    )


# ---------------------------------------------------------------------------
# New structured log family.
# ---------------------------------------------------------------------------


def test_edit_prompt_emits_started_log_event(client, mock_staging_deps, caplog):
    """The endpoint must emit ``staging.variation_edit_prompt.started``
    after the preflight write succeeds. Log analytics needs to count
    Edit Prompt usage separately from regen usage — the event name
    must NOT be ``staging.variation_regen.started``."""
    import logging
    caplog.set_level(logging.INFO, logger="backend.api.endpoints.staging")

    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    mock_pipeline = mock_staging_deps["pipeline"]
    mock_pipeline.process_single_variation = _empty_async_gen

    with client.stream(
        "POST",
        f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
        json={"adapted_prompt": "anything"},
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_bytes():
            pass

    started_records = [
        r for r in caplog.records
        if "staging.variation_edit_prompt.started" in r.getMessage()
    ]
    assert started_records, (
        "Expected one log line containing 'staging.variation_edit_prompt.started'; "
        f"all messages: {[r.getMessage() for r in caplog.records]}"
    )
    msg = started_records[0].getMessage()
    # Carries identifiers for forensics.
    assert PROJECT_ID in msg
    assert ROOM_ID in msg
    assert SOURCE_VARIATION_ID in msg


def test_edit_prompt_emits_completed_log_on_success(client, mock_staging_deps, caplog):
    """When the pipeline yields ``variation_completed``, the endpoint
    must emit ``staging.variation_edit_prompt.completed`` (NOT
    ``.regen.completed``) so log analytics can distinguish."""
    import logging
    caplog.set_level(logging.INFO, logger="backend.api.endpoints.staging")

    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    async def _success_psv(self, project, room, variation, adapted_prompt):
        # Mark the variation completed so the finalizer recompute treats
        # the room as completed, mirroring the real pipeline contract.
        variation.status = "completed"
        variation.image_url = "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/new.png"
        yield {
            "type": "variation_completed",
            "room_id": room.id,
            "variation_index": 1,
            "image_url": variation.image_url,
            "error": None,
            "elapsed_ms": 1234,
            "tokens_used": 5678,
            "model": "gpt-image-2",
            "adapted_prompt": adapted_prompt,
        }

    from backend.core.staging_pipeline import StagingPipeline
    with patch.object(StagingPipeline, "process_single_variation", _success_psv):
        with client.stream(
            "POST",
            f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
            json={"adapted_prompt": "anything"},
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_bytes():
                pass

    completed_records = [
        r for r in caplog.records
        if "staging.variation_edit_prompt.completed" in r.getMessage()
    ]
    assert completed_records, (
        "Expected a 'staging.variation_edit_prompt.completed' log line on success."
    )
    # Must NOT alias to the regen log family.
    assert not any(
        "staging.variation_regen.completed" in r.getMessage()
        for r in caplog.records
    ), (
        "Edit Prompt completion must not masquerade as regen completion — the "
        "log family must be distinct so analytics counts are clean."
    )


def test_edit_prompt_emits_failed_log_on_failure(client, mock_staging_deps, caplog):
    """Pipeline yields ``variation_failed`` → emit
    ``staging.variation_edit_prompt.failed``."""
    import logging
    caplog.set_level(logging.INFO, logger="backend.api.endpoints.staging")

    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    async def _fail_psv(self, project, room, variation, adapted_prompt):
        # Failure path: pipeline rolls back the appended variation. We
        # simulate that by leaving status=processing (the pre-call
        # placeholder) and yielding variation_failed. The endpoint's
        # finally block is responsible for forcing it to FAILED.
        yield {
            "type": "variation_failed",
            "room_id": room.id,
            "variation_index": 1,
            "image_url": None,
            "error": "image-gen exploded",
            "elapsed_ms": 1234,
            "tokens_used": None,
            "model": "gpt-image-2",
            "adapted_prompt": adapted_prompt,
        }

    from backend.core.staging_pipeline import StagingPipeline
    with patch.object(StagingPipeline, "process_single_variation", _fail_psv):
        with client.stream(
            "POST",
            f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
            json={"adapted_prompt": "anything"},
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_bytes():
                pass

    failed_records = [
        r for r in caplog.records
        if "staging.variation_edit_prompt.failed" in r.getMessage()
    ]
    assert failed_records, (
        "Expected a 'staging.variation_edit_prompt.failed' log line on failure."
    )


# ---------------------------------------------------------------------------
# The blocking bug the rubber-duck flagged: appended variation must
# end up FAILED after a pipeline failure, NOT stranded in PROCESSING.
# ---------------------------------------------------------------------------


def test_edit_prompt_failure_marks_appended_variation_failed_not_stranded(client, mock_staging_deps):
    """Critical regression: ``process_single_variation``'s built-in
    failure rollback restores the variation's prior status. For an
    APPENDED variation that was preset to PROCESSING, that rollback
    leaves status=PROCESSING — stranded forever.

    The endpoint's ``finally`` block must detect a non-completed
    appended variation and force it to FAILED so the room/project
    status calculator classifies the room correctly.

    Asserted by inspecting the final replace_item body — the appended
    variation's status must be ``failed`` (NOT ``processing``) after
    the endpoint's terminal write lands.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    # Track storage state across calls — the endpoint reads, the
    # pipeline (mocked) "writes" via _update_room_in_project, then the
    # finalizer reads again. Mirror that by having read_item return the
    # current persisted state from previous replace_item calls.
    persisted_state = [project_data]

    def _read_item(item, partition_key):
        return persisted_state[-1]

    def _replace_item(item, body):
        persisted_state.append(body)
        return body

    mock_container.read_item.side_effect = _read_item
    mock_container.replace_item.side_effect = _replace_item

    async def _fail_psv(self, project, room, variation, adapted_prompt):
        # Simulate the pipeline's failure rollback: it leaves the
        # variation in its prior state (PROCESSING for our appended
        # variation) and yields variation_failed. The fix in the
        # endpoint must override this to FAILED in the finalizer.
        # We DO write to storage here to mimic the pipeline's
        # _update_room_in_project call (which is what creates the
        # stranded-PROCESSING state pre-fix).
        for i, r in enumerate(project.rooms):
            if r.id == room.id:
                project.rooms[i] = room
                break
        from backend.models.staging import StagingProject as _SP
        # Project here already has the new variation appended (preflight),
        # and pipeline restored its status to processing (rollback).
        # Persist that state so the finalizer's re-read sees it.
        # (Use the raw container directly — the test isn't about the
        # storage layer's internal serialization.)
        mock_container.replace_item(item=project.id, body=json.loads(project.json()))
        yield {
            "type": "variation_failed",
            "room_id": room.id,
            "variation_index": 1,
            "image_url": None,
            "error": "synthetic image-gen failure",
            "elapsed_ms": 100,
            "tokens_used": None,
            "model": "gpt-image-2",
            "adapted_prompt": adapted_prompt,
        }

    from backend.core.staging_pipeline import StagingPipeline
    with patch.object(StagingPipeline, "process_single_variation", _fail_psv):
        with client.stream(
            "POST",
            f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
            json={"adapted_prompt": "anything"},
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_bytes():
                pass

    # Final persisted state: the appended variation's status must be
    # 'failed', not 'processing'. Inspect the last replace_item body.
    final_body = persisted_state[-1]
    final_variations = final_body["rooms"][0]["variations"]
    appended = final_variations[-1]
    assert appended["status"] == "failed", (
        f"Appended variation must be marked FAILED after a pipeline failure, "
        f"NOT stranded in '{appended['status']}'. "
        f"Without the endpoint's finalizer fix-up, the rollback in "
        f"process_single_variation leaves it in PROCESSING."
    )
    # The error message should be set so the user sees what happened.
    assert appended.get("error"), (
        "Appended variation must have an error message after failure so the "
        "frontend can render the standard FAILED thumbnail with retry."
    )


# ---------------------------------------------------------------------------
# Status recomputation via ProjectStatusCalculator.
# ---------------------------------------------------------------------------


def test_edit_prompt_finalizer_recomputes_status_via_calculator(client, mock_staging_deps):
    """The finalizer must recompute project.status via
    ``ProjectStatusCalculator.compute_status`` (issue 001) so the badge
    stays truthful after the new variation lands."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_one_completed_variation()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    async def _success_psv(self, project, room, variation, adapted_prompt):
        variation.status = "completed"
        variation.image_url = "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/new.png"
        yield {
            "type": "variation_completed",
            "room_id": room.id,
            "variation_index": 1,
            "image_url": variation.image_url,
            "error": None,
            "elapsed_ms": 100,
            "tokens_used": 200,
            "model": "gpt-image-2",
            "adapted_prompt": adapted_prompt,
        }

    from backend.core.staging_pipeline import StagingPipeline
    with patch(
        "backend.api.endpoints.staging.ProjectStatusCalculator.compute_status",
        return_value="completed",
    ) as calc_mock:
        with patch.object(StagingPipeline, "process_single_variation", _success_psv):
            with client.stream(
                "POST",
                f"/api/v1/staging/projects/{PROJECT_ID}/rooms/{ROOM_ID}/variations/{SOURCE_VARIATION_ID}/edit-prompt",
                json={"adapted_prompt": "anything"},
            ) as response:
                assert response.status_code == 200
                for _ in response.iter_bytes():
                    pass

    assert calc_mock.called, (
        "ProjectStatusCalculator.compute_status must be invoked in the "
        "edit-prompt finalizer (issue 001 single-source-of-truth pattern)."
    )
